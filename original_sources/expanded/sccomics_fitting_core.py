"""In-memory SC fitting and source inference, without dataset or Gold file I/O.

The caller supplies separately authorized TRAIN documents and verified cold
source caches. This module cannot authorize a fit or open held-out annotations.
All full candidate rows are constructed before supervised row selection.
"""
from __future__ import annotations

import hashlib
import json
import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from sccomics_source_models import (ARCHITECTURES, PAIR_OUTPUTS, TEXT_TYPES,
    donor_permutation, explicit_features, mention_matrix, pair_population,
    span_candidates, token_features)
from sccomics_training_projection import (project_pair_supervision,
    project_span_supervision, sampled_training_rows)

SEEDS = (20261013, 20261014, 20261015)
EPOCHS = 10
ACCUMULATION = 8
TRAIN_IDS = tuple(range(201, 1001))
INFERENCE_ROWS = 512


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stream_seed(value):
    return int.from_bytes(bytes.fromhex(json_digest(value))[:16], 'big')


def seed_fit(seed):
    require(type(seed) is int and seed in SEEDS, 'undeclared fit seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def document_order(seed, epoch, source_ids=TRAIN_IDS):
    require(type(seed) is int and seed in SEEDS and type(epoch) is int and
            1 <= epoch <= EPOCHS, 'undeclared seed/epoch')
    require(len(source_ids) == len(set(source_ids)) and
            all(type(i) is int for i in source_ids), 'unique integer source IDs required')
    payload = {'policy': 'sccomics_document_order_v1', 'seed': seed, 'epoch': epoch}
    rng = np.random.Generator(np.random.PCG64(stream_seed(payload)))
    return [source_ids[int(i)] for i in rng.permutation(len(source_ids))]


def lr_factor(update, *, total=1000, warmup=100):
    require(type(update) is int and type(total) is int and type(warmup) is int and
            0 < warmup < total and 1 <= update <= total, 'invalid update schedule')
    return update / warmup if update <= warmup else (total - update) / (total - warmup)


def optimizer_groups(model):
    """Exactly one explicit encoder/head x matrix/1D partition per parameter."""
    groups, names, seen = {}, {}, set()
    for name, parameter in model.named_parameters():
        require(parameter.requires_grad, 'unexpected frozen task parameter')
        require(id(parameter) not in seen, 'shared task parameter not explicitly supported')
        seen.add(id(parameter))
        encoder = name.startswith('last_layer.')
        decay = parameter.ndim > 1 and not name.endswith('.bias')
        key = encoder, decay
        groups.setdefault(key, []).append(parameter)
        names.setdefault(key, []).append(name)
    require(groups and len(seen) == len(list(model.parameters())), 'incomplete parameter partition')
    result = []
    for encoder, decay in sorted(groups):
        result.append({'params': groups[(encoder, decay)],
            'lr': 2e-5 if encoder else 1e-3,
            'base_lr': 2e-5 if encoder else 1e-3,
            'weight_decay': .01 if decay else 0.,
            'parameter_names': names[(encoder, decay)],
            'partition': ('encoder' if encoder else 'head') + ('_matrix' if decay else '_bias_1d')})
    return result


def make_optimizer(model):
    groups = optimizer_groups(model)
    return torch.optim.AdamW(groups, foreach=False)


def _validate_cache(saved):
    require(saved.get('source_only') is True and saved.get('annotation_content_used') is False,
            'source-only cache required')
    require(type(saved.get('source_id')) is int and type(saved.get('wordpieces')) is int and
            saved['wordpieces'] >= 0 and len(saved['offsets']) == saved['wordpieces'],
            'invalid source cache metadata')
    require(saved.get('chunks'), 'source chunks required')
    for chunk in saved['chunks']:
        length = chunk['end'] - chunk['start'] + 2
        require(type(chunk['start']) is int and type(chunk['end']) is int and
                0 <= chunk['start'] <= chunk['end'] <= saved['wordpieces'] and
                chunk['hidden'].shape == (1, length, 768) and
                chunk['mask'].shape == (1, 1, 1, length) and
                chunk['hidden'].dtype == torch.float32 and chunk['mask'].dtype == torch.float32 and
                bool(torch.isfinite(chunk['hidden']).all()) and bool(torch.isfinite(chunk['mask']).all()),
                'invalid source hidden/mask')


def prepared_supervision(doc, saved, *, kind, seed, epoch):
    _validate_cache(saved)
    require(saved['split'] == 'train' and saved['source_id'] in TRAIN_IDS and
            saved['source_sha256'] == hashlib.sha256(doc.source_bytes).hexdigest(),
            'only matching authorized TRAIN source may supervise a fit')
    if kind == 'span':
        candidates = span_candidates(saved['offsets'])
        targets, diagnostics = project_span_supervision(doc, candidates)
        full = {'candidates': candidates}
    else:
        require(kind in ARCHITECTURES, 'unknown fitting architecture')
        mentions, targets, diagnostics = project_pair_supervision(doc)
        pairs = pair_population(mentions)
        features = explicit_features(mentions, pairs, len(doc.source_text))
        donor = donor_permutation(mentions, saved['source_id'], saved['source_sha256'], seed)
        require(len(pairs) == len(mentions) ** 2 and features.shape ==
                (len(pairs), 23) and donor.shape == (len(pairs),), 'incomplete training source population')
        full = {'mentions': mentions, 'pairs': pairs, 'full_features': features, 'donor': donor}
        diagnostics = dict(diagnostics, donor_population_rows=len(donor),
            donor_sha256=hashlib.sha256(donor.astype('<i8').tobytes()).hexdigest(),
            donor_fixed_points=int(np.count_nonzero(donor == np.arange(len(donor)))))
    rows = sampled_training_rows(targets, seed=seed, epoch=epoch,
        source_id=saved['source_id'], task='span' if kind == 'span' else 'pair')
    return full, targets, rows, dict(diagnostics, complete_training_rows=len(targets),
        sampled_training_rows=len(rows), sampled_original_rows=rows.tolist(),
        sampled_rows_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest())


def source_loss(model, doc, saved, *, kind, seed, epoch, device):
    population, target, rows, diagnostic = prepared_supervision(doc, saved,
        kind=kind, seed=seed, epoch=epoch)
    # Even a zero-candidate source is retained in the optimizer group's document
    # divisor. Explicit zero gradients keep AdamW's update policy well-defined.
    if not len(rows):
        loss = sum(parameter.reshape(-1)[0] * 0. for parameter in model.parameters())
        return loss, dict(diagnostic, zero_candidate_document=True, train_zero_wordpiece_mentions=0)
    tokens = token_features(model.last_layer, saved, device)
    if kind == 'span':
        logits = model.spans(tokens, [population['candidates'][int(i)] for i in rows])
        empty = []
    else:
        matrix, empty = mention_matrix(tokens, saved['offsets'], population['mentions'], allow_train_zero=True)
        logits = model.pairs(matrix, population['mentions'],
            [population['pairs'][int(i)] for i in rows], len(doc.source_text),
            full_features=population['full_features'], donor=population['donor'])
    require(logits.shape == (len(rows), target.shape[1]) and bool(torch.isfinite(logits).all()),
            'nonfinite/incomplete training logits')
    expected = torch.from_numpy(target[rows]).to(device)
    loss = F.binary_cross_entropy_with_logits(logits, expected, reduction='mean')
    require(bool(torch.isfinite(loss)), 'nonfinite source loss')
    return loss, dict(diagnostic, zero_candidate_document=False,
        train_zero_wordpiece_mentions=len(empty), train_zero_wordpiece_mention_rows=list(empty))


def train_epoch(model, optimizer, load_source, *, kind, seed, epoch, device,
                log_event, source_ids=TRAIN_IDS, total_updates=1000, warmup=100):
    """The callback must validate source hashes before returning each TRAIN doc.

    Nondefault source IDs/budget exist only for isolated INVENTED CPU fixtures;
    the actual wrapper locks 800 documents, 1000 updates and ten epochs.
    """
    order = document_order(seed, epoch, source_ids)
    require(len(order) % ACCUMULATION == 0, 'complete eight-document groups required')
    model.train()
    summaries = []
    for offset in range(0, len(order), ACCUMULATION):
        group = order[offset:offset + ACCUMULATION]
        update = (epoch - 1) * (len(order) // ACCUMULATION) + offset // ACCUMULATION + 1
        factor = lr_factor(update, total=total_updates, warmup=warmup)
        for partition in optimizer.param_groups:
            partition['lr'] = partition['base_lr'] * factor
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for source_id in group:
            doc, saved = load_source(source_id)
            require(saved['source_id'] == source_id, 'TRAIN callback returned different source')
            loss, diagnostic = source_loss(model, doc, saved, kind=kind, seed=seed,
                epoch=epoch, device=device)
            total_loss += float(loss.detach().cpu())
            (loss / len(group)).backward()
            log_event({'event': 'train_document_backward_completed', 'kind': kind,
                'seed': seed, 'epoch': epoch, 'source_id': source_id, 'update': update,
                'source_loss': float(loss.detach().cpu()), 'diagnostics': diagnostic})
            del loss
        gradient_parameters = sum(p.grad is not None for p in model.parameters())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        require(bool(torch.isfinite(norm)), 'nonfinite gradient norm')
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'nonfinite fitted parameter')
        entry = {'event': 'optimizer_update_completed', 'kind': kind, 'seed': seed,
            'epoch': epoch, 'update': update, 'source_ids': group,
            'group_mean_source_loss': total_loss / len(group),
            'gradient_parameter_count': gradient_parameters, 'unclipped_l2_gradient_norm': float(norm.cpu()),
            'lr_factor': factor, 'learning_rates': [g['lr'] for g in optimizer.param_groups]}
        summaries.append(entry)
        log_event(entry)
    return summaries


def _probabilities(logits):
    require(bool(torch.isfinite(logits).all()), 'nonfinite source logits')
    values = logits.sigmoid().detach().cpu().float().numpy()
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
            'invalid/nonfinite source probabilities')
    return values


def span_probabilities(model, saved, device):
    _validate_cache(saved)
    model.eval()
    candidates = span_candidates(saved['offsets'])
    with torch.no_grad():
        tokens = token_features(model.last_layer, saved, device)
        pieces = [_probabilities(model.spans(tokens, candidates[i:i + INFERENCE_ROWS]))
                  for i in range(0, len(candidates), INFERENCE_ROWS)]
    values = np.concatenate(pieces) if pieces else np.zeros((0, len(TEXT_TYPES)), dtype=np.float32)
    require(values.shape == (len(candidates), len(TEXT_TYPES)), 'incomplete source span probabilities')
    return candidates, values


def predicted_mentions(candidates, probabilities, threshold):
    require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 <= threshold <= 1,
            'invalid source detector threshold')
    values = np.asarray(probabilities)
    require(values.shape == (len(candidates), len(TEXT_TYPES)) and np.isfinite(values).all() and
            ((values >= 0) & (values <= 1)).all(), 'complete source span probabilities required')
    return [{'type': kind, 'segments': [[int(candidate[2]), int(candidate[3])]]}
        for candidate, row in zip(candidates, values)
        for kind, value in zip(TEXT_TYPES, row) if value >= threshold]


def pair_probabilities(model, saved, mentions, *, seed, device, text_length):
    _validate_cache(saved)
    model.eval()
    # The complete donor is fixed before any computational block, even for the
    # descriptive heads which do not consume it. There is no candidate cap.
    pairs = pair_population(mentions)
    features = explicit_features(mentions, pairs, text_length)
    donor = donor_permutation(mentions, saved['source_id'], saved['source_sha256'], seed)
    with torch.no_grad():
        tokens = token_features(model.last_layer, saved, device)
        matrix, empty = mention_matrix(tokens, saved['offsets'], mentions, allow_train_zero=False)
        require(not empty, 'default source predictions may not substitute zero representations')
        pieces = [_probabilities(model.pairs(matrix, mentions, pairs[i:i + INFERENCE_ROWS],
            text_length, full_features=features, donor=donor))
            for i in range(0, len(pairs), INFERENCE_ROWS)]
    values = np.concatenate(pieces) if pieces else np.zeros((0, len(PAIR_OUTPUTS)), dtype=np.float32)
    require(values.shape == (len(mentions) ** 2, len(PAIR_OUTPUTS)), 'incomplete source pair probabilities')
    return values, {'complete_directed_rows': len(pairs),
        'donor_sha256': hashlib.sha256(donor.astype('<i8').tobytes()).hexdigest(),
        'donor_fixed_points': int(np.count_nonzero(donor == np.arange(len(donor)))),
        'source_mentions_sha256': json_digest(mentions), 'Gold_consulted': False,
        'inference_row_block_size': INFERENCE_ROWS, 'candidate_cap': None}
