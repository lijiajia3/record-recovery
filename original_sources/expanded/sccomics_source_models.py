"""Prospective SC-CoMIcs source representations, with no annotation/file loader.

This module is untrained. It does not import any previous task-fitted model or
Gold loader. The native parser and scoring contracts are separate. Geometry
permutation is over the complete N**2 population before supervised sampling.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
from torch import nn

TEXT_TYPES = tuple(sorted(('Characterization', 'Material', 'Property', 'Element',
                         'Process', 'Value', 'SC', 'Main', 'Doping')))
R_TYPES = ('Condition', 'Equivalent', 'Target')
ROLE_TYPES = ('Dopant', 'Site')
PAIR_OUTPUTS = R_TYPES + ROLE_TYPES
WIDTH, CONTENT, STRIDE = 32, 510, 384
FEATURE_DIM = 2 * len(TEXT_TYPES) + 5
ARCHITECTURES = ('aligned', 'permuted', 'ordered_context', 'biaffine')


def source_encoding(text, tokenizer):
    """Unmodified text offsets; sliding windows retain every WordPiece."""
    enc = tokenizer(text, add_special_tokens=False, truncation=False,
                    return_offsets_mapping=True)
    ids, offsets = enc['input_ids'], enc['offset_mapping']
    chunks = []
    if not ids:
        chunks.append({'start': 0, 'end': 0,
                       'ids': [tokenizer.cls_token_id, tokenizer.sep_token_id]})
    for start in range(0, len(ids), STRIDE):
        end = min(start + CONTENT, len(ids))
        chunks.append({'start': start, 'end': end,
                       'ids': [tokenizer.cls_token_id] + ids[start:end] +
                              [tokenizer.sep_token_id]})
        if end == len(ids):
            break
    return {'chunks': chunks, 'offsets': offsets, 'wordpieces': len(ids)}


def span_candidates(offsets):
    """All distinct contiguous source boundaries of up to 32 WordPieces."""
    result, seen = [], set()
    for a, (start, first_end) in enumerate(offsets):
        if start >= first_end:
            continue
        for b in range(a, min(a + WIDTH, len(offsets))):
            if offsets[b][0] >= offsets[b][1]:
                continue
            end = offsets[b][1]
            identity = int(start), int(end)
            if start < end and identity not in seen:
                seen.add(identity)
                result.append((a, b, *identity))
    return result


def mention_key(mention):
    return mention['type'], tuple(tuple(map(int, segment))
                                  for segment in mention['segments'])


def pair_population(mentions):
    # Do not query R/E labels or filter candidates to positive-eligible types.
    return [(h, t) for h in range(len(mentions)) for t in range(len(mentions))]


def donor_permutation(mentions, source_id, source_sha256, seed):
    """A reproducible bijection of complete candidate rows, independent of R/E.

    Mention types/spans may be TRAIN supervision or source-only predictions.
    They are part of the source candidate pool; R/E targets are never an input.
    Fixed points and one-row populations are retained and explicitly auditable.
    """
    keys = [mention_key(m) for m in mentions]
    payload = json.dumps({'source_id': str(source_id), 'source_sha256': source_sha256,
                          'seed': int(seed), 'ordered_mentions': keys},
                         sort_keys=True, separators=(',', ':')).encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:16], 'big'))
    return rng.permutation(len(mentions) ** 2).astype(np.int64)


def segment_union_width(segments):
    """Feature-only union length; the native identity/order is never rewritten."""
    intervals = sorted((int(a), int(b)) for a, b in segments if b > a)
    total, end = 0, None
    for a, b in intervals:
        if end is None or a >= end:
            total += b - a
        elif b > end:
            total += b - end
        end = b if end is None else max(end, b)
    return total


def explicit_features(mentions, pairs, text_length):
    result = np.zeros((len(pairs), FEATURE_DIM), dtype=np.float32)
    denominator = math.log1p(max(1, text_length))
    for row, (h, t) in enumerate(pairs):
        a, b = mentions[h], mentions[t]
        for offset, m in ((0, a), (len(TEXT_TYPES), b)):
            if m['type'] in TEXT_TYPES:
                result[row, offset + TEXT_TYPES.index(m['type'])] = 1
            # An unsupported TRAIN native type has zero type bits; it is never
            # relabeled to an available class or removed from scoring Gold.
        astart = min(s[0] for s in a['segments'])
        bstart = min(s[0] for s in b['segments'])
        delta = bstart - astart
        signed = math.copysign(math.log1p(abs(delta)), delta) / denominator
        overlap = any(max(x, u) < min(y, v) for x, y in a['segments']
                      for u, v in b['segments'])
        awidth = segment_union_width(a['segments'])
        bwidth = segment_union_width(b['segments'])
        result[row, 2 * len(TEXT_TYPES):] = [signed, abs(signed), int(overlap),
                math.log1p(awidth) / denominator, math.log1p(bwidth) / denominator]
    return result


def token_features(last_layer, saved, device):
    """Update an independently cold-started last block over source caches."""
    features, global_ids = [], []
    for chunk in saved['chunks']:
        values = last_layer(chunk['hidden'].to(device),
                            attention_mask=chunk['mask'].to(device))[0][0]
        features.append(values[1:1 + chunk['end'] - chunk['start']])
        global_ids.extend(range(chunk['start'], chunk['end']))
    if saved['wordpieces'] == 0:
        return features[0].new_zeros((0, 768))
    values = torch.cat(features, dim=0)
    indices = torch.tensor(global_ids, dtype=torch.long, device=device)
    sums = values.new_zeros((saved['wordpieces'], 768)).index_add(0, indices, values)
    counts = values.new_zeros(saved['wordpieces']).index_add(
        0, indices, values.new_ones(len(indices)))
    if not torch.all(counts > 0):
        raise ValueError('Missing source WordPiece representation')
    return sums / counts[:, None]


def mention_matrix(tokens, offsets, mentions, *, allow_train_zero=False):
    """Pool the union of native segments; no invented hull or nearby tokens."""
    pool = np.zeros((len(mentions), len(offsets)), dtype=np.float32)
    empty = []
    for row, mention in enumerate(mentions):
        covered = [i for i, (a, b) in enumerate(offsets)
                   if any(max(a, start) < min(b, end)
                          for start, end in mention['segments'])]
        if covered:
            pool[row, covered] = 1 / len(covered)
        else:
            empty.append(row)
    if empty and not allow_train_zero:
        raise ValueError('Predicted mention without source WordPiece')
    return torch.from_numpy(pool).to(tokens.device) @ tokens, empty


class SpanDetector(nn.Module):
    def __init__(self, last_layer):
        super().__init__()
        self.last_layer = last_layer
        self.norm = nn.LayerNorm(768)
        self.width = nn.Embedding(WIDTH + 1, 32)
        self.head = nn.Sequential(nn.Linear(768 * 3 + 32, 128), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(128, len(TEXT_TYPES)))

    def spans(self, tokens, candidates):
        if not candidates:
            return tokens.new_empty((0, len(TEXT_TYPES)))
        normalized = self.norm(tokens)
        sums = torch.cat([normalized.new_zeros((1, 768)), normalized.cumsum(0)])
        a = torch.tensor([c[0] for c in candidates], device=tokens.device)
        b = torch.tensor([c[1] for c in candidates], device=tokens.device)
        width = b - a + 1
        mean = (sums[b + 1] - sums[a]) / width[:, None]
        return self.head(torch.cat([normalized[a], normalized[b], mean,
                                   self.width(width)], dim=1))


class DirectedHead(nn.Module):
    def __init__(self, last_layer, architecture):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError('Unknown prospective architecture')
        self.last_layer, self.architecture = last_layer, architecture
        self.norm = nn.LayerNorm(768)
        if architecture == 'biaffine':
            self.types = nn.Embedding(len(TEXT_TYPES) + 1, 32)
            self.head = nn.Sequential(nn.Linear(800, 128), nn.GELU(), nn.Dropout(.1))
            self.tail = nn.Sequential(nn.Linear(800, 128), nn.GELU(), nn.Dropout(.1))
            self.biaffine = nn.Parameter(torch.empty(len(PAIR_OUTPUTS), 129, 129))
            for label in self.biaffine:
                nn.init.xavier_uniform_(label)
        else:
            self.content = nn.Sequential(nn.Linear(1536, 128), nn.GELU())
            self.geometry = nn.Sequential(nn.Linear(FEATURE_DIM, 32), nn.GELU())
            self.output = nn.Sequential(nn.Linear(160, 64), nn.GELU(), nn.Dropout(.1),
                                        nn.Linear(64, len(PAIR_OUTPUTS)))

    def pairs(self, matrix, mentions, pairs, text_length, *, full_features=None,
              donor=None):
        """Use donor[row] for the full row h*N+t even after pair sampling."""
        if not pairs:
            return matrix.new_empty((0, len(PAIR_OUTPUTS)))
        indices = torch.tensor(pairs, dtype=torch.long, device=matrix.device)
        normalized = self.norm(matrix)
        if self.architecture == 'biaffine':
            kinds = torch.tensor([TEXT_TYPES.index(m['type']) if m['type'] in TEXT_TYPES
                                  else len(TEXT_TYPES) for m in mentions], device=matrix.device)
            content = torch.cat([normalized, self.types(kinds)], dim=1)
            h, t = self.head(content), self.tail(content)
            h = torch.cat([h, h.new_ones((len(h), 1))], dim=1)
            t = torch.cat([t, t.new_ones((len(t), 1))], dim=1)
            return torch.einsum('ni,kij,nj->nk', h[indices[:, 0]], self.biaffine,
                                t[indices[:, 1]])
        h, t = normalized[indices[:, 0]], normalized[indices[:, 1]]
        if self.architecture == 'ordered_context':
            features = matrix.new_zeros((len(pairs), FEATURE_DIM))
        elif self.architecture == 'aligned':
            features = torch.from_numpy(explicit_features(mentions, pairs, text_length)).to(matrix.device)
        else:
            n = len(mentions)
            if full_features is None or donor is None:
                raise ValueError('Permuted head requires the complete source population')
            if full_features.shape != (n * n, FEATURE_DIM) or donor.shape != (n * n,):
                raise ValueError('Incomplete donor population')
            if not np.array_equal(np.sort(donor), np.arange(n * n)):
                raise ValueError('Donor is not a complete bijection')
            rows = np.asarray([h * n + t for h, t in pairs], dtype=np.int64)
            features = torch.from_numpy(full_features[donor[rows]]).to(matrix.device)
        return self.output(torch.cat([self.content(torch.cat([h, t], dim=1)),
                                      self.geometry(features)], dim=1))
