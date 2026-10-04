"""Supplementary reverse-closure diagnostic, guarded by complete source graphs."""
import json
import copy
from pathlib import Path
from mulms_neural_family import ROOT, FREEZE_REL, DEST_REL, SEEDS, ARCHITECTURES, sha, complete_family, span_key
from mulms_neural_family_test_analysis import load_targets
from mulms_experiment import canonical, valid_key
from directed_symmetry_ceiling import ceiling, graph_diagnostic, counts_f1


def main():
    dest = ROOT / DEST_REL / 'descriptive_direction_symmetry.json'
    assert not dest.exists(), 'Never replace a supplementary diagnostic'
    review_path = ROOT / 'research/mulms_direction_symmetry_source_review.json'
    reviewed = json.loads(review_path.read_text())['file_sha256']
    assert reviewed
    for name, expected in reviewed.items():
        assert sha(ROOT / name) == expected, name
    freeze, manifest, inputs, detectors, graphs = complete_family(ROOT)
    gold, _ = load_targets(freeze, inputs, ROOT)
    output = {}
    for seed in SEEDS:
        seed_output = {'architectures': {a: {'valid_non_diagonal_predicted_edges': 0,
            'edges_missing_same_label_reverse': 0, 'invalid_including_diagonal_predictions': 0}
            for a in ARCHITECTURES}, 'full_gold_entity_oracle_counts': {'tp': 0, 'fp': 0, 'fn': 0},
            'predicted_entity_oracle_counts': {'tp': 0, 'fp': 0, 'fn': 0}, 'by_original_paper': {}}
        for inp in inputs:
            identity = inp['id']
            targets = {canonical(r, inp['text']) for r in gold[identity]}
            assert all(valid_key(e) for e in targets)
            available = {span_key(e) for e in detectors[seed]['graphs'][identity]}
            full, reachable = ceiling(targets), ceiling(targets, available)
            paper = seed_output['by_original_paper'].setdefault(inp['doc_key'], {
                name: {'tp': 0, 'fp': 0, 'fn': 0} for name in
                ('full_gold_entity_oracle_counts', 'predicted_entity_oracle_counts')})
            paper.setdefault('architectures', {a: {k: 0 for k in seed_output['architectures'][a]} for a in ARCHITECTURES})
            for name, result in [('full_gold_entity_oracle_counts', full), ('predicted_entity_oracle_counts', reachable)]:
                for k, count in result['oracle_counts'].items():
                    seed_output[name][k] += count
                    paper[name][k] += count
            for architecture in ARCHITECTURES:
                predicted = {canonical(r, inp['text']) for r in graphs[architecture, seed]['graphs'][identity]}
                valid = {e for e in predicted if valid_key(e)}
                row = seed_output['architectures'][architecture]
                for k, count in graph_diagnostic(valid).items():
                    row[k] += count
                    paper['architectures'][architecture][k] += count
                row['invalid_including_diagonal_predictions'] += len(predicted - valid)
                paper['architectures'][architecture]['invalid_including_diagonal_predictions'] += len(predicted - valid)
        for name in ('full_gold_entity_oracle_counts', 'predicted_entity_oracle_counts'):
            c = seed_output[name]
            seed_output[name + '_f1'] = counts_f1(c['tp'], c['fp'], c['fn'])
        output[str(seed)] = seed_output
    pooled = copy.deepcopy(output[str(SEEDS[0])]['by_original_paper'])
    for seed in SEEDS[1:]:
        for paper, values in output[str(seed)]['by_original_paper'].items():
            for name in ('full_gold_entity_oracle_counts', 'predicted_entity_oracle_counts'):
                for k, count in values[name].items():
                    pooled[paper][name][k] += count
            for architecture, counts in values['architectures'].items():
                for k, count in counts.items():
                    pooled[paper]['architectures'][architecture][k] += count
    assert set(pooled) == set(manifest['paper_ids']) and len(pooled) == 7
    totals = {}
    for name in ('full_gold_entity_oracle_counts', 'predicted_entity_oracle_counts'):
        c = {k: sum(p[name][k] for p in pooled.values()) for k in ('tp', 'fp', 'fn')}
        totals[name] = dict(c, f1=counts_f1(c['tp'], c['fp'], c['fn']))
    totals['architectures'] = {a: {k: sum(p['architectures'][a][k] for p in pooled.values())
        for k in output[str(SEEDS[0])]['architectures'][a]} for a in ARCHITECTURES}
    report = {'scope': 'Gold-aware supplementary direction-invariance diagnostic; no implementable oracle or new P',
        'all_seeds': output, 'pooled_by_original_paper': pooled, 'pooled_counts': totals,
        'test_papers': 7, 'test_sentences': 1114, 'seeds_are_independent_paper_units': False,
        'freeze_sha256': sha(ROOT / FREEZE_REL),
        'generation_manifest_sha256': sha(ROOT / DEST_REL / 'generation_manifest.json'),
        'protocol_sha256': sha(ROOT / 'research/mulms_direction_symmetry_diagnostic_protocol.md'),
        'independent_source_review_sha256': sha(review_path),
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
            [Path(__file__).resolve(), ROOT / 'src/directed_symmetry_ceiling.py']},
        'gold_loaded_after_all_three_detector_and_nine_relation_graphs': True,
        'primary_comparisons_thresholds_and_holm_families_changed': False,
        'oracle_is_actual_implementable_method': False}
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({s: {'architectures': v['architectures'],
        'predicted_entity_oracle_f1': v['predicted_entity_oracle_counts_f1']} for s, v in output.items()}, indent=2))


if __name__ == '__main__':
    main()
