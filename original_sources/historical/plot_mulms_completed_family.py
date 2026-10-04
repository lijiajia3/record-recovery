"""Plot completed, independently replayed MuLMS scores without reading Gold."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = 'results/local_baseline/mulms_neural_family_v1'
ORDER = ('mean', 'capacity_mean', 'typed', 'ordered_context', 'biaffine')
NAMES = ('Mean\nlinear', 'Mean\nmatched', 'Role +\nfeatures', 'Ordered\ncontext', 'Classical\nbiaffine')
COLORS = ('#737B86', '#416E99', '#C25D30', '#357D72', '#865D8C')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def reports():
    # The read-only replay is a separate prerequisite, never launched here.
    audit = read('research/mulms_neural_independent_replay.json')
    assert audit['matches_saved_summaries'] and audit['all_eighteen_test_graphs_before_replay_Gold']
    assert audit['replay_source_sha256'] == digest(ROOT / 'src/replay_mulms_neural_offline.py')
    for name, expected in audit['source_summary_sha256'].items():
        assert digest(ROOT / name) == expected
    main = read(DEST + '/test_summary.json')
    assert main['complete_detector_graph_barrier'] == 3 and main['complete_relation_graph_barrier'] == 9
    assert main['test_papers'] == 7 and main['test_sentences'] == 1114
    assert main['seeds_are_independent_papers'] is False
    values = dict(main['architecture_results'])
    for architecture in ('ordered_context', 'biaffine'):
        value = read('results/local_baseline/mulms_' + architecture + '_reference_test_v1/test_summary.json')
        assert value['paper_ids'] == main['paper_ids']
        assert value['fresh_independent_holdout_after_primary_score_claimed'] is False
        assert value['planned_primary_holm2_and_global4_modified'] is False
        values[architecture] = value
    global_report = read('results/local_baseline/joint_supervised_global_holm4.json')
    assert len(global_report['contrasts']) == 4
    assert global_report['all_four_contrasts_present']
    return main, values, global_report, audit


def f1(c):
    denominator = 2 * c['tp'] + c['fp'] + c['fn']
    return 100 * 2 * c['tp'] / denominator if denominator else 0.


def main():
    report, values, global_report, audit = reports()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(6.4, 5.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1.15, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    ax = axes[0]
    high_points = []
    for x, (name, color) in enumerate(zip(ORDER, COLORS)):
        value = values[name]
        low, high = np.asarray(value['paper_bootstrap_95_f1_ci']) * 100
        seeds = [f1(value['all_seeds'][str(seed)]['total']) for seed in (20261003, 20261004, 20261005)]
        ax.scatter(x + np.asarray([-.12, 0, .12]), seeds, s=17, facecolors='none', edgecolors=color, linewidths=.8, zorder=2)
        ax.vlines(x, low, high, color=color, lw=1.8, zorder=3)
        ax.plot(x, f1(value['pooled_primary']), 's', color=color, markersize=4.5, zorder=4)
        high_points.extend([high, *seeds])
    ax.axvspan(2.5, 4.5, color='#EDEEEF', alpha=.6, zorder=0)
    ax.set_xlim(-.5, 4.5)
    ax.set_xticks(range(5), NAMES, fontsize=6.8)
    ax.set_ylabel('Strict end-to-end relation micro-F1 (%)')
    ax.set_ylim(0, max(5., max(high_points) * 1.15))
    ax.set_title('a   All fitted seeds and heads', loc='left', fontweight='bold')
    ax.text(0, -.32, 'Squares: pooled counts; circles: three fitted seeds\nLines: paper CI; shade: descriptive references', transform=ax.transAxes, fontsize=6.8)
    ax = axes[1]
    ax.axvline(0, color='#A0A0A0', lw=.8, ls='--')
    for y, contrast in enumerate(report['planned_contrasts']):
        lo, hi = contrast['paired_paper_bootstrap_ci95']
        ax.hlines(y, lo, hi, color=COLORS[2], lw=1.8)
        ax.plot(contrast['delta_f1_points'], y, 's', color=COLORS[2], ms=4.5)
        global_value = next(r['global_holm4_p'] for r in global_report['contrasts']
            if r['domain'] == 'mulms' and r['contrast'] == contrast['name'])
        ax.annotate('Global Holm p = ' + format(global_value, '.3g'), (contrast['delta_f1_points'], y),
            xytext=(0, 10), textcoords='offset points', ha='center', fontsize=6.8)
    ax.set_yticks([0, 1], ['vs mean\nlinear', 'vs mean\nmatched'])
    ax.set_ylim(-.65, 1.65)
    ax.set_xlabel('Role/features minus control (F1 points)', fontsize=7.4)
    ax.set_title('b   Two planned contrasts', loc='left', fontweight='bold')
    ax = axes[2]
    papers = report['paper_ids']
    x = np.arange(len(papers))
    for offset, reference, color in ((-.18, 'ordered_context', COLORS[3]), (.18, 'biaffine', COLORS[4])):
        differences = [f1(values['typed']['pooled_by_original_paper'][p]) - f1(values[reference]['pooled_by_original_paper'][p]) for p in papers]
        ax.bar(x + offset, differences, width=.32, color=color, label='vs ' + reference.replace('_', ' '))
    ax.axhline(0, color='#A0A0A0', lw=.8)
    ax.set_xticks(x, papers, fontsize=6.2)
    ax.set_xlabel('Original source paper; descriptive reused-holdout comparisons')
    ax.set_ylabel('Directed-reference difference\n(F1 points)')
    ax.set_title('c   Differences from directed references', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=7, loc='best')
    fig.subplots_adjust(left=.1, right=.985, bottom=.12, top=.94, wspace=.62, hspace=.9)
    dest = ROOT / 'figures'
    paths = []
    for extension in ('png', 'pdf', 'svg'):
        path = dest / ('figure5_mulms_end_to_end.' + extension)
        assert not path.exists(), 'Preserve completed figure; edit under a new audited version'
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    source = dest / 'figure5_mulms_end_to_end_source.csv'
    assert not source.exists()
    with source.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('architecture', 'seed_or_pooled', 'paper', 'tp', 'fp', 'fn', 'f1'))
        for name in ORDER:
            for seed, value in values[name]['all_seeds'].items():
                for paper, c in value['by_paper'].items():
                    writer.writerow((name, seed, paper, c['tp'], c['fp'], c['fn'], f1(c) / 100))
            for paper, c in values[name]['pooled_by_original_paper'].items():
                writer.writerow((name, 'pooled', paper, c['tp'], c['fp'], c['fn'], f1(c) / 100))
    paths.append(source)
    manifest = dest / 'figure5_mulms_end_to_end_manifest.json'
    assert not manifest.exists()
    manifest.write_text(json.dumps({'source_summary_sha256': audit['source_summary_sha256'],
        'independent_replay_sha256': digest(ROOT / 'research/mulms_neural_independent_replay.json'),
        'plot_source_sha256': digest(Path(__file__).resolve()),
        'files': {str(p.relative_to(ROOT)): digest(p) for p in paths},
        'bootstrap_unit': 'original paper', 'bootstrap_conditional_on_fitted_seeds': True,
        'directed_reference_differences_descriptive_no_additional_P': True,
        'new_tests_created_by_figure': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
