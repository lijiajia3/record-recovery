"""Integrate completed saved SC counts; never fit, rescore or redraw inference."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT/'manuscript/work_in_progress/expanded_science_template_full_working_assembly_20261004_v6_execution.md'
AFTER = ROOT/'manuscript/work_in_progress/expanded_completed_science_manuscript_20261004_initial.md'
BASE = ROOT/'local_cloud_analysis/SC_NATIVE_FOLD1_V5_TEST_GPU_MIGRATION_V1_20261004_score_test'
LABELS = {'aligned':'Aligned', 'permuted':'Permuted', 'ordered_context':'Ordered context', 'biaffine':'Biaffine'}
ENDS = {'R_trigger_projected_complete':'Relation', 'E_global_sharing_consistent_complete':'Event'}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def percent(row, name):
    return float(Fraction(*row[name]))*100


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |',
                      *['| '+' | '.join(str(v) for v in r)+' |' for r in rows]])


def main():
    assert not AFTER.exists()
    done=json.loads((BASE/'completed_local_action_receipt.json').read_bytes())
    assert done['status']=='completed_local_score_test'
    for n,b in done['output_bindings_before_completion'].items():
        p=ROOT/n;assert digest(p)==b['sha256'] and p.stat().st_size==b['size_bytes']
    p=json.loads((BASE/'primary_test_analysis.json').read_bytes())
    native=json.loads((BASE/'independent_native_count_replay.json').read_bytes())
    arithmetic=json.loads((BASE/'independent_statistics_replay.json').read_bytes())
    assert p['three_fit_pooled_count_scores']==native['three_fit_pooled_count_scores']
    assert arithmetic['status']=='independent_arithmetic_replay_passed_for_supplied_counts_only'
    assert p['test_source_ids']==[str(i) for i in range(1,101)] and p['seed_ids']==[20261013,20261014,20261015]
    s=BEFORE.read_text()
    old_abstract=s.split('## Abstract\n\n',1)[1].split('\n\n## Introduction',1)[0]
    abstract='''Extraction pipelines can improve aggregate scores while leaving their proposed correction mechanism or complete scientific records unresolved. We compare a fixed language-model feedback pilot with three supervised materials-text studies, retaining directed controls and native target populations. On 24 supplied-entity SciERC abstracts, a guarded pipeline improves F1 from 33.08% to 44.44%, but targeted review does not establish a benefit over generic review. Combined inputs improve against unordered mean controls in POLYIE and MuLMS; an adapted directed biaffine reference nearly matches the richer MuLMS head. On 100 SC-CoMIcs test abstracts, aligned and permuted features yield relation F1 of 22.28% and 21.64%, without an adjusted correspondence benefit. Biaffine reaches 29.10% relation and 31.41% complete-event F1, exceeding the aligned head on both fixed comparisons. Doping-trigger F1 is 93.02%, but 77 of the aligned head's 90 exact events have no annotated roles. All fits remain in source-level uncertainty calculations, and separate implementations replay native matching and inferential arithmetic. These results distinguish pipeline gains, comparator restrictions and incomplete record recovery. They support explicit attribution controls and preserve the gap between annotation agreement and a validated scientific decision.'''
    s=s.replace(old_abstract,abstract)
    result='''The complete SC-CoMIcs test contained 4,083 native text-bound occurrences, 224 directed relations and 149 events in 100 abstracts per fit. Three fits were pooled within each abstract, giving 672 relation and 447 event targets without adding independent source units. The shared detector reached 63.41% complete native typed-span F1 and 93.02% Doping-trigger F1. Every selected checkpoint and threshold preceded all 1,500 test outputs and the separate Gold-scoring action.

Aligned features achieved 22.28% relation and 22.09% globally sharing-consistent complete-event F1. Permuted features achieved 21.64% and 20.61%, and ordered context achieved 21.57% and 20.86%. Aligned minus permuted was +0.65 relation points (pointwise paired-source 95% interval, −0.21 to 1.59; Holm-six P = 0.273927) and +1.47 event points (0.29–2.89; Holm-six P = 0.256117). The ordered-context contrasts were +0.71 and +1.23 points, each with Holm-six P = 0.273927. Positive pointwise event intervals do not override the declared adjusted randomization tests. These observations do not establish a correspondence benefit or equivalence.

The adapted biaffine reference achieved 29.10% relation and 31.41% event F1. Aligned minus biaffine was −6.81 relation points (−10.72 to −2.98; Holm-six P = 0.010600) and −9.33 event points (−14.89 to −4.50; Holm-six P = 0.004680). The historically informed Holm-ten sensitivity retained P = 0.014840 and 0.006240, respectively. These are conditional comparisons of the fixed fitted procedures, not proof of optimized architecture superiority or that a particular explicit feature causes harm. Table 6 and figure 4 retain all six signed contrasts, including these adverse outcomes.

Complete-event agreement was dominated by role-free targets. The original 149 events included 27 with zero roles and 122 with roles. In a post-test partition of the selected full-population matching optimum, the aligned head recovered 77 zero-role and 13 nonempty-role instances across the three fits; biaffine recovered 76 and 52. The corresponding nonempty-role false-negative counts were 353 and 314 out of 366 repeated targets. This descriptive partition preserves the primary matching objective and denominator; it is not a rescored restricted task. Condition-relation F1 was 5.71% for aligned and 26.67% for biaffine, with no additional label-level tests. High trigger agreement therefore coexisted with poor joint role and condition recovery.'''
    s=s.replace('{{SC_RESULTS_PENDING_WHOLE1500_SOURCE_AND_SEPARATE_GOLD_REPLAY}}',result)
    outcome='''The directed SC-CoMIcs comparison further limits a feature-specific explanation. Aligned inputs had small positive differences from permuted and zero-feature inputs, but none passed the declared six-test correction. The adapted biaffine procedure performed better on both native endpoints under that same fixed family. Its distinct parameterization prevents causal attribution of the difference to explicit features alone. Most aligned event matches were zero-role annotations, and its nonempty-role and Condition counts remained low. The resulting evidence supports the controlled performance comparisons and their limits, without establishing a generally improved feedback mechanism or a scientific-use benefit.'''
    s=s.replace('{{SC_OUTCOME_PARAGRAPH_PENDING_COMPLETE_NATIVE_AND_ARITHMETIC_REPLAY}}',outcome)
    s=s.replace('All selected settings, including search-boundary choices, must be reported.','All four SC heads selected the last allowed epoch, ten, so the larger grid also leaves convergence unresolved. Selected thresholds of 0.9 and 0.95 were interior settings of the declared grids.')
    s=s.replace('Test results are pending in this assembly; no effect direction or physical-usefulness claim is inferred','No adjusted positive correspondence effect; adapted biaffine exceeds aligned on both native endpoints, while joint role recovery remains limited')
    st=p['statistics'];ci=st['paired_conditional_bootstrap'];h6={v['id']:v for v in st['holm6_primary']['entries']};h10={v['id']:v for v in st['holm10_historically_informed_sensitivity']['entries']}
    pooled=[]
    for h in p['systems']:
        for e in ENDS:
            r=p['three_fit_pooled_count_scores']['heads'][h][e];c=r['counts'];interval=ci['system_intervals'][h][e]
            pooled.append([LABELS[h],ENDS[e],c['tp'],c['fp'],c['fn'],f"{percent(r,'precision_fraction'):.2f}",f"{percent(r,'recall_fraction'):.2f}",f"{percent(r,'f1_fraction'):.2f}",f"{interval['lower']*100:.2f}–{interval['upper']*100:.2f}"])
    contrast_rows=[]
    for r in st['primary_comparisons']:
        n=r['contrast_id'];ep='Relation' if n.startswith('SC_R_') else 'Event';other=n.split('_minus_',1)[1];interval=ci['signed_difference_intervals'][n]
        contrast_rows.append([ep,'Aligned minus '+LABELS[other].lower(),f"{r['signed_aligned_minus_reference']['decimal']*100:+.2f}",f"{interval['lower']*100:.2f}–{interval['upper']*100:.2f}",f"{r['raw_p']['decimal']:.6f}",f"{h6[n]['adjusted']['decimal']:.6f}",f"{h10[n]['adjusted']['decimal']:.6f}"])
    tables='''Table 5. Complete native SC-CoMIcs primary counts. R projects event endpoints to typed triggers; E preserves the entire rooted event closure and shared identities globally. Counts pool all three fits within each of 100 abstracts, with 672 R and 447 E targets. Intervals are pointwise paired-source percentiles conditional on the retained fits. These tasks differ from the source authors' projected metrics and downstream slots.

'''+table(['Head','Endpoint','TP','FP','FN','Precision %','Recall %','F1 %','95% interval'],pooled)+'''

Table 6. Every predeclared SC aligned-minus-reference comparison. Whole-abstract exchanges retain all three fits together; Monte Carlo tails use 100,000 fixed draws and the plus-one correction. Holm-six is primary. Holm-ten adds the unchanged four earlier probabilities as a historically informed sensitivity. Positive pointwise intervals and adjusted randomization decisions are not interchangeable.

'''+table(['Endpoint','Contrast','F1 points','Paired 95% interval','Raw P','Holm6 P','Historical Holm10 P'],contrast_rows)
    s=s.replace('{{SC_MAIN_TABLES5_AND6_PENDING_ACTUAL_NATIVE_COUNTS_AND_ALL_SIX_CONTRASTS}}',tables)
    image='/Users/jiajia/Desktop/untitled folder/figures/sccomics_completed_results_20261004_v4/figure_sccomics.png'
    figure=f'''![Figure 4]({image})

**Figure 4. Correspondence controls and directed references qualify native extraction gains.** SC-CoMIcs native fold 1 contains 100 test abstracts. (A) Directed trigger-projected relations and (B) globally sharing-consistent complete events for all four heads. Diamonds show three-fit count-pooled F1; open circles retain every fitted seed. Lines are pointwise conditional 95% source-cluster intervals from the same 10,000 paired draws. (C) Every aligned-minus-reference contrast, including both negative biaffine comparisons. Probabilities and both complete corrections appear in Table 6. Fits stay together within each abstract and do not increase the number of source units. No parameter matching, author-exact replication, downstream scientific decision or physical verification is implied.'''
    s=s.replace('{{SC_FIGURE4_PENDING_ACTUAL_NATIVE_REPLAY}}',figure)
    s=s.replace('The historical two-domain archive has already passed actual fresh extraction and independent saved-output replay. The new SC-CoMIcs complete packet and its fresh native/statistical replay are still pending. Original licenses are retained separately: SC-CoMIcs text is CC BY-NC 3.0 and annotations are CC BY 4.0; the original SC-CoMIcs repository has no confirmed code redistribution license and its code is not supplied as if it had one. Availability will be updated only from the completed expanded export and replay records.','Original corpora and the source model are available from their cited releases [12,15,16,18,25]. Evaluation code, fixed choices, complete predictions, all epoch checkpoints, original phase receipts and native/statistical replay records are retained with the companion local research package. It also includes the actual editing lineage, final figures and supplied official template. This saved-output package reproduces counting and arithmetic; it does not claim a new neural-training replication or a public repository DOI. Original licenses remain separate: SC-CoMIcs text is CC BY-NC 3.0 and annotations are CC BY 4.0. The original SC-CoMIcs repository has no confirmed code redistribution license, and its code is excluded. Original MuLMS code/corpus and model-license notices are preserved without granting additional rights.')
    s=s.replace('its native and statistical replay results remain pending in this working draft.','the complete independently implemented native replay validated all counts and the primary matching witnesses, and the separate Fraction-based statistical replay reproduced all integer tails, corrections and 28 interval bounds (maximum absolute bound difference, zero). These checks concern saved outputs, not independent training or physical adjudication.')
    seed_rows=[]
    for h in ['NER',*p['systems']]:
        for seed in p['seed_ids']:
            endpoints=['T_complete_native','T_Doping_triggers'] if h=='NER' else list(ENDS)
            for e in endpoints:
                r=(p['per_seed_summaries']['NER'][str(seed)] if h=='NER' else p['per_seed_summaries']['heads'][h][str(seed)])[e];c=r['counts']
                label={'T_complete_native':'All native T','T_Doping_triggers':'Doping trigger',**ENDS}[e]
                seed_rows.append(['Shared detector' if h=='NER' else LABELS[h],seed,label,c['tp'],c['fp'],c['fn'],f"{percent(r,'precision_fraction'):.2f}",f"{percent(r,'recall_fraction'):.2f}",f"{percent(r,'f1_fraction'):.2f}"])
    labels=json.loads((ROOT/'research/sccomics_completed_tables_20261004/all_literal_T_R_E_labels.json').read_bytes())
    label_rows=[['Shared detector' if r['head']=='NER' else LABELS[r['head']],r['family'],r['literal_label'] if r['literal_label'] is not None else '<None label>',r['tp'],r['fp'],r['fn'],f"{r['f1_percent']:.2f}"] for r in labels]
    role=json.loads((ROOT/'research/sccomics_completed_tables_20261004/event_role_presence_post_test_descriptive_partition.json').read_bytes())
    checked=json.loads((ROOT/'research/sccomics_completed_tables_20261004/event_role_presence_independent_saved_native_crosscheck.json').read_bytes());assert checked['rows']==role['rows']
    role_rows=[[LABELS[r['head']],'No roles' if r['selected_primary_witness_partition']=='zero_roles' else 'Nonempty roles',r['tp'],r['fp'],r['fn']] for r in role['rows']]
    sensitivity=[]
    for h in p['systems']:
        for e,lab in [('R_full_endpoint_graph_descriptive','Full-endpoint R'),('E_root_local_descriptive','Root-local E')]:
            r=p['three_fit_pooled_count_scores']['heads'][h][e];c=r['counts'];sensitivity.append([LABELS[h],lab,c['tp'],c['fp'],c['fn'],f"{percent(r,'f1_fraction'):.2f}"])
    supplement='''The complete analyses retain all 100 abstracts, all three fitted seeds, every native target and all four heads. The independent native implementation reproduced document-level counts and validated primary witnesses against separately parsed closures. A separate statistical implementation reproduced all 100,000-draw integer tails, 10,000 common bootstrap vectors and both complete corrections; the 28 interval bounds had zero absolute difference. Shared fixed NumPy streams make these deterministic replays, not new random samples.

Table S13. Every fitted SC detector/head and endpoint. Detector T includes all native typed text-bound occurrences; Doping triggers are a descriptive subset. All heads use the same locked detector for each seed. All selected epochs are ten; detector threshold is 0.9, and every head has relation threshold 0.9 and role threshold 0.95. These are the outcomes of the complete original grids.

'''+table(['Head','Seed','Endpoint','TP','FP','FN','Precision %','Recall %','F1 %'],seed_rows)+'''

Table S14. Every literal native label in the completed T/R/E profiles, pooled across fits. The text-bound rows use the shared detector, while relation/event rows retain each head. Zero counts remain. Literal labels are kept separately and their partitions exactly sum to the complete family counts. This descriptive table introduces no label-specific P values.

'''+table(['Head','Family','Literal label','TP','FP','FN','F1 %'],label_rows)+'''

Table S15. Native reference inventory once across the 100 test abstracts, without fit repetition. The multiple-event trigger groups were inspected after completed primary scoring and retained rather than deduplicated. These abstract records do not establish independent physical experiments.

'''+table(['Record or attribute','Count','Treatment'],[
    ['All native text-bound occurrences',4083,'Complete T denominator'],['Directed relations',224,'Complete R denominator'],['Events',149,'Complete E denominator'],['Doping triggers',121,'Typed T triggers; Main and Element remain distinct'],['Zero-role events',27,'Remain legitimate native event targets'],['Nonempty-role events',122,'Remain complete event targets'],['Literal Dopant roles',117,'Exact native role labels retained'],['Literal Site roles',40,'Exact native role labels retained'],['Triggers carrying multiple events',20,'Fifteen carry two, three carry three, one carries four, one carries five; all event occurrences retained']])+'''

Table S16. Post-test role-presence partition of the selected complete-population globally consistent optimum. Gold totals are 81 zero-role and 366 nonempty-role occurrences after pooling three fits. Every head's two rows exactly sum to its original primary E counts. The separate native implementation's saved optimum and a direct original-annotation role reader give the same eight rows. No subset is rematched, no primary denominator changes and no P value is added. Agreement here does not prove partition identity for every possible equally optimal witness.

'''+table(['Head','Role presence','TP','FP','FN'],role_rows)+'''

Table S17. Saved descriptive native-metric sensitivities. Full-endpoint R requires the referenced event closure, whereas the primary R metric projects event endpoints to typed triggers. Root-local E permits each rooted match separately, whereas primary E requires one shared identity map across matched closures. These views retain the same source populations and introduce no new inferential family or replace the primary endpoints.

'''+table(['Head','Endpoint','TP','FP','FN','F1 %'],sensitivity)
    s=s.replace('{{SC_SUPPLEMENT_PENDING_ACTUAL_ENDPOINT_LABEL_SEED_COUNTS_AND_SIX_CONTRASTS}}',supplement)
    assert '{{SC_' not in s and 'still pending in this working draft' not in s
    # Preserve every earlier empirical table row; only the extension's previously
    # pending evidence-boundary row is legitimately updated.
    oldrows=Counter(x for x in BEFORE.read_text().splitlines() if x.startswith('|') and not x.startswith('| SC-CoMIcs supervised'))
    newrows=Counter(x for x in s.splitlines() if x.startswith('|'))
    assert all(newrows[row]>=n for row,n in oldrows.items())
    AFTER.write_text(s)
    record={'scope':'Actual completed SC saved-count integration; initial full manuscript before the new six editorial passes',
            'before':{'path':str(BEFORE.relative_to(ROOT)),'sha256':digest(BEFORE)},
            'after':{'path':str(AFTER.relative_to(ROOT)),'sha256':digest(AFTER)},
            'completed_action_sha256':digest(BASE/'completed_local_action_receipt.json'),
            'primary_sha256':digest(BASE/'primary_test_analysis.json'),
            'all_earlier_empirical_table_rows_preserved_with_multiplicity':True,
            'all8_primary_scores_all6_contrasts_all30_seed_endpoint_rows_all25_literal_label_rows':True,
            'adverse_biaffine_results_retained':True,'post_test_role_partition_explicitly_descriptive':True,
            'new_six_paper_edits_complete':False,'final_template_or_export_complete':False}
    (ROOT/'research/expanded_completed_initial_manuscript_assembly_actual_20261004.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'actual_completed_initial_manuscript':str(AFTER),'word_count':len(s.split()),'SC_pending_slots':0}))


if __name__=='__main__':main()
