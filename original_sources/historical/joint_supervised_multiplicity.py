"""Predeclared global Holm4 sensitivity; refuse partial/mismatched families."""
import json
from pathlib import Path
import numpy as np
from polyie_local_adapter import ROOT,sha
from polyie_neural_family import complete_family as polyie_complete
from mulms_neural_family import complete_family as mulms_complete
from holdout_statistics import exact_paired_exchange,holm_adjust
from polyie_local_baseline_training import write

SOURCES={'polyie':('results/local_baseline/polyie_neural_family_v1/test_summary.json',14),
         'mulms':('results/local_baseline/mulms_neural_family_v1/test_summary.json',7)}
CONTRASTS=('typed_minus_mean','typed_minus_capacity_mean')


def correct_complete(reports):
    assert set(reports)==set(SOURCES),'Both domains must be complete; never drop or fabricate a failed setting'
    rows=[]
    for domain,(_,papers)in SOURCES.items():
        report=reports[domain];assert report['test_papers']==papers and report['seeds_are_independent_papers']is False
        ids=report['paper_ids'];assert len(ids)==len(set(ids))==papers
        planned=report['planned_contrasts'];assert len(planned)==2 and {c['name']for c in planned}==set(CONTRASTS)
        local_adjusted=holm_adjust([c['raw_exact_p']for c in planned])
        assert all(abs(c['holm_p_two_planned_neural_contrasts']-h)<=1e-12 for c,h in zip(planned,local_adjusted))
        methods=report['architecture_results'];assert set(methods)=={'mean','typed','capacity_mean'}
        for name in CONTRASTS:
            control=name.removeprefix('typed_minus_')
            vectors=[]
            for architecture in ['typed',control]:
                counts=methods[architecture]['pooled_by_original_paper'];assert set(counts)==set(ids)
                vectors.append(np.asarray([[counts[p][k]for k in ['tp','fp','fn']]for p in ids]))
            recomputed=exact_paired_exchange(*vectors);stored=next(c for c in planned if c['name']==name)
            assert abs(recomputed['raw_exact_p']-stored['raw_exact_p'])<=1e-12
            assert abs(recomputed['delta_f1_points']-stored['delta_f1_points'])<=1e-9
            assert stored['permutation_units']==papers and stored['enumerated_assignments']==1<<papers
            rows.append({'domain':domain,'contrast':name,'raw_exact_p':recomputed['raw_exact_p'],
                'delta_f1_points':recomputed['delta_f1_points'],'original_local_holm2_p':stored['holm_p_two_planned_neural_contrasts']})
    for row,value in zip(rows,holm_adjust([r['raw_exact_p']for r in rows])):row['global_holm4_p']=value
    return rows


def main():
    polyie_family=polyie_complete(ROOT);mulms_family=mulms_complete(ROOT)
    reports={domain:json.loads((ROOT/file).read_text())for domain,(file,_)in SOURCES.items()}
    for domain,family in [('polyie',polyie_family),('mulms',mulms_family)]:
        _,manifest=family[:2];report=reports[domain];file=SOURCES[domain][0]
        assert report['freeze_sha256']==manifest['freeze_sha256']
        assert report['generation_manifest_sha256']==sha(ROOT/Path(file).parent/'generation_manifest.json')
        assert report['paper_ids']==manifest['paper_ids']
    result={'scope':'predeclared global four-contrast supervised two-domain sensitivity, separate from original LLM family',
        'complete_family_required':True,'all_four_contrasts_present':True,'contrasts':correct_complete(reports),
        'source_summary_sha256':{domain:sha(ROOT/file)for domain,(file,_)in SOURCES.items()},
        'local_holm2_reports_modified':False,'training_seeds_are_independent_papers':False,
        'script_sha256':sha(ROOT/'src/joint_supervised_multiplicity.py')}
    write(ROOT/'results/local_baseline/joint_supervised_global_holm4.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
