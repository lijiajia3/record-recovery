#!/usr/bin/env python3
"""Report all eight declared contrasts; never cherry-pick completed settings."""
import hashlib,json
from pathlib import Path
from holdout_statistics import holm_adjust
from pilot import ROOT,read_json,write_json


def main():
    settings=[(dataset,model) for dataset in ['polyie','mulms']
              for model in ['deepseek-ai/DeepSeek-V3.2','Qwen/Qwen3.5-122B-A10B']]
    paths=[ROOT/f'results/{dataset}/{model.replace("/","__")}__holdout_v1__all3_summary.json'
           for dataset,model in settings]
    missing=[str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        print(json.dumps({'status':'not_ready','missing_settings':missing,'global_family_computed':False},indent=2));return
    rows=[];evidence={}
    for (dataset,model),path in zip(settings,paths):
        summary=read_json(path)
        assert (summary['dataset'],summary['model'],summary['generation_version'])==(dataset,model,'holdout_v1')
        assert summary['repeat_ids']==[0,1,2] and len(summary['primary_contrasts'])==2
        evidence[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for name,entry in summary['primary_contrasts'].items():
            rows.append(dict(setting=dataset+'|'+model,contrast=name,**entry))
    corrected=holm_adjust([row['raw_exact_p'] for row in rows])
    for row,pvalue in zip(rows,corrected):row['joint_eight_family_holm_p']=pvalue
    output={'generation_protocol':'holdout_v1','all_four_settings_required':True,
            'global_family_size':8,'missing_or_failed_settings_not_omitted':True,
            'source_summary_sha256':evidence,'contrasts':rows,
            'interpretation':'Individual two-contrast setting families are conditional claims; joint eight-contrast multiplicity is reported for broader claims. Consistency, task scope and nonrandom/public source limitations still apply.'}
    path=ROOT/'results/holdout_v1_joint_eight_contrasts.json';write_json(path,output)
    print(json.dumps({'path':str(path.relative_to(ROOT)),'contrasts':rows},indent=2))


if __name__=='__main__':main()
