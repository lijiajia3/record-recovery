"""Invented Holm4 checks, including no dropped/tampered/local family values."""
import copy
import numpy as np
from joint_supervised_multiplicity import correct_complete
from holdout_statistics import exact_paired_exchange,holm_adjust


def main():
    reports={}
    for domain,papers in [('polyie',14),('mulms',7)]:
        ids=[str(i)for i in range(papers)];methods={}
        for name,tp in [('typed',2),('mean',1),('capacity_mean',0)]:
            methods[name]={'pooled_by_original_paper':{p:{'tp':tp,'fp':1,'fn':3-tp}for p in ids}}
        planned=[]
        for control in ['mean','capacity_mean']:
            a=np.array([[2,1,1]]*papers);b=np.array([[1,1,2]if control=='mean'else[0,1,3]]*papers)
            planned.append(dict(name='typed_minus_'+control,**exact_paired_exchange(a,b)))
        for row,h in zip(planned,holm_adjust([r['raw_exact_p']for r in planned])):row['holm_p_two_planned_neural_contrasts']=h
        reports[domain]={'test_papers':papers,'paper_ids':ids,'architecture_results':methods,
            'planned_contrasts':planned,'seeds_are_independent_papers':False}
    rows=correct_complete(reports)
    assert len(rows)==4 and rows[0]['global_holm4_p']==8/16384 and rows[2]['global_holm4_p']==4/128
    dropped={k:v for k,v in reports.items()if k=='polyie'}
    raw=copy.deepcopy(reports);raw['mulms']['planned_contrasts'][0]['raw_exact_p']=0
    local=copy.deepcopy(reports);local['mulms']['planned_contrasts'][0]['holm_p_two_planned_neural_contrasts']=0
    changed=copy.deepcopy(reports);changed['mulms']['architecture_results']['typed']['pooled_by_original_paper']['0']['tp']=1
    changed['mulms']['architecture_results']['typed']['pooled_by_original_paper']['0']['fn']=2
    for bad in [dropped,raw,local,changed]:
        try:correct_complete(bad)
        except AssertionError:pass
        else:raise AssertionError('Dropped or altered global/local/count contrast accepted')
    print('PASS: invented global Holm4, fixed paper units, all four required; altered counts/raw P/local Holm rejected')


if __name__=='__main__':main()
