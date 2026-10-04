"""Offline adversarial checks for clustered counts and randomized expectations."""
import itertools
import numpy as np
from holdout_statistics import (exact_paired_exchange,exact_randomized_expectation,
                                f1_counts,holm_adjust,paired_count_vectors,
                                paired_paper_bootstrap)


def main():
    a=np.array([[2,0,0],[3,0,0],[1,0,0]],dtype=float)
    b=np.array([[0,2,2],[0,3,3],[0,1,1]],dtype=float)
    result=exact_paired_exchange(a,b)
    assert result['enumerated_assignments']==8 and result['raw_exact_p']==.25
    assert exact_paired_exchange(a,a)['raw_exact_p']==1
    assert exact_paired_exchange(np.zeros((7,3)),np.zeros((7,3)))['raw_exact_p']==1
    repeated=exact_paired_exchange(3*a,3*b)
    assert repeated['permutation_units']==3 and repeated['raw_exact_p']==result['raw_exact_p']
    assert np.allclose(holm_adjust([.01,.03,.04]),[.03,.06,.06])
    fractional=np.array([[.5,.5,1.5],[1.25,.75,.75]])
    assert f1_counts(fractional.sum(axis=0))==3.5/7.0
    assert paired_paper_bootstrap(a,a,resamples=100)['paired_paper_bootstrap_ci95']==[0,0]
    try:paired_count_vectors(a,b+np.array([0,0,1]))
    except ValueError:pass
    else:raise AssertionError('Different gold denominators were accepted.')
    old={1,2,3,8,9};gold={1,2,3,4,5,6,7}
    additions={5:{4,10,11}};deletions={5:{1,8}};budgets={'add':{'5':2},'delete':{'5':1}}
    expected=exact_randomized_expectation(old,gold,additions,deletions,budgets)
    scores=[];tps=[]
    for plus in itertools.combinations(additions[5],2):
        for minus in itertools.combinations(deletions[5],1):
            prediction=(old-set(minus))|set(plus)
            tp=len(prediction&gold);tps.append(tp)
            scores.append(2*tp/(len(gold)+len(prediction)))
    assert abs(expected['exact_expected_f1']-np.mean(scores))<1e-12
    assert abs(expected['random_identity_tp_variance']-np.var(tps))<1e-12
    assert abs(expected['random_identity_f1_variance']-np.var(scores))<1e-12
    unchanged=exact_randomized_expectation(old,gold,{}, {},{'add':{},'delete':{}})
    assert unchanged['exact_expected_f1']==2*len(old&gold)/(len(gold)+len(old))
    print('PASS: exhaustive two-sided paper exchange, repeated-run clustering, fractional counts, Holm, paired bootstrap, and exact combinatorial random-edit mean/variance; no API or dataset reads.')


if __name__=='__main__':main()
