"""Paper-clustered inference for frozen complete extraction count vectors.

No data loader, model request, threshold selection, or manuscript claim lives
here. Repeated generations must already be summed within each source paper.
Fractional expected randomized-control counts are retained as floats.
"""
import numpy as np


def f1_counts(counts):
    counts=np.asarray(counts,dtype=float)
    denominator=2*counts[...,0]+counts[...,1]+counts[...,2]
    return np.divide(2*counts[...,0],denominator,
                     out=np.zeros_like(denominator),where=denominator>0)


def paired_count_vectors(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or not len(a):
        raise ValueError('Need paired nonempty paper-by-3 TP/FP/FN arrays.')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any():
        raise ValueError('Counts must be finite and nonnegative.')
    if not np.allclose(a[:,0]+a[:,2],b[:,0]+b[:,2]):
        raise ValueError('Both methods must retain the same target set in every paper.')
    return a,b


def exact_paired_exchange(a,b):
    """Exhaustive joint within-paper method-label swaps; two-sided statistic.

    This tests the paired-exchangeability null for the declared benchmark.
    It is not causal random treatment assignment or a universal-paper guarantee.
    Zero-difference papers remain in scope. All repetitions on one paper move
    together and cannot increase the number of permutation units.
    """
    a,b=paired_count_vectors(a,b)
    n=len(a)
    if n>20:
        raise ValueError('Exact enumeration limited to 20 paper units; do not silently substitute Monte Carlo.')
    observed=float(f1_counts(a.sum(axis=0))-f1_counts(b.sum(axis=0)))
    total=1<<n;extreme=0
    pooled=(a+b).sum(axis=0);shift=a-b
    for start in range(0,total,4096):
        bits=((np.arange(start,min(start+4096,total),dtype=np.uint64)[:,None]
                >>np.arange(n,dtype=np.uint64))&1).astype(float)
        left=b.sum(axis=0)+bits@shift
        values=f1_counts(left)-f1_counts(pooled-left)
        extreme+=int(np.count_nonzero(np.abs(values)>=abs(observed)-1e-12))
    return {'delta_f1_points':100*observed,'raw_exact_p':extreme/total,
            'permutation_units':n,'enumerated_assignments':total,
            'extreme_assignments':extreme,'two_sided':True,
            'null':'Within-source-paper paired method-label exchangeability',
            'repeated_runs_are_separate_units':False}


def paired_paper_bootstrap(a,b,*,resamples=10000,seed=20261003):
    a,b=paired_count_vectors(a,b);rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(a),size=(resamples,len(a)))
    differences=100*(f1_counts(a[indices].sum(axis=1))-f1_counts(b[indices].sum(axis=1)))
    return {'paired_paper_bootstrap_ci95':np.quantile(differences,[.025,.975]).tolist(),
            'resamples':resamples,'seed':seed,'n_source_papers':len(a)}


def holm_adjust(pvalues):
    values=np.asarray(pvalues,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError('P values must be a finite vector in [0,1].')
    order=np.argsort(values,kind='stable');adjusted=np.empty_like(values);maximum=0.0
    for rank,index in enumerate(order):
        maximum=max(maximum,(len(values)-rank)*values[index])
        adjusted[index]=min(1.0,maximum)
    return adjusted.tolist()


def exact_randomized_expectation(old,gold,add_pools,delete_pools,budgets):
    """Evaluate uniform fixed-budget edit identities, never use gold to select.

    Caller supplies source-only, deduplicated, disjoint confidence-stratum pools.
    Exact expectation is a secondary check of the fixed 100-seed primary control.
    F1 is linear in TP here because the final prediction/target counts are fixed.
    """
    old,gold=set(old),set(gold)
    if any(set(pool)&old for pool in add_pools.values()):
        raise ValueError('Eligible additions must be absent from initial graph.')
    if any(not set(pool)<=old for pool in delete_pools.values()):
        raise ValueError('Eligible deletions must belong to initial graph.')
    mean_tp=float(len(old&gold));variance_tp=0.0;final_n=len(old)
    for operation,pools,sign in [('add',add_pools,1),('delete',delete_pools,-1)]:
        seen=set()
        for score,pool in pools.items():
            pool=set(pool)
            if seen&pool:raise ValueError('An identity cannot belong to two confidence strata.')
            seen|=pool
            k=int(budgets[operation].get(str(score),0));n=len(pool)
            if not 0<=k<=n:raise ValueError('Edit budget exceeds its source-only pool.')
            final_n+=sign*k
            if n:
                probability=len(pool&gold)/n
                mean_tp+=sign*k*probability
                if n>1:variance_tp+=k*probability*(1-probability)*(n-k)/(n-1)
        if any(int(k) and str(score) not in {str(s) for s in pools}
               for score,k in budgets[operation].items()):
            raise ValueError('Nonzero edit budget has no eligible pool.')
    counts=np.array([mean_tp,final_n-mean_tp,len(gold)-mean_tp])
    denominator=len(gold)+final_n
    return {'expected_tp':float(counts[0]),'expected_fp':float(counts[1]),
            'expected_fn':float(counts[2]),'exact_expected_f1':float(f1_counts(counts)),
            'fixed_final_prediction_count':final_n,'random_identity_tp_variance':variance_tp,
            'random_identity_f1_variance':4*variance_tp/denominator**2 if denominator else 0.0}

