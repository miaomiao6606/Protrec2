import pandas as pd
import numpy as np
from .utils import safe_logit, safe_expit

def PROTREC_complex(data, cplx, cplx_key, fdr, fnr, threshold, initial_prob):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    prot_list = sorted(list(set([protein for sublist in cplx for protein in sublist])))
    pri = {protein: 1 - fdr if protein in data.index else fdr for protein in prot_list}
    size_complex = len(cplx[cplx_key])
    n = sum(pri.get(p, 0) for p in cplx[cplx_key])
    m = max(threshold, size_complex)
    p_cpx = min(1 - fdr, (n / m * (1 - fdr) + initial_prob[cplx_key]) / 2)
    return p_cpx

def PROTREC_protprob_update(protein, cplx, complex_key, fdr, prot_prob, complex_prob):
    size_complex = len(cplx[complex_key])
    other_proteins = [p for p in cplx[complex_key] if p != protein]
    sum_probs = sum(prot_prob[p] for p in other_proteins)
    p_cpx = min(1 - fdr, ((sum_probs + 1 - fdr) / size_complex))
    return p_cpx

def PROTREC_protprob_bayesian(data, cplx, meanp, fdr, fnr, threshold, initial_prob, max_iter=1, eps=1e-3):
    prot_list = sorted(list(set([protein for sublist in cplx for protein in sublist])))
    unob = 0.1
    old_prob = {protein: 1 - fdr if protein in data.index else unob for protein in prot_list}

    for iteration in range(max_iter):
        new_prob = {}
        for protein in prot_list:
            max_pxcp = 0
            for complex_key in cplx.keys():
                if protein not in cplx[complex_key]:
                    continue
                p_cp = initial_prob[complex_key]
                if abs(old_prob[protein] - (1 - fdr)) < eps:
                    max_pxcp = max(max_pxcp, 1 - fdr)
                elif protein in data.index:
                    p_obs = p_cp + (1 - fdr) * (1 - p_cp)
                    max_pxcp = max(max_pxcp, p_obs)
                else:
                    p_cpx = PROTREC_protprob_update(protein, cplx, complex_key, fdr, old_prob, p_cp)
                    log_prior = safe_logit(old_prob[protein])
                    log_support = safe_logit(p_cpx)
                    log_complex = safe_logit(p_cp)
                    combined_logit = (1 / 3) * log_prior + (1 / 3) * log_support + (1 / 3) * log_complex
                    p_xcp = safe_expit(combined_logit)
                    p_xcp = min(p_xcp, 1 - fdr)
                    max_pxcp = max(max_pxcp, p_xcp)
            new_prob[protein] = max(old_prob[protein], max_pxcp)

        delta = sum(abs(new_prob[p] - old_prob[p]) for p in prot_list)
        #print(f"Iteration {iteration + 1}, total change: {delta:.6f}")
        if delta < eps:
            break
        old_prob = new_prob.copy()

    probs = pd.DataFrame(list(new_prob.items()), columns=['Protein', 'Probability'])
    mask = probs['Probability'] < 1 - fdr
    mean_prob = probs.loc[mask, 'Probability'].mean()
    std_prob = probs.loc[mask, 'Probability'].std()
    probs.loc[mask, 'Probability'] = (probs.loc[mask, 'Probability'] - mean_prob) / std_prob
    new_min = meanp
    new_max = 1 - fdr
    probs.loc[mask, 'Probability'] = ((probs.loc[mask, 'Probability'] - probs.loc[mask, 'Probability'].min()) /
                                      (probs.loc[mask, 'Probability'].max() - probs.loc[mask, 'Probability'].min())) * (new_max - new_min) + new_min
    return probs
