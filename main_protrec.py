# -*- coding: utf-8 -*-
# main_protrec.py

"""
End-to-end PROTREC workflow ,using HeLa as an exmple.

Steps
-----
1. Load  ➜ MS1/2 intensities, percolator PSMs, reference DIA/proteomics lists.
2. Build ➜ tissue-specific complex graph.
3. Compute ➜ per-complex priors via Bayesian / Random-Forest hybrid.
4. Infer  ➜ per-protein probabilities.
5. Output ➜ Final scores based on graph features + complex priors.
-----

"""

# ================================================================ #
# 0. CONFIG & IMPORTS
# ================================================================ #
from pathlib import Path
from functools import reduce
import re, ast, csv
import numpy as np
import pandas as pd
import scipy.stats
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics  import mean_squared_error
from sklearn.model_selection import train_test_split

from complex_data import (complex_list, initial_prob,
                          complex_list1, initial_prob1)

ROOT           = Path(r"C:")
HELA_CSV       = ROOT / "processed_hela.csv"
PERCOLATOR_TSV = Path(r"C:"
                      r"\New_Hela_100ng_Slot2-39_1_811_percolator_target_psms.tsv")
PPIN_FILE      = ROOT / "uterine_cervix_top_task_1.csv"

# extra DIA / literature references used to build all_dia


# PROTREC / model hyper-params
FDR       = 0.01
FNR       = 0
THRESHOLD = 5
RAND_SEED = 42

# ================================================================ #
# 1. RAW DATA
# ================================================================ #
# HeLa intensities
HeLa = pd.read_csv(HELA_CSV)
HeLa.set_index(HeLa.columns[0], inplace=True)
HeLa.columns = ["811", "812", "813"]
HeLa.index.name = "Index"

# Percolator validated proteins
per  = pd.read_table(PERCOLATOR_TSV).iloc[:, [2, 5]]
per["proteinIds"] = per["proteinIds"].str.split(";").explode()
per  = per[per["proteinIds"].str.startswith("sp")]
per  = per[per.iloc[:, 1].astype(float) < 0.1]
per_unique = per["proteinIds"].str.split("|").str[1].unique()
hela_validate = per_unique

# DIA / literature references 

all_dia = []

# tissue-specific PPIN graph
ppin_df = pd.read_csv(PPIN_FILE)
G = nx.from_pandas_edgelist(ppin_df,
        source="Protein1", target="Protein2",
        edge_attr="score", create_using=nx.Graph())

# ================================================================ #
# 2. PROTREC CORE FUNCTIONS  (unchanged from your script)
# ================================================================ #
def PROTREC_protprob_update(protein, cplx, k, fdr, prot_prob,
                            complex_prob, validate_list):
    size = len(cplx[k]) * (1 - FNR)
    others = [p for p in cplx[k] if p != protein]
    sum_other = sum(prot_prob[p] if p not in validate_list else 1 - fdr
                    for p in others)
    return min(1 - fdr, (sum_other + 1 - fdr) / size)

def PROTREC_complex_update(cplx, k, fdr, init, prot_prob):
    return min(1 - fdr, init[k])

def PROTREC_cplx_prob(data, cplx, fdr, fnr, thr):
    res, detected = [], data.index.tolist()
    for members in cplx:
        if len(members) >= thr:
            n = len(set(detected).intersection(members))
            m = max(thr, len(members)) * (1 - fnr)
            res.append(min(1, n / m) * (1 - fdr))
        else:
            res.append(fdr)
    return res

def _rescale(df, low, high):
    mask = df["Probability"] < 1 - FDR
    m, s = df.loc[mask, "Probability"].mean(), df.loc[mask, "Probability"].std()
    df.loc[mask, "Probability"] = (df.loc[mask, "Probability"] - m) / s
    rng = df.loc[mask, "Probability"].max() - df.loc[mask, "Probability"].min()
    if rng == 0: rng = 1
    df.loc[mask, "Probability"] = (
        (df.loc[mask, "Probability"] - df.loc[mask, "Probability"].min()) / rng
    ) * (high - low) + low

def PROTREC_protprob_bayesian(data, cplx, meanp, fdr, fnr,
                              thr, validate, init, max_iter=1, eps=1e-1):
    prot_set = sorted({p for sub in cplx for p in sub})
    pri = {p: 1 - fdr if p in data.index else 0.1 for p in prot_set}
    old = pri.copy()
    cprob = PROTREC_cplx_prob(data, cplx, fdr, fnr, thr)
    for _ in range(max_iter):
        new = old.copy()
        for k in cplx.keys():
            p_cp = PROTREC_complex_update(cplx, k, fdr, init, old)
            for p in cplx[k]:
                if abs(pri[p] - (1 - fdr)) < eps:
                    new[p] = 1 - fdr
                else:
                    tmp = (min(1 - fdr,
                               1 - cprob[k] + (1 - fdr)*cprob[k])
                           if p in data.index else
                           min(1 - fdr, cprob[k]))
                    p_cpx = PROTREC_protprob_update(
                        p, cplx, k, fdr, old, p_cp, validate)
                    p_xcp = min(1 - fdr,
                                (tmp + p_cpx)/2 * old[p] / min(p_cp, meanp))
                    new[p] = max(new[p], p_xcp)
        old = new.copy()
    df = pd.DataFrame(list(old.items()), columns=["Protein", "Probability"])
    _rescale(df, meanp, 1 - fdr)
    return df

def PROTREC_complex(data, cplx, k, fdr, fnr, thr, init):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    prot_set = sorted({p for sub in cplx for p in sub})
    pri = {p: 1 - fdr if p in data.index else fdr for p in prot_set}
    size = len(cplx[k])
    n = sum(pri.get(p, 0) for p in cplx[k])
    m = max(thr, size)
    return min(1 - fdr, (n/m*(1 - fdr) + init[k]) / 2)

# ================================================================ #
# 3. GRAPH FEATURES
# ================================================================ #
def extract_features(cmplx, graph):
    sg = graph.subgraph(cmplx)
    scores = [d["score"] for _, _, d in sg.edges(data=True)]
    cc = nx.clustering(sg)
    avg_clust = np.mean(list(cc.values())) if cc else 0
    if scores:
        return dict(
            avg_score=np.mean(scores),
            max_score=np.max(scores),
            min_score=np.min(scores),
            std_score=np.std(scores),
            skewness=scipy.stats.skew(scores),
            kurtosis=scipy.stats.kurtosis(scores),
            avg_clustering_coeff=avg_clust)
    return dict.fromkeys(
        ["avg_score", "max_score", "min_score", "std_score",
         "skewness", "kurtosis", "avg_clustering_coeff"], 0)

def calc_edge_features(cmplx, graph):
    sg = graph.subgraph(cmplx)
    edges = [(u, v, d["score"]) for u, v, d in sg.edges(data=True)]
    overall = np.mean([s for *_, s in edges]) if edges else 0
    p_scores = {}
    for u, v, s in edges:
        p_scores.setdefault(u, []).append(s)
        p_scores.setdefault(v, []).append(s)
    p_means = {p: np.mean(s) for p, s in p_scores.items()}
    return overall, p_means

# ================================================================ #
# 4. MAIN PROCESS (replicates 0 – 2)
# ================================================================ #
for i in range(3):
    print(f"\n>>> replicate {i}  ({HeLa.columns[i]})")
    col_series = HeLa.iloc[:, i]
    filtered   = col_series[col_series > 0]

    # (1) refresh priors for each complex
    for k in complex_list.keys():
        initial_prob.loc[k] = PROTREC_complex(
            filtered, complex_list, k, FDR, FNR, THRESHOLD, initial_prob)
    for k in complex_list1.keys():
        initial_prob1.loc[k] = PROTREC_complex(
            filtered, complex_list1, k, FDR, FNR, THRESHOLD, initial_prob1)

    # (2) Random-Forest per complex set
    feat1 = pd.DataFrame([extract_features(c, G) for c in complex_list1])
    rf1   = RandomForestRegressor(random_state=RAND_SEED)
    rf1.fit(feat1.values, initial_prob1.values)
    pred1, meanp1 = rf1.predict(feat1.values), np.mean(rf1.predict(feat1.values))

    feat  = pd.DataFrame([extract_features(c, G) for c in complex_list])
    rf    = RandomForestRegressor(random_state=RAND_SEED)
    rf.fit(feat.values, initial_prob.values)
    pred,  meanp  = rf.predict(feat.values), np.mean(rf.predict(feat.values))

    # (3) aggregate complex → protein scores
    comp1 = pd.DataFrame({
        "Protein":  [p for sub in complex_list1 for p in sub],
        "ComplexID":[idx for idx, sub in enumerate(complex_list1) for p in sub]
    })
    comp1["ComplexProb"] = comp1["ComplexID"].apply(lambda x: pred1[x])
    protein_cpx1 = comp1.groupby("Protein").agg(
        MaxComplexProb=("ComplexProb", "max"),
        AvgComplexProb=("ComplexProb", "mean")).reset_index()

    comp = pd.DataFrame({
        "Protein":  [p for sub in complex_list for p in sub],
        "ComplexID":[idx for idx, sub in enumerate(complex_list) for p in sub]
    })
    comp["ComplexProb"] = comp["ComplexID"].apply(lambda x: pred[x])
    protein_cpx = comp.groupby("Protein").agg(
        MaxComplexProb=("ComplexProb", "max"),
        AvgComplexProb=("ComplexProb", "mean")).reset_index()

    # edge-ratio per protein
    edge_ratios = []
    for idx, c in enumerate(complex_list):
        overall, p_means = calc_edge_features(c, G)
        for p, ms in p_means.items():
            edge_ratios.append({
                "Protein": p,
                "ComplexID": idx,
                "EdgeRatio": ms / overall if overall else 0})
    edge_df = pd.DataFrame(edge_ratios)
    max_edge = edge_df.groupby("Protein")["EdgeRatio"].max().reset_index()\
                      .rename(columns={"EdgeRatio": "MaxEdgeRatio"})
    protein_cpx = protein_cpx.merge(max_edge, on="Protein", how="left")

    # (4) PROTREC protein-probability
    PROTREC_prot1 = PROTREC_protprob_bayesian(
        HeLa.iloc[:, i], complex_list1, meanp1, FDR, FNR,
        THRESHOLD, hela_validate, pred1)
    combined1 = pd.merge(PROTREC_prot1.iloc[:, 0:2],
                         protein_cpx, on="Protein", how="left")

    PROTREC_prot = PROTREC_protprob_bayesian(
        HeLa.iloc[:, i], complex_list, meanp, FDR, FNR,
        THRESHOLD, hela_validate, pred)
    combined = pd.merge(PROTREC_prot.iloc[:, 0:2],
                        protein_cpx, on="Protein", how="left")

    # (5) Final output
    df   = combined.dropna().copy()
    df3  = combined[combined.isna().any(axis=1)].copy()
    X    = df[["MaxComplexProb", "MaxEdgeRatio"]]
    y    = df["Probability"]
    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2,
                                          random_state=RAND_SEED)
    rf_corr = RandomForestRegressor(random_state=RAND_SEED)
    rf_corr.fit(Xtr, ytr)
    mse = mean_squared_error(yts, rf_corr.predict(Xts))
    print(f"    second-RF MSE = {mse:.4e}")

    df.loc[:, "New_Probability"] = rf_corr.predict(X)
    if not df3.empty:
        df3["New_Probability"] = rf_corr.predict(
            df3[["MaxComplexProb", "MaxEdgeRatio"]])
    df2 = pd.concat([df, df3], ignore_index=True)
    df2["Max_Value"] = df2[["Probability", "New_Probability"]].max(axis=1)

    tmpPROTREC_prot       = df2.iloc[:, [0, 3]].copy()
    tmpPROTREC_prot.index = df2.index
    df3_subset            = df3.iloc[:, [0, 1]].copy()
    df3_subset.columns    = tmpPROTREC_prot.columns
    tmpPROTREC_prot1      = PROTREC_prot1.iloc[:, 0:2].copy()
    tmpPROTREC_prot1.columns = tmpPROTREC_prot.columns

    # CORUM-only
    tmpPROTREC_protfinal = pd.concat(
        [tmpPROTREC_prot, df3_subset]).groupby("Protein", as_index=False).max()
    tmpPROTREC_protfinal.to_csv(f"CORUM_only_rep{i}.csv", index=False)

    # comprehensive complex
    tmpPROTREC_protfinal2 = pd.concat(
        [tmpPROTREC_prot, df3_subset, tmpPROTREC_prot1])\
        .groupby("Protein", as_index=False).max()
    tmpPROTREC_protfinal2.to_csv(f"Comprehensive_rep{i}.csv", index=False)

print("\nPipeline finished for all three replicates.")
