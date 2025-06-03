# -*- coding: utf-8 -*-
# complex_data.py
"""
Load and pre-process every protein-complex resource required by the
PROTREC downstream code.
Tissue list: 'Bile', 'Bladder', 'Bone', 'Brain', 'breast',
       'cervical', 'colorectal', 'esophagel', 'Galbladder', 'Gastric', 'head',
       'Kidney', 'Leukemia', 'liver', 'lung', 'lymphoma', 'Myeloma',
       'Neuroblastoma', 'Non-cancerous', 'ovarian', 'pancreatic', 'prostate',
       'Rhabdoid', 'Sarcoma', 'skin', 'testis', 'thyroid', 'Uncategorized',
       'Uterine'
Remember to change the number according to which tissue you are using. Here 
we use 6 (cervical) for illustrasion.

Public objects
--------------
complex_list   : pd.Series  – short complexes (<5 subunits)  + tissue-2
initial_prob   : pd.Series  – corresponding prior probability (col-6)

complex_list1  : pd.Series  – reference complexes (tissue-0/1 + extra list)
initial_prob1  : pd.Series  – corresponding prior probability (col-6)
"""

from pathlib import Path
import csv, re, ast
import pandas as pd
import numpy as np

# --------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------- #

ROOT = Path(r"C:\Users\33143\Documents")        # base dir for all input files

CORUM_FILE        = ROOT / "human_complexes_2022.txt"
TISSUE_FILE_0     = "updated_tissue_complex_0.csv"
TISSUE_FILE_1     = "updated_tissue_complex_1.csv"
TISSUE_FILE_2     = "updated_tissue_complex_2.csv"
EXTRA_COMPLEX_CSV = "complex_list.csv"
GO_FILE           = "GO.tsv"

# --------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------- #

def _load_human_complexes(max_subunits: int = 4) -> pd.Series:
    """Return a Series (index = CORUM ID, value = list[str]) for human complexes
    containing ≤ *max_subunits* proteins."""
    df = pd.read_csv(CORUM_FILE, sep="\t")
    df = df[df.iloc[:, 2] == "Human"]
    series = df.iloc[:, 5].str.split(";")
    series.index = df.iloc[:, 0]
    return series[series.apply(len) <= max_subunits]


def _read_matrix(name: str, replace: dict | None = None) -> pd.DataFrame:
    """Read a CSV under *ROOT* and optionally replace values."""
    df = pd.read_csv(ROOT / name)
    if replace:
        df = df.replace(replace)
    return df


def _apply_go_adjustment(df: pd.DataFrame, number) -> None:
    """
    Overwrite column-6 in *df* by averaging it with the share of common GO
    terms among all proteins in each complex.
    """
    protein_go = pd.read_table(ROOT / GO_FILE)
    go_map: dict[str, set[str]] = {}
    for _, row in protein_go.iterrows():
        go_map[row[0]] = set(re.findall(r"\[GO:\d+\]", str(row[4]))) if pd.notna(row[4]) else set()

    def shared_fraction(cplx):
        prots = ast.literal_eval(cplx) if not isinstance(cplx, list) else cplx
        go_sets = [go_map.get(p, set()) for p in prots]
        if not go_sets:
            return 0.0
        return len(set.intersection(*go_sets)) / max(len(prots), 1)

    df.iloc[:, 6] = (df.iloc[:, 6] + df.iloc[:, 0].apply(shared_fraction)) / 2


def _strip_isoform(lst):
    """Remove isoform suffix (e.g. P12345-2 → P12345)."""
    return [re.sub(r"-.*", "", x) for x in lst]


# --------------------------------------------------------------------- #
# SET A ─ short complexes + tissue-2  (→ complex_list / initial_prob)
# --------------------------------------------------------------------- #

short_cplx = _load_human_complexes()

tissue2 = _read_matrix(TISSUE_FILE_2, {0: 1, 0.5: 1, 1: 1})
dummy = pd.DataFrame(0.5, index=short_cplx.index, columns=tissue2.columns)
dummy.iloc[:, 0] = short_cplx
combined_a = pd.concat([tissue2, dummy], ignore_index=True)

_apply_go_adjustment(combined_a,6)

complex_list = combined_a.iloc[:, 0].apply(
    lambda x: _strip_isoform(ast.literal_eval(x) if not isinstance(x, list) else x)
)
initial_prob = combined_a.iloc[:, 6] #Change to apply for other tissue

# --------------------------------------------------------------------- #
# SET B ─ original tissue-0/1 + extras  (→ complex_list1 / initial_prob1)
# --------------------------------------------------------------------- #

t0 = _read_matrix(TISSUE_FILE_0, {0: 0.2, 0.5: 0.5, 1: 0.99})
t1 = _read_matrix(TISSUE_FILE_1, {0: 0.2, 0.5: 0.5, 1: 0.99})
t1.rename(columns={"Unnamed: 0": "Protein Complex"}, inplace=True)
combined_b = pd.concat([t0, t1], ignore_index=True)

with open(ROOT / EXTRA_COMPLEX_CSV, newline="") as fh:
    for row in csv.reader(fh):
        if str(row) not in combined_b["Protein Complex"].values:
            combined_b.loc[len(combined_b)] = [str(row)] + [0.5]*(combined_b.shape[1]-1)

_apply_go_adjustment(combined_b,6) #Change to apply for other tissue

complex_list1 = combined_b.iloc[:, 0].apply(
    lambda x: _strip_isoform(ast.literal_eval(x) if not isinstance(x, list) else x)
)
initial_prob1 = combined_b.iloc[:, 6] #Change to apply for other tissue

# --------------------------------------------------------------------- #
# PUBLIC API
# --------------------------------------------------------------------- #

__all__ = ["complex_list", "initial_prob", "complex_list1", "initial_prob1"]

if __name__ == "__main__":
    print(f"Set A → {len(complex_list):>5} complexes")
    print(f"Set B → {len(complex_list1):>5} complexes")
    print("initial_prob (head):")
    print(initial_prob.head())
