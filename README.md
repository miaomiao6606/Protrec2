## README 

This repo contains two Python scripts for running a **Protrec2** workflow on HeLa DIA data and producing protein-level existence probabilities.

| file                  | purpose                                                                                                                                                                                                                             |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`complex_data.py`** | Loads/cleans complex resources (CORUM, tissue matrices, GO). Exposes four ready-to-use objects:<br> • `complex_list`, `initial_prob` – short complexes<br> • `complex_list1`, `initial_prob1` – full complex catalogue              |
| **`main_protrec.py`** | 1. Reads the input files, tissue-specific network and comprehensive complexes.<br>2. Builds graph features and Random-Forest priors.<br>3. Runs the Protrec2 Bayesian update.<br>4. Writes per-protein probability tables (CSV) for the input files. |

### Requirements

```bash
pip install numpy pandas scipy networkx scikit-learn
```

### Quick run

```bash
python main_protrec.py
```

Outputs (per replicate, take HeLa as an example):

```
CORUM_only_rep0/1/2.csv        # using CORUM complexes only
Comprehensive_rep0/1/2.csv     # using the full complex set
```

If your input paths differ, edit the **CONFIG** blocks at the top of each script. Remember to Change the tissue number when apply to other datasets.
