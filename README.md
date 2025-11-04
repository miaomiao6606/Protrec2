# PROTREC2

**Tissue-Specific Network-based Missing Protein Recovery Method**

## Overview

PROTREC2 is an enhanced protein recovery framework that integrates tissue-specific protein complex annotations with Bayesian inference to recover unreported but biologically present proteins in proteomics datasets. This method addresses the persistent challenge of missing proteins in mass spectrometry-based proteomics.

### Key Features
- **Tissue-specific protein complex annotations** across 27 distinct tissues
- **Bayesian probabilistic framework** for iterative protein likelihood updates  
- **Comprehensive protein complex database** integrating CORUM, Complex Portal, and Reactome
- **Gene Ontology (GO)-based functional coherence** scoring
- **Superior performance** compared to existing methods (PROTREC, FCS, HE, GSEA)

## Citation

If you use PROTREC2 in your research, please cite:

Kong W, Goh WWB, Wong L. Protrec2: Tissue Specific Network-based Missing Protein Recovery Method. [Publication details]

## Installation

### Prerequisites
- Python 3.7 or higher
- pip or conda package manager

### Method 1: Using pip
```bash
# Clone the repository
git clone https://github.com/miaomiao6606/Protrec2.git
cd Protrec2

# Install dependencies
pip install -r requirements.txt

# Install the package (optional)
pip install -e .
```

### Method 2: Using conda
```bash
# Clone the repository  
git clone https://github.com/miaomiao6606/Protrec2.git
cd Protrec2

# Create conda environment
conda create -n protrec2 python=3.8
conda activate protrec2

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage
```bash
python main.py \
  --expression example/example_expression.csv \
  --output_dir results/ \
  --tissue lung \
  --fdr 0.01 \
  --threshold 5
```

### Available Tissues
Run the following command to see all available tissue types:
```bash
python main.py --help
```

Available tissues include:
- bile, bladder, bone, brain, breast, cervical, colorectal, esophageal
- gallbladder, gastric, head_and_neck, kidney, leukemia, liver, lung
- lymphoma, myeloma, neuroblastoma, ovarian, pancreatic, prostate
- rhabdoid_tumor, sarcoma, skin, testis, thyroid, uterine

## Input Data Format

### 1. Expression Matrix (Required)
- **Format**: CSV file
- **Structure**: 
  - Rows: Protein identifiers (UniProt IDs or gene symbols)
  - Columns: Sample names
  - Values: Expression levels (numeric, non-negative)
- **Example**: `example/example_expression.csv`

Example format:
```csv
,Sample1,Sample2,Sample3
P12345,10.5,0,8.3
Q67890,0,5.2,7.1
...
```

### 2. Built-in Data Resources

PROTREC2 includes comprehensive built-in databases:

- **Protein Complex Databases**:
  - CORUM database (manually curated mammalian complexes)
  - Complex Portal (EMBL-EBI validated complexes)
  - Reactome pathway-derived interactions
  - STRING-expanded complexes (confidence > 780)
  - Total: 3,885 unique protein complexes

- **Tissue-Specific Annotations**:
  - `data/updated_tissue_complex_0.csv`: Tissue-specific complex scores
  - `data/updated_tissue_complex_1.csv`: Additional tissue annotations
  - `data/updated_tissue_complex_2.csv`: Extended tissue profiles

- **Gene Ontology Data**:
  - `data/GO.tsv`: GO term associations for functional coherence

## Algorithm Workflow

### Step 1: Comprehensive Protein Complex Generation
1. Load curated complexes from CORUM (size ≥ 5 proteins)
2. Expand using STRING interactions (score > 780)
3. Integrate Complex Portal and Reactome data
4. Filter for biological relevance (>80% valid UniProt IDs)

### Step 2: Tissue-Specific Probability Assignment
1. Assign tissue relevance scores:
   - 1.0: Explicit tissue support
   - 0.5: Ambiguous/unspecified
   - 0.0: Explicitly non-relevant
2. Calculate GO-based functional coherence
3. Integrate scores into tissue-specific priors

### Step 3: Bayesian Protein Recovery
1. Initialize protein probabilities based on observation status
2. For each protein, integrate:
   - Direct observation probability
   - Complex membership support
   - Tissue-specific prior probability
3. Iteratively update using Bayesian inference
4. Normalize and threshold final probabilities

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--expression` | Required | Path to expression matrix CSV |
| `--output_dir` | Required | Output directory for results |
| `--tissue` | "lung" | Tissue type for priors |
| `--fdr` | 0.01 | False discovery rate |
| `--fnr` | 0.0 | False negative rate |
| `--threshold` | 5 | Minimum complex size threshold |

## Output Format

### Per-sample Results
- **Location**: `{output_dir}/PROTREC2_result_{sample_name}.csv`
- **Format**: CSV with columns:
  - `Protein`: UniProt ID or gene symbol
  - `Probability`: Recovery probability (0-1)

Example output:
```csv
Protein,Probability
P12345,0.982
Q67890,0.873
R11223,0.651
...
```

### Interpretation
- **High probability (>0.9)**: Strong evidence for protein presence
- **Medium probability (0.5-0.9)**: Moderate evidence
- **Low probability (<0.5)**: Weak evidence

## Performance

PROTREC2 has been validated on:
- **HeLa proteomes**: 96.5% recovery rate (453 proteins validated)
- **A549 proteomes**: 98.4% recovery rate (650 proteins validated)
- **Lung tumor-normal pairs**: >85% prediction accuracy against CPTAC

Superior performance compared to:
- PROTREC (original version)
- Functional Class Scoring (FCS)
- Hypergeometric Enrichment (HE)
- Gene Set Enrichment Analysis (GSEA)

## Example Analysis

```bash
# 1. Create output directory
mkdir -p results

# 2. Run PROTREC2 on example data
python main.py \
  --expression example/example_expression.csv \
  --output_dir results \
  --tissue lung \
  --fdr 0.01 \
  --threshold 5

# 3. View results
head results/PROTREC2_result_811.csv
```

## Repository Structure

```
Protrec2/
├── Protrec2/                 # Core package
│   ├── complex_data.py       # Complex data loading and processing
│   ├── core.py              # Main PROTREC2 algorithms
│   └── utils.py             # Utility functions
├── data/                    # Built-in databases
│   ├── GO.tsv              # Gene Ontology annotations
│   ├── complex_list.csv    # Protein complex list
│   ├── human_complexes_2022.txt  # CORUM complexes
│   └── updated_tissue_complex_*.csv  # Tissue annotations
├── example/                 # Example data
│   └── example_expression.csv
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Troubleshooting

### Common Issues

1. **Memory errors with large datasets**:
   - Process samples in batches
   - Reduce complex size threshold

2. **No proteins recovered**:
   - Check expression matrix format
   - Verify protein IDs are UniProt format
   - Ensure expression values are non-negative

3. **Tissue type not recognized**:
   - Check available tissues in data files
   - Use default "lung" if unsure

## Contact

- **Weijia Kong**: First Author
- **Wilson Wen Bin Goh**: wilsongoh@ntu.edu.sg
- **Limsoon Wong**: wongls@comp.nus.edu.sg

## License

MIT License (see LICENSE file for details)

## Acknowledgments

- CORUM database for protein complex annotations
- Gene Ontology Consortium for GO annotations  
- Complex Portal (EMBL-EBI) for validated complexes
- Reactome for pathway data
- STRING database for protein interactions