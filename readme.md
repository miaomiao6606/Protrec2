# PROTREC2

A protein recovery and inference framework using tissue-specific priors and complex-based Bayesian modeling.

##  Installation
```
pip install -r requirements.txt
```

## Input Format
### Expression Matrix (CSV)
- Rows: proteins (UniProt ID or symbol)
- Columns: samples
- Example: `example/example_expression.csv`

### Tissue-Specific Prior
- Built-in from `updated_tissue_complex_*.csv`
- Tissue types: `lung`, `breast`, `brain`, etc. You may find the names in the excel sheet.

##  Run Example
```
python main.py \
  --expression example/example_expression.csv \
  --output_dir results/ \
  --tissue lung \
  --fdr 0.01 \
  --threshold 5
```

##  Output
- CSV files per sample, each with recovered protein probabilities:
```
Protein,Probability
P12345,0.862
Q67890,0.721
...
```
