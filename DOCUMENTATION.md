# PROTREC2 Technical Documentation

## Algorithm Details

### 1. Comprehensive Protein Complex Generation

#### 1.1 Initial Construction and Expansion
- **Base Dataset**: CORUM database (694 human complexes with ≥5 proteins)
- **STRING Expansion**: 
  - Threshold: interaction score > 780
  - Method: Add proteins with high-confidence interactions to all complex members
  - Result: 1,592 complexes

#### 1.2 Database Integration
- **Complex Portal**: Manually curated EMBL-EBI complexes
- **Reactome**: Pathway-derived protein associations
- **Final Dataset**: 3,885 unique protein complexes

#### 1.3 Quality Control
- Minimum complex size: 5 proteins
- UniProt ID validation: >80% valid identifiers required
- Redundancy removal and standardization

### 2. Tissue-Specific Probability Assignment

#### 2.1 Tissue Annotation Scoring
```
Score = 1.0: Explicit tissue support in literature/database
Score = 0.5: Ambiguous or unspecified tissue association  
Score = 0.0: Explicitly non-relevant to tissue
```

#### 2.2 GO-Based Functional Coherence
For complexes without strong tissue evidence:
1. Extract GO terms for all proteins in complex
2. Calculate intersection of GO terms
3. Coherence score = min(|common GO terms|, |complex size|) / |complex size|
4. Final score = (tissue_score + coherence_score) / 2

#### 2.3 Supported Tissues (27 total)
- **Cancer types**: leukemia, lymphoma, myeloma, neuroblastoma, rhabdoid tumor, sarcoma
- **Organ-specific**: brain, lung, liver, kidney, pancreatic, prostate, testis, thyroid
- **System-specific**: bladder, bone, bile, gallbladder, gastric, colorectal, esophageal
- **Other**: breast, cervical, ovarian, uterine, skin, head and neck

### 3. Bayesian Inference Framework

#### 3.1 Initialization
```python
# For observed proteins
P(protein|observed) = 1 - FDR

# For unobserved proteins  
P(protein|unobserved) = 0.1
```

#### 3.2 Complex-Level Probability (PROTREC_complex)
```python
def PROTREC_complex(data, complex_list, complex_key, fdr, fnr, threshold, initial_prob):
    # Count observed proteins in complex
    n = count_observed_proteins(complex, data)
    
    # Get complex size with threshold
    m = max(threshold, len(complex))
    
    # Calculate probability
    p_complex = min(1 - fdr, (n/m * (1-fdr) + initial_prob) / 2)
    
    return p_complex
```

#### 3.3 Protein-Level Bayesian Update (PROTREC_protprob_bayesian)
For each unobserved protein:

1. **Complex Support Calculation**:
   ```python
   sum_probs = sum(P(other_protein) for other_protein in complex)
   p_support = min(1 - fdr, sum_probs / complex_size)
   ```

2. **Logit-Based Integration**:
   ```python
   log_prior = logit(P(protein|previous))
   log_support = logit(p_support)  
   log_complex = logit(P(complex|tissue))
   
   combined_logit = (log_prior + log_support + log_complex) / 3
   P(protein|new) = expit(combined_logit)
   ```

3. **Iterative Refinement**:
   - Continue until convergence (Δ < 1e-3)
   - Maximum iterations: configurable (default=1)

#### 3.4 Final Normalization
```python
# For proteins below observation threshold
if P(protein) < (1 - fdr):
    # Z-score normalization
    z_score = (P(protein) - mean) / std
    
    # Scale to [mean_prior, 1-fdr] range
    P(protein|final) = scale_to_range(z_score, mean_prior, 1-fdr)
```

## Implementation Details

### File: `Protrec2/complex_data.py`

**Function**: `load_complex_and_go_data(tissue)`
- Loads and processes all complex databases
- Integrates tissue-specific annotations
- Calculates GO-based coherence scores
- Returns: (complex_list, initial_probabilities)

### File: `Protrec2/core.py`

**Function**: `PROTREC_complex()`
- Calculates complex-level probabilities
- Integrates observed protein evidence
- Applied per complex, per sample

**Function**: `PROTREC_protprob_update()`
- Updates protein probability based on complex members
- Used in iterative Bayesian inference

**Function**: `PROTREC_protprob_bayesian()`
- Main Bayesian inference engine
- Iteratively refines protein probabilities
- Outputs final protein recovery scores

### File: `Protrec2/utils.py`

**Function**: `safe_logit(p, eps=1e-6)`
- Numerically stable logit transformation
- Clips probabilities to [eps, 1-eps]

**Function**: `safe_expit(x)`
- Numerically stable inverse logit (sigmoid)

## Performance Optimization

### Memory Management
- Sparse matrix representation for large complex sets
- Batch processing for multiple samples
- Efficient pandas operations using vectorization

### Computational Efficiency
- Early convergence checking in iterations
- Cached tissue-specific priors
- Parallel processing support for multiple samples

## Validation Metrics

### Upper-bound Evaluation
- Test on complete proteomes
- Randomly mask proteins
- Measure recovery rate

### Lower-bound Evaluation  
- Test on sparse proteomes
- Use biological replicates
- Measure precision against validation set

### Performance Metrics
- **Recovery Rate**: % of masked proteins recovered
- **Precision**: % of predictions validated
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

## Best Practices

### 1. Data Preparation
- Ensure protein IDs are in UniProt format
- Remove contaminants and reverse sequences
- Log-transform intensities if needed
- Handle missing values appropriately

### 2. Parameter Selection
- **FDR**: Use proteomics search engine FDR (typically 0.01)
- **Threshold**: 5 for high confidence, 3 for more coverage
- **Tissue**: Match to sample origin for best results

### 3. Result Interpretation
- Focus on high-probability predictions (>0.9)
- Validate using orthogonal methods
- Consider biological context
- Check complex membership for validation

## Troubleshooting Guide

### Issue: Low Recovery Rate
**Solutions**:
- Check if tissue type matches sample
- Verify protein ID format
- Lower threshold parameter
- Ensure adequate complex coverage

### Issue: High False Positives
**Solutions**:
- Increase FDR parameter
- Use stricter threshold
- Filter by complex size
- Require multiple complex support

### Issue: Memory Errors
**Solutions**:
- Process samples individually
- Reduce complex database size
- Use sparse matrix operations
- Increase system memory

## References

1. Kong et al. (2022) PROTREC: A probability-based approach for recovering missing proteins. J Proteomics 250:104392
2. Giurgiu et al. (2019) CORUM: the comprehensive resource of mammalian protein complexes. Nucleic Acids Res 47:D559-D563
3. von Mering et al. (2003) STRING: a database of predicted functional associations. Nucleic Acids Res 31:258-261
4. Meldal et al. (2022) Complex Portal 2022: new curation frontiers. Nucleic Acids Res 50:D578-D586
5. Croft et al. (2011) Reactome: a database of reactions, pathways and biological processes. Nucleic Acids Res 39:D691-D697
6. UniProt Consortium (2015) UniProt: a hub for protein information. Nucleic Acids Res 43:D204-D212
7. Greene et al. (2015) Understanding multicellular function and disease with human tissue-specific networks. Nat Genet 47:569-576
8. Gene Ontology Consortium (2000) Gene ontology: tool for the unification of biology. Nat Genet 25:25-29