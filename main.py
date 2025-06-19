import argparse
import pandas as pd
import os
from Protrec2.complex_data import load_complex_and_go_data
from Protrec2.core import PROTREC_complex, PROTREC_protprob_bayesian

def main():
    
    parser = argparse.ArgumentParser(description="Run PROTREC2 on protein expression data")
    parser.add_argument('--expression', required=True, help="Protein expression matrix CSV (proteins × samples)")
    parser.add_argument('--output_dir', required=True, help="Directory to save results")
    parser.add_argument('--tissue', type=str, default="lung", help="Tissue type column used for prior (e.g., lung, breast, brain)")
    parser.add_argument('--fdr', type=float, default=0.01, help="False discovery rate")
    parser.add_argument('--fnr', type=float, default=0.0, help="False negative rate")
    parser.add_argument('--threshold', type=int, default=5, help="Minimum complex size threshold")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    expression = pd.read_csv(args.expression, index_col=0)
    full_complex_list, original_initial_prob = load_complex_and_go_data(tissue=args.tissue)
    #full_complex_list.to_csv("new_complex_list.csv", index=False, header=False)
    for col in expression.columns:
        print(f"Processing sample: {col}")
        expr_subset = expression[[col]].loc[expression[col] > 0]
        complex_list = full_complex_list.copy()
        initial_prob = original_initial_prob.copy()

        for key in complex_list.index:
            initial_prob.loc[key] = PROTREC_complex(expr_subset, complex_list, key, args.fdr, args.fnr, args.threshold, initial_prob)

        meanp = initial_prob.mean()
        result_df = PROTREC_protprob_bayesian(expr_subset, complex_list, meanp, args.fdr, args.fnr, args.threshold, initial_prob)
        result_df.to_csv(os.path.join(args.output_dir, f"PROTREC2_result_{col}.csv"), index=False)

    print("All samples processed successfully.")

if __name__ == '__main__':
    df = pd.read_csv("data/updated_tissue_complex_0.csv")
    tissue_names = list(df.columns[1:])
    print("Available tissues:", tissue_names)
    main()