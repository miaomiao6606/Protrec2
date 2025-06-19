import pandas as pd
import numpy as np
import re
import csv


def load_complex_and_go_data(tissue: str = "cervical"):
    # === Step 1: Load CORUM/ASTRA complexes ===
    human_complexes_2022 = pd.read_csv("data/human_complexes_2022.txt", delimiter='\t')
    human_complexes_2022 = human_complexes_2022[human_complexes_2022.iloc[:, 2] == "Human"]

    complex_vector2022 = human_complexes_2022.iloc[:, 5].apply(lambda x: x.split(';'))
    complex_vector2022.index = human_complexes_2022.iloc[:, 0]
    complex_vector = complex_vector2022[complex_vector2022.apply(len) < 5].apply(lambda x: list(x))

    a_2 = pd.read_csv("data/updated_tissue_complex_2.csv")
    a_2 = a_2.replace({0: 1, 0.5: 1, 1: 1})
    num_columns = a_2.shape[1]
    complex_vector_df = pd.DataFrame(0.5, index=complex_vector.index, columns=range(num_columns))
    complex_vector_df.iloc[:, 0] = complex_vector
    complex_vector_df.columns = a_2.columns
    combined_a = pd.concat([a_2, complex_vector_df], ignore_index=True)
    combined_a.iloc[:, 0] = combined_a.iloc[:, 0].apply(str)

    with open("data/complex_list.csv", 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        complex_lis = [row for row in reader if len(row) >= 0]

    a_0 = pd.read_csv("data/updated_tissue_complex_0.csv")
    a_1 = pd.read_csv("data/updated_tissue_complex_1.csv")
    a_1.rename(columns={'Unnamed: 0': 'Protein Complex'}, inplace=True)
    combined_a1 = pd.concat([a_0, a_1], ignore_index=True)
    combined_a1 = combined_a1.replace({0: 0.2, 0.5: 0.5, 1: 0.99})

    new_rows = []
    for entry in complex_lis:
        row = [str(entry)]
        if row[0] not in combined_a1['Protein Complex'].values:
            row.extend([0.5] * (len(combined_a1.columns) - 1))
            new_rows.append(row)
    new_rows_df = pd.DataFrame(new_rows, columns=combined_a1.columns)
    combined_a1 = pd.concat([combined_a1, new_rows_df], ignore_index=True)

    protein_GO = pd.read_csv("data/GO.tsv", sep='\t')
    protein_go = []
    for idx, row in protein_GO.iterrows():
        tmp = row[4]
        if pd.notna(tmp):
            matches = re.findall(r'\[GO:\d+\]', tmp)
            s = [row[0], matches]
        else:
            s = [row[0], np.nan]
        protein_go.append(s)

    protein_go_dict = {}
    for entry in protein_go:
        protein_id = entry[0]
        go_terms = entry[1]
        if isinstance(go_terms, list):
            protein_go_dict[protein_id] = set(go_terms)
        else:
            protein_go_dict[protein_id] = set()

    def find_common_go_terms(proteins, protein_go_dict):
        go_sets = [protein_go_dict[p] for p in proteins if p in protein_go_dict]
        if not go_sets:
            return 0
        common_go_terms = set.intersection(*go_sets)
        return min(len(common_go_terms), len(proteins))

    results = []
    for complex_ in combined_a1.iloc[:, 0]:
        proteins = eval(complex_)
        shared_go_count = find_common_go_terms(proteins, protein_go_dict)
        results.append(shared_go_count / len(proteins))

    combined_a1.loc[:, tissue] = [
        (result + current_value) / 2
        for current_value, result in zip(combined_a1.iloc[:, 6], results)
    ]

    combined_a.iloc[:, 0] = combined_a.iloc[:, 0].apply(str)
    combined_a1.iloc[:, 0] = combined_a1.iloc[:, 0].apply(str)
    combined_all = pd.concat([combined_a, combined_a1], ignore_index=True).drop_duplicates().reset_index(drop=True)
    combined_all.iloc[:, 0] = combined_all.iloc[:, 0].apply(
        lambda x: str([re.sub(r'-.*', '', protein) for protein in eval(x)])
    )

    complex_list = combined_all.iloc[:, 0].apply(eval)
    initial_prob = combined_all.loc[:, tissue].copy()
    return complex_list, initial_prob
