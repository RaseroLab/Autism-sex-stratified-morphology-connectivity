import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score

# Set random seed for reproducibility
np.random.seed(42)

def load_demographics(path):
    df = pd.read_csv(path)
    df = df[~df["Age"].isna()]
    return df


def load_matrix_folder(folder, sub_ids, desc):
    matrices = []
    valid_ids = []
    for sid in tqdm(sub_ids, desc=f"Loading {desc}"):
        filepath = os.path.join(folder, f"{sid}_atlas-aparc_mind.csv")
        if os.path.exists(filepath):
            try:
                mat = pd.read_csv(filepath).iloc[:, 1:].to_numpy()
                matrices.append(mat)
                valid_ids.append(sid)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    if matrices:
        return np.stack(matrices), valid_ids
    return None, []


def load_matrix_special(folder_root, sub_ids):
    matrices = []
    valid_ids = []
    for sid in tqdm(sub_ids, desc="Loading MC_Vol_SA_SD"):
        sid_str = str(sid)
        filename = f"{sid_str}_aparc_MC_Vol_SD_SA.csv"
        filepath = os.path.join(folder_root, sid_str, filename)
        if os.path.exists(filepath):
            try:
                mat = pd.read_csv(filepath).iloc[1:69, :].to_numpy()
                if mat.shape[0] == 68:
                    matrices.append(mat)
                    valid_ids.append(sid_str)
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")
    if matrices:
        return np.stack(matrices), valid_ids
    return None, []


def extract_edge_strengths(matrices, edge_df):
    return np.column_stack([
        matrices[:, int(row["3Drow"] - 1), int(row["3Dcol"] - 1)]
        for _, row in edge_df.iterrows()
    ])


def compute_auc(matrices, demo_df, valid_ids, edge_df, gender):
    gender_label = "M" if gender == "male" else "F"
    matched_df = demo_df[(demo_df.Gender == gender_label) & (demo_df.SUB_ID.isin(valid_ids))]
    
    if matched_df.empty:
        return None, None, None

    y_true = (matched_df.Cohort == "ASD").astype(int).to_numpy()
    if len(np.unique(y_true)) < 2:
        print(f"Skipping {gender} due to class imbalance")
        return None, None, None

    indices = [valid_ids.index(sid) for sid in matched_df.SUB_ID]
    selected_matrices = matrices[indices]

    summed_strength = np.sum(extract_edge_strengths(selected_matrices, edge_df), axis=1)
    fpr, tpr, _ = roc_curve(y_true, summed_strength)
    auc = roc_auc_score(y_true, summed_strength)
    return fpr, tpr, auc


def plot_roc_curves(results_dict, gender, out_dir):
    plt.figure(dpi=300)
    for label, (fpr, tpr, auc) in results_dict.items():
        if fpr is not None:
            plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve Comparison ({gender.capitalize()})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'roc_{gender}.png'))
    plt.close()


def main(gender):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    mat_dir = os.path.join(data_dir, "matrices")
    fig_dir = os.path.join(base_dir, "figures")

    demo_file = os.path.join(data_dir, "demographics.csv")
    edge_file = os.path.join(data_dir, f"cohort_{gender}.csv")

    demo_df = load_demographics(demo_file)
    sub_ids = demo_df.SUB_ID.tolist()
    edge_df = pd.read_csv(edge_file)

    datasets = {
        "All": load_matrix_folder(os.path.join(mat_dir, "CT_MC_Vol_SD_SA"), sub_ids, "All"),
        "no CT": load_matrix_special(os.path.join(mat_dir, "MC_Vol_SA_SD"), sub_ids),
        "no SA": load_matrix_folder(os.path.join(mat_dir, "CT_MC_Vol_SD"), sub_ids, "no SA"),
        "no SD": load_matrix_folder(os.path.join(mat_dir, "CT_MC_Vol_SA"), sub_ids, "no SD"),
        "no Vol": load_matrix_folder(os.path.join(mat_dir, "CT_MC_SD_SA"), sub_ids, "no Vol"),
        "no MC": load_matrix_folder(os.path.join(mat_dir, "CT_Vol_SD_SA"), sub_ids, "no MC"),
    }

    results = {}
    for label, (mats, val_ids) in datasets.items():
        if mats is None or len(val_ids) < 1:
            print(f"Skipping {label}: no valid data")
            continue
        fpr, tpr, auc = compute_auc(mats, demo_df, val_ids, edge_df, gender)
        if auc is not None:
            results[label] = (fpr, tpr, auc)

    plot_roc_curves(results, gender, fig_dir)
    print(f"ROC curve saved to {fig_dir}/roc_{gender}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AUC-based ROC comparison across connectivity models.")
    parser.add_argument("--gender", type=str, required=True, choices=["male", "female"], help="Specify gender cohort")
    args = parser.parse_args()
    main(args.gender)
