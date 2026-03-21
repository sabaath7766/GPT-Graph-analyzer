# main_analysis.py
import json
import random
import pandas as pd
import numpy as np
import datetime
import os
import graph_utils as graph_analyzer

RANDOM_SEED = 42
SAMPLE_SIZE = 100  # max subgraphs sent to LLM

def load_and_prepare_data():
    filepath = 'datasets/energydata.csv'
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Súbor '{filepath}' nebol nájdený.")
        exit()
    df = df.drop(columns=['date', 'lights'])
    return df

def compute_correlation_matrix(df):
    return df.corr()

def trim_correlation_matrix(correlation_matrix, alpha=0.1):
    """
    alpha=None  → no trimming, only zeros the diagonal
    alpha=float → original dynamic threshold trimming
    """
    result = correlation_matrix.copy().astype(float)
    np.fill_diagonal(result.values, 0)

    if alpha is None:
        print("Bez orezania: používa sa plná korelačná matica.")
        return result

    cor_values = result.values[np.where(~np.eye(result.shape[0], dtype=bool))]
    cor_max  = np.max(cor_values)
    cor_mean = np.mean(cor_values)
    threshold = (cor_max + cor_mean) / 2 + alpha
    print(f"Dynamický prah pre korelácie: {threshold:.4f}")

    trimmed_np = np.where(np.abs(result) > threshold, result.values, 0)
    np.fill_diagonal(trimmed_np, 0)
    return pd.DataFrame(trimmed_np, index=result.index, columns=result.columns)


def sample_subgraphs(subgraphs, sample_size, graph_type):
    """
    Randomly samples up to sample_size subgraphs.
    No filtering by edge weight — all subgraphs are eligible.
    """
    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(subgraphs, min(sample_size, len(subgraphs)))
    print(f"  [{graph_type}] {len(subgraphs)} total → sampled {len(sampled)} for LLM")
    return sampled


def run_for_alpha(df, correlation_matrix, alpha, base_timestamp):
    label = f"alpha{alpha}" if alpha is not None else "untrimmed"
    output_dir = os.path.join("outputs", f"{base_timestamp}_{label}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"RUN: {label}  →  {output_dir}")
    print('='*60)

    trimmed_matrix = trim_correlation_matrix(correlation_matrix, alpha=alpha)

    nodes = trimmed_matrix.columns.tolist()
    G = graph_analyzer.construct_graph(trimmed_matrix.values, nodes)
    print(f"Graf: {G.number_of_nodes()} uzlov, {G.number_of_edges()} hrán.")

    claws   = graph_analyzer.find_claw_subgraphs(G)
    cliques = graph_analyzer.find_cliques(G)
    print(f"Nájdených {len(cliques)} klík a {len(claws)} claw podgrafov.")

    # Save full sets (unchanged from original)
    cliques_json = [list(c) for c in cliques]
    claws_json   = [list(c) for c in claws]
    with open(os.path.join(output_dir, "cliques.json"), 'w', encoding='utf-8') as f:
        json.dump(cliques_json, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "claws.json"), 'w', encoding='utf-8') as f:
        json.dump(claws_json, f, indent=2, ensure_ascii=False)

    # Sample for LLM — split budget 50/50 between claws and cliques
    half = SAMPLE_SIZE // 2
    sampled_claws   = sample_subgraphs(claws_json,   half, "claw")
    sampled_cliques = sample_subgraphs(cliques_json, half, "clique")
    subgraphs_for_llm = sampled_claws + sampled_cliques
    print(f"Total subgraphs queued for LLM: {len(subgraphs_for_llm)}")

    llm_input_dir = "analyzed_subgraphs"
    os.makedirs(llm_input_dir, exist_ok=True)
    llm_path = os.path.join(llm_input_dir, "subgraphs_to_analyze.json")
    with open(llm_path, 'w', encoding='utf-8') as f:
        json.dump(subgraphs_for_llm, f, indent=2, ensure_ascii=False)
    print(f"LLM input saved → '{llm_path}'")

    print(f"Run '{label}' dokončený.")


def main():
    base_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df = load_and_prepare_data()
    correlation_matrix = compute_correlation_matrix(df)

    # run_for_alpha(df, correlation_matrix, alpha=0.1,  base_timestamp=base_timestamp)
    run_for_alpha(df, correlation_matrix, alpha=None, base_timestamp=base_timestamp)

    print("\nVšetky behy dokončené.")

if __name__ == "__main__":
    main()