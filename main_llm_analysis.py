# main_llm_analysis.py
import json
import os
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from data import ATTRIBUTE_DESCRIPTIONS
import llm_logic
import graph_analyzer_lib as graph_analyzer

RESULTS_DIR = Path("results")


def build_cytoscape_data(nodes, correlation_matrix):
    """Build cytoscape edge/node data for a subgraph."""
    cyto_nodes = [{"data": {"id": n}} for n in nodes]
    cyto_edges = []
    for i, a in enumerate(nodes):
        for b in nodes[i+1:]:
            if a in correlation_matrix.columns and b in correlation_matrix.columns:
                cyto_edges.append({"data": {"source": a, "target": b}})
    return {"nodes": cyto_nodes, "edges": cyto_edges}


def build_full_graph_cytoscape(G, correlation_matrix):
    """Build cytoscape data for the full graph."""
    nodes = [{"data": {"id": n}} for n in G.nodes()]
    edges = []
    for u, v in G.edges():
        weight = float(abs(correlation_matrix.loc[u, v])) if u in correlation_matrix.columns and v in correlation_matrix.columns else 0.0
        edges.append({"data": {"source": u, "target": v, "weight": weight}})
    return {"nodes": nodes, "edges": edges}


def process_subgraph_file(input_path: str, output_list: list, subgraph_type: str, llm_instance, correlation_matrix, results_path: str, all_results_ref: list):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            subgraphs = json.load(f)
    except FileNotFoundError:
        print(f"Chyba: Vstupný súbor '{input_path}' nebol nájdený.")
        return

    print(f"\nNačítaných {len(subgraphs)} podgrafov typu: '{subgraph_type}'. Spúšťam analýzu...")

    for i, subgraph_nodes in enumerate(subgraphs):
        print("\n" + "=" * 50)
        print(f"Spracúvam '{subgraph_type}' #{i + 1}: {subgraph_nodes}")
        print("=" * 50)

        final_answer, original_responses = llm_logic.get_synthesized_answer(
            llm_instance, subgraph_nodes, ATTRIBUTE_DESCRIPTIONS, subgraph_type=subgraph_type
        )

        print("\n--- Pôvodné 3 odpovede od LLM ---")
        for j, r in enumerate(original_responses):
            print(f"Odpoveď #{j + 1}: {r}")

        print("\n--- Finálna syntetizovaná odpoveď ---")
        print(f"✅ {final_answer}")

        entry = {
            "subgraph_type": subgraph_type,
            "nodes_data": subgraph_nodes,
            "synthesized_analysis": final_answer,
            "original_responses": original_responses,
            "visualization_b64": None,
            "cytoscape_data": build_cytoscape_data(subgraph_nodes, correlation_matrix),
            "attributes": {n: ATTRIBUTE_DESCRIPTIONS.get(n, "No description available.") for n in subgraph_nodes}
        }

        output_list.append(entry)
        all_results_ref.append(entry)

        # Incremental save to results/ in full format after every subgraph
        with open(results_path, 'r', encoding='utf-8') as f:
            current = json.load(f)
        current["analyses"] = all_results_ref
        current["metadata"]["total_analyses_processed"] = len(all_results_ref)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(current, f, indent=2, ensure_ascii=False)


def run_llm_analysis():
    """Hlavná funkcia, ktorá riadi celý proces."""

    # Find the correct output dir (second to last, same as original)
    try:
        all_dirs = sorted([d for d in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", d))])
        target_dir_index = -2 if len(all_dirs) >= 2 else -1
        latest_output_dir = all_dirs[target_dir_index]
        input_dir = os.path.join("outputs", latest_output_dir)
        print(f"Spracúvam dáta z priečinka: '{input_dir}'")
    except (IndexError, FileNotFoundError):
        print("Chyba: Nebol nájdený žiadny výstupný priečinok v 'outputs'.")
        return

    # Load correlation matrix for cytoscape edge building
    try:
        df = pd.read_csv("datasets/energydata.csv")
        df = df.drop(columns=['date', 'lights'], errors='ignore')
        correlation_matrix = df.corr()
    except Exception as e:
        print(f"Warning: Nepodarilo sa načítať korelačnú maticu: {e}")
        correlation_matrix = pd.DataFrame()

    # Build full graph cytoscape data from the trimmed graph used in this run
    try:
        with open(os.path.join(input_dir, "cliques.json"), 'r', encoding='utf-8') as f:
            all_cliques = json.load(f)
        with open(os.path.join(input_dir, "claws.json"), 'r', encoding='utf-8') as f:
            all_claws = json.load(f)
        all_nodes = list({n for sg in all_cliques + all_claws for n in sg})
        G = nx.Graph()
        G.add_nodes_from(all_nodes)
        for sg in all_cliques:
            for i, a in enumerate(sg):
                for b in sg[i+1:]:
                    G.add_edge(a, b)
        full_graph_cytoscape = build_full_graph_cytoscape(G, correlation_matrix)
    except Exception as e:
        print(f"Warning: Nepodarilo sa vytvoriť full graph cytoscape: {e}")
        full_graph_cytoscape = {"nodes": [], "edges": []}

    # Parse alpha from folder name
    alpha_normalized = None
    alpha_internal = None
    if "untrimmed" in latest_output_dir:
        alpha_normalized = 0
        alpha_internal = 0
    elif "alpha" in latest_output_dir:
        try:
            alpha_internal = float(latest_output_dir.split("alpha")[-1])
            alpha_normalized = alpha_internal / 0.3
        except:
            pass

    # Build initial result structure matching the expected format
    dataset_name = "energydata"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{dataset_name}-alpha_norm{alpha_normalized if alpha_normalized is not None else 'untrimmed'}.json"

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = str(RESULTS_DIR / filename)

    # Count total subgraphs to analyze
    sampled_path = os.path.join("analyzed_subgraphs", "subgraphs_to_analyze.json")
    try:
        with open(sampled_path, 'r', encoding='utf-8') as f:
            sampled = json.load(f)
    except FileNotFoundError:
        print(f"Chyba: '{sampled_path}' nebol nájdený.")
        return

    initial_structure = {
        "metadata": {
            "dataset_name": dataset_name,
            "csv_path": "datasets/energydata.csv",
            "descriptions_path": "datasets/attribute_descriptions.json",
            "alpha_internal": alpha_internal,
            "alpha_normalized": alpha_normalized,
            "total_analyses_found": len(sampled),
            "total_analyses_processed": 0,
            "timed_out": False,
            "attribute_descriptions": ATTRIBUTE_DESCRIPTIONS
        },
        "full_graph_cytoscape_data": full_graph_cytoscape,
        "analyses": []
    }

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(initial_structure, f, indent=2, ensure_ascii=False)

    print(f"Výsledky sa budú priebežne ukladať do: '{results_path}'")

    # Split sampled back into claws and cliques
    all_claws_set = [tuple(x) for x in all_claws]
    sampled_claws   = [s for s in sampled if tuple(s) in all_claws_set]
    sampled_cliques = [s for s in sampled if tuple(s) not in all_claws_set]

    sampled_claws_path   = os.path.join("analyzed_subgraphs", "sampled_claws.json")
    sampled_cliques_path = os.path.join("analyzed_subgraphs", "sampled_cliques.json")
    with open(sampled_claws_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_claws, f, indent=2, ensure_ascii=False)
    with open(sampled_cliques_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_cliques, f, indent=2, ensure_ascii=False)

    llm_instance = llm_logic.load_llm_model()

    claw_results = []
    clique_results = []
    all_results_ref = []

    process_subgraph_file(sampled_claws_path,   claw_results,   "claw",   llm_instance, correlation_matrix, results_path, all_results_ref)
    process_subgraph_file(sampled_cliques_path, clique_results, "clique", llm_instance, correlation_matrix, results_path, all_results_ref)

    # Final save with timed_out=False
    with open(results_path, 'r', encoding='utf-8') as f:
        current = json.load(f)
    current["metadata"]["timed_out"] = False
    current["metadata"]["total_analyses_processed"] = len(all_results_ref)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(current, f, indent=2, ensure_ascii=False)

    print(f"\nAnalýza dokončená! Výsledky uložené do '{results_path}'.")


if __name__ == "__main__":
    run_llm_analysis()