from flask import Flask, jsonify
from flask_cors import CORS
import json
import os
import pandas as pd
import numpy as np
import graph_analyzer
from data import ATTRIBUTE_DESCRIPTIONS

app = Flask(__name__)
CORS(app)

ANALYSES_DATA = []
GRAPH_INSTANCE = None


def load_data_and_graph():
    """
    Načíta finálne výsledky, zotriedi ich (claws prvé), očísluje
    a načíta pôvodný graf pre vizualizácie.
    """
    global ANALYSES_DATA, GRAPH_INSTANCE

    analyses_path = os.path.join("llm_analysis_results", "all_analyses.json")
    try:
        with open(analyses_path, 'r', encoding='utf-8') as f:
            raw_analyses = json.load(f)
    except FileNotFoundError:
        print(f"KRITICKÁ CHYBA: Súbor s analýzami '{analyses_path}' nebol nájdený.")
        return

    try:
        df = pd.read_csv('datasets/energydata.csv').drop(columns=['date', 'lights'])
        corr_matrix = df.corr()
        cor_values = corr_matrix.values[~np.eye(corr_matrix.shape[0], dtype=bool)]
        threshold = (corr_matrix.values.max() + cor_values.mean()) / 2 + 0.1
        trimmed_matrix_np = np.where(np.abs(corr_matrix) > threshold, corr_matrix, 0)
        np.fill_diagonal(trimmed_matrix_np, 0)
        nodes = corr_matrix.columns.tolist()
        GRAPH_INSTANCE = graph_analyzer.construct_graph(trimmed_matrix_np, nodes)
        print("Pôvodný graf úspešne zrekonštruovaný pre vizualizácie.")
    except Exception as e:
        print(f"CHYBA pri rekonštrukcii grafu: {e}. Vizualizácie nebudú dostupné.")
        GRAPH_INSTANCE = None

    # --- NOVÁ ČASŤ: Zoradenie a oddelené číslovanie pre claws a kliky
    claws = []
    cliques = []

    for item in raw_analyses:
        if item.get("subgraph_type") == 'clique':
            cliques.append(item)
        else:
            claws.append(item)

    # Spojíme zoznamy - claws budú prvé
    sorted_analyses = claws + cliques

    # --- UPRAVENÁ ČASŤ: Formátovanie dát s novým číslovaním ---
    formatted_analyses = []
    claw_counter = 1
    clique_counter_3 = 1
    clique_counter_4 = 1
    clique_counter_5 = 1

    for i, item in enumerate(sorted_analyses):
        nodes_data = item.get("nodes_data", [])
        subgraph_type = item.get("subgraph_type", "unknown")

        name = ""
        viz_b64 = None

        if subgraph_type == 'clique':
            node_count = len(nodes_data)
            if node_count == 3:
                name = f"3-Node Clique #{clique_counter_3}"
                clique_counter_3 += 1
            elif node_count == 4:
                name = f"4-Node Clique #{clique_counter_4}"
                clique_counter_4 += 1
            elif node_count == 5:
                name = f"5-Node Clique #{clique_counter_5}"
                clique_counter_5 += 1
            else:
                name = f"Clique (Unknown Size) #{i}" # Fallback
            if GRAPH_INSTANCE:
                viz_b64 = graph_analyzer.create_single_clique_viz_base64(GRAPH_INSTANCE, nodes_data)
        else:  # Predpokladáme, že všetko ostatné je 'claw'
            name = f"Claw #{claw_counter} (Center: {nodes_data[0]})" if nodes_data else f"Claw #{claw_counter}"
            claw_counter += 1
            if GRAPH_INSTANCE:
                viz_b64 = graph_analyzer.create_single_claw_viz_base64(GRAPH_INSTANCE, nodes_data)

        # Doplnenie údajov o podgrafe pre jednoduchšie filtrovanie v React
        formatted_analyses.append({
            "id": f"{subgraph_type}-{i}",
            "name": name,
            "attributes": {node: True for node in nodes_data},
            "correlationText": item.get("synthesized_analysis", "Analysis not available."),
            "originalResponses": item.get("original_responses", []),
            "visualization_b64": viz_b64,
            "subgraph_type": subgraph_type,
            "nodes_data": nodes_data
        })

    ANALYSES_DATA = formatted_analyses
    print(
        f"Dáta úspešne načítané. Pripravených {len(ANALYSES_DATA)} analýz ({len(claws)} claws, {len(cliques)} cliques).")


@app.route('/api/analyses')
def get_analyses():
    if not ANALYSES_DATA:
        return jsonify({"error": "No analysis data found. Please run the data processing pipeline first."}), 500
    return jsonify(ANALYSES_DATA)


@app.route('/api/attribute_metadata')
def get_attribute_metadata():
    metadata_for_react = {
        key: {"description": value}
        for key, value in ATTRIBUTE_DESCRIPTIONS.items()
    }
    return jsonify(metadata_for_react)

@app.route('/api/full_graph')
def get_full_graph_visualization():
    if GRAPH_INSTANCE is None:
        return jsonify({"error": "Main graph is not available. Please check the backend console for errors during graph reconstruction."}), 500
    try:
        viz_b64 = graph_analyzer.create_full_graph_viz_base64(GRAPH_INSTANCE)
        return jsonify({"visualization_b64": viz_b64})
    except Exception as e:
        print(f"Error creating full graph visualization: {e}")
        return jsonify({"error": "Failed to create full graph visualization. Please check the backend logs."}), 500


if __name__ == '__main__':
    load_data_and_graph()
    print("\nSpúšťam Python Flask server na adrese http://127.0.0.1:5000...")
    app.run(debug=True, port=5000)