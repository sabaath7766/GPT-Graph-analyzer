from flask import Flask, jsonify
from flask_cors import CORS
import json
import os

# Importujeme popisy atribútov priamo z vášho existujúceho súboru
from data import ATTRIBUTE_DESCRIPTIONS

# Inicializácia Flask aplikácie
app = Flask(__name__)
CORS(app)


def find_target_directory():
    """
    Nájde a vráti cestu k predposlednému (alebo poslednému, ak je len jeden)
    výstupnému priečinku v adresári 'outputs'.
    """
    try:
        all_dirs = sorted([d for d in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", d))])
        if not all_dirs:
            return None, "No output directories found in 'outputs'."

        # Použijeme predposledný (-2), ak existujú aspoň dva, inak posledný (-1)
        target_dir_name = all_dirs[-2] if len(all_dirs) >= 2 else all_dirs[-1]
        return os.path.join("outputs", target_dir_name), None
    except FileNotFoundError:
        return None, "The 'outputs' directory was not found."
    except Exception as e:
        return None, f"An error occurred while finding the directory: {e}"


def process_subgraph_file(filepath, subgraph_type):
    """
    Načíta súbor s podgrafmi, priradí im typ a vráti zoznam objektov.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            subgraphs_data = json.load(f)
    except FileNotFoundError:
        print(f"Info: File not found at '{filepath}', skipping.")
        return []  # Vrátime prázdny zoznam, ak súbor neexistuje

    processed_list = []
    for nodes in subgraphs_data:
        processed_list.append({
            "type": subgraph_type,
            "nodes": nodes
        })
    return processed_list


@app.route('/api/analyses')
def get_analyses():
    """
    Tento endpoint dynamicky nájde správny priečinok, načíta dáta
    z `cliques.json` a `claws.json`, spojí ich a pošle do frontendu.
    """
    target_dir, error = find_target_directory()
    if error:
        return jsonify({"error": error}), 404

    # Načítame a spracujeme oba typy súborov
    cliques = process_subgraph_file(os.path.join(target_dir, "cliques.json"), "clique")
    claws = process_subgraph_file(os.path.join(target_dir, "claws.json"), "claw")

    all_subgraphs = cliques + claws

    if not all_subgraphs:
        return jsonify({"error": f"No cliques or claws found in the target directory: {target_dir}"}), 404

    # Teraz dáta preformátujeme pre React, rovnako ako predtým
    formatted_graphs = []
    for i, analysis_obj in enumerate(all_subgraphs):
        subgraph_type = analysis_obj["type"]
        nodes = analysis_obj["nodes"]
        name = ""

        if subgraph_type == 'clique':
            name = f"Clique #{i + 1}"
        elif subgraph_type == 'claw' and nodes:
            name = f"Claw #{i + 1} (Center: {nodes[0]})"
        else:
            name = f"Subgraph #{i + 1}"

        formatted_graphs.append({
            "id": i,
            "name": name,
            "description": f"A {subgraph_type} structure with {len(nodes)} attributes.",
            "attributes": {node: True for node in nodes},
            # Tieto polia pridáme ako prázdne, pretože ich LLM analýza ešte neprebehla
            "correlationText": "Awaiting LLM analysis...",
            "originalResponses": []
        })

    return jsonify(formatted_graphs)


@app.route('/api/attribute_metadata')
def get_attribute_metadata():
    """
    Tento endpoint vracia popisy atribútov.
    """
    metadata_for_react = {
        key: {"description": value}
        for key, value in ATTRIBUTE_DESCRIPTIONS.items()
    }
    return jsonify(metadata_for_react)


if __name__ == '__main__':
    print("Spúšťam Python Flask server na adrese http://127.0.0.1:5000...")
    app.run(debug=True, port=5000)

