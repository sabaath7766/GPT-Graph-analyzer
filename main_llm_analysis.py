import json
import os
from data import ATTRIBUTE_DESCRIPTIONS
import llm_logic


def process_subgraph_file(input_path: str, output_list: list, subgraph_type: str):
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
            subgraph_nodes, ATTRIBUTE_DESCRIPTIONS, subgraph_type=subgraph_type
        )

        ## ZMENA: Pridaný výpis 3 pôvodných odpovedí
        print("\n--- Pôvodné 3 odpovede od LLM ---")
        for j, r in enumerate(original_responses):
            print(f"Odpoveď #{j + 1}: {r}")

        print("\n--- Finálna syntetizovaná odpoveď ---")
        print(f"✅ {final_answer}")

        output_list.append({
            "subgraph_type": subgraph_type,
            "nodes_data": subgraph_nodes,
            "synthesized_analysis": final_answer,
            "original_responses": original_responses
        })


def run_llm_analysis():
    """Hlavná funkcia, ktorá riadi celý proces."""
    try:
        all_dirs = sorted([d for d in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", d))])
        # Použijeme predposledný (-2) priečinok podľa vašej požiadavky
        target_dir_index = -2 if len(all_dirs) >= 2 else -1
        latest_output_dir = all_dirs[target_dir_index]
        input_dir = os.path.join("outputs", latest_output_dir)
        print(f"Spracúvam dáta z priečinka: '{input_dir}'")
    except (IndexError, FileNotFoundError):
        print("Chyba: Nebol nájdený žiadny výstupný priečinok v 'outputs'.")
        return

    claw_results = []
    clique_results = []

    # Poradie je teraz CLAWS -> CLIQUES
    process_subgraph_file(os.path.join(input_dir, "claws.json"), claw_results, "claw")
    process_subgraph_file(os.path.join(input_dir, "cliques.json"), clique_results, "clique")

    all_results = claw_results + clique_results
    if not all_results:
        print("\nNeboli nájdené žiadne podgrafy na analýzu. Proces končí.")
        return

    output_dir_llm = "llm_analysis_results"
    os.makedirs(output_dir_llm, exist_ok=True)
    results_path = os.path.join(output_dir_llm, "all_analyses.json")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nAnalýza dokončená! Výsledky uložené do '{results_path}'.")


if __name__ == "__main__":
    run_llm_analysis()