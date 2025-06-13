import json
import os
from data import ATTRIBUTE_DESCRIPTIONS
import llm_logic


def process_subgraph_file(input_path: str, output_path: str, subgraph_type: str):
    """
    Načíta súbor s podgrafmi, spustí LLM analýzu pre každý z nich
    a uloží výsledky do výstupného súboru.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            subgraphs = json.load(f)
    except FileNotFoundError:
        print(f"Chyba: Vstupný súbor '{input_path}' nebol nájdený.")
        return

    print("\n" + "=" * 50)
    print(f"Zahajujem analýzu pre {len(subgraphs)} podgrafov typu: '{subgraph_type}'")
    print("=" * 50)

    all_results = []
    for i, subgraph_nodes in enumerate(subgraphs):
        print(f"\n--- Spracúvam '{subgraph_type}' #{i + 1}: {subgraph_nodes} ---")

        # Zavoláme funkciu z knižnice s príslušným typom
        final_answer, original_responses = llm_logic.get_synthesized_answer(
            subgraph_nodes,
            ATTRIBUTE_DESCRIPTIONS,
            subgraph_type=subgraph_type  # Tu posielame typ podgrafu
        )

        print(f"Finálna analýza: {final_answer}")

        all_results.append({
            "subgraph_nodes": subgraph_nodes,
            "synthesized_analysis": final_answer,
            "original_responses": original_responses
        })

    # Uloženie všetkých výsledkov pre daný typ podgrafu
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAnalýza pre '{subgraph_type}' dokončená! Výsledky uložené do '{output_path}'.")


def run_llm_analysis():
    """Hlavná funkcia, ktorá riadi celý proces."""

    # Priečinok, z ktorého budeme čítať (výstup z main_analysis.py)
    # Tento priečinok by mal obsahovať cliques.json a claws.json
    input_dir = "analyzed_subgraphs"

    # Nový priečinok pre výstupy LLM analýzy
    output_dir = "llm_analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    # Spracujeme kliky
    cliques_input_path = os.path.join(input_dir, "cliques.json")
    cliques_output_path = os.path.join(output_dir, "cliques_analysis.json")
    process_subgraph_file(cliques_input_path, cliques_output_path, "clique")

    # Spracujeme "claw" podgrafy
    claws_input_path = os.path.join(input_dir, "claws.json")
    claws_output_path = os.path.join(output_dir, "claws_analysis.json")
    process_subgraph_file(claws_input_path, claws_output_path, "claw")

    print("\n" + "=" * 50)
    print("Všetky analýzy boli úspešne dokončené.")
    print(f"Výsledky nájdete v priečinku '{output_dir}'.")


if __name__ == "__main__":
    run_llm_analysis()