import json
import os
from data import ATTRIBUTE_DESCRIPTIONS
import llm_logic


def run_llm_analysis():
    """
    Načíta podgrafy, spustí pre ne LLM analýzu a uloží výsledky.
    """
    # Cesta k súboru s podgrafmi z predchádzajúceho kroku
    subgraphs_path = os.path.join("analyzed_subgraphs", "subgraphs_to_analyze.json")

    # Načítanie podgrafov
    try:
        with open(subgraphs_path, 'r') as f:
            subgraphs = json.load(f)
    except FileNotFoundError:
        print(f"Chyba: Súbor '{subgraphs_path}' nebol nájdený.")
        print("Najprv spustite skript 'main_analysis.py' na vygenerovanie podgrafov.")
        return

    print(f"Načítaných {len(subgraphs)} podgrafov na analýzu.")

    # Pripravíme si zoznam na uloženie výsledkov
    all_results = []

    # Prejdeme každý podgraf a analyzujeme ho
    for i, subgraph_nodes in enumerate(subgraphs):
        print("\n" + "=" * 50)
        print(f"Spracúvam podgraf #{i + 1}: {subgraph_nodes}")
        print("=" * 50)

        # Zavoláme hlavnú funkciu z našej knižnice
        final_answer, original_responses = llm_logic.get_synthesized_answer(
            subgraph_nodes,
            ATTRIBUTE_DESCRIPTIONS
        )

        print("\n--- Pôvodné odpovede od LLM ---")
        for j, r in enumerate(original_responses):
            print(f"Odpoveď {j + 1}: {r}")

        print("\n--- FINÁLNA SYNTETIZOVANÁ ODPOVEĎ ---")
        print(final_answer)

        # Uložíme výsledok pre tento podgraf
        all_results.append({
            "subgraph_nodes": subgraph_nodes,
            "synthesized_analysis": final_answer,
            "original_responses": original_responses
        })

    # Uloženie všetkých výsledkov do jedného JSON súboru
    results_path = os.path.join("analyzed_subgraphs", "llm_analysis_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"Analýza je dokončená! Všetky výsledky boli uložené do súboru '{results_path}'.")


if __name__ == "__main__":
    run_llm_analysis()