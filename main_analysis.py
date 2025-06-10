import json

import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

import graph_analyzer


def load_and_prepare_data():
    """Načíta a pripraví dataset."""
    filepath = 'datasets/energydata.csv'
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Súbor '{filepath}' nebol nájdený. Uistite sa, že existuje.")
        exit()
    # Preskočíme stĺpec s dátumom a svetlami
    df = df.drop(columns=['date', 'lights'])
    return df


def compute_correlation_matrix(df):
    """Vypočíta korelačnú maticu."""
    return df.corr()


def trim_correlation_matrix(correlation_matrix, alpha=0.1):
    """Oseká korelačnú maticu na základe dynamického prahu."""
    cor_values = correlation_matrix.values[np.where(~np.eye(correlation_matrix.shape[0], dtype=bool))]
    cor_max = np.max(cor_values)
    cor_mean = np.mean(cor_values)
    # threshold = cor_max - alpha * (cor_max - cor_mean)  # Vylepšená formula prahu
    threshold = (cor_max + cor_mean) / 2 + alpha
    print(f"Dynamický prah pre korelácie: {threshold:.4f}")

    trimmed_matrix_np = np.where(np.abs(correlation_matrix) > threshold, correlation_matrix, 0)
    # Diagonálu necháme nulovú
    np.fill_diagonal(trimmed_matrix_np, 0)

    return pd.DataFrame(trimmed_matrix_np, index=correlation_matrix.index, columns=correlation_matrix.columns)


def main():
    """Hlavná funkcia na vykonanie všetkých krokov."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Výstupy sa budú ukladať do priečinka: '{output_dir}'")

    # Krok 1-3: Príprava dát a matice
    df = load_and_prepare_data()
    correlation_matrix = compute_correlation_matrix(df)
    trimmed_matrix = trim_correlation_matrix(correlation_matrix, alpha=0.1)

    # Krok 4: Vytvorenie grafu pomocou knižnice
    nodes = trimmed_matrix.columns.tolist()
    G = graph_analyzer.construct_graph(trimmed_matrix.values, nodes)
    print(f"Graf vytvorený s {G.number_of_nodes()} uzlami a {G.number_of_edges()} hranami.")

    # Krok 5-6: Nájdenie podgrafov pomocou knižnice
    claws = graph_analyzer.find_claw_subgraphs(G)
    cliques = graph_analyzer.find_cliques(G)
    print(f"Nájdených {len(cliques)} klík a {len(claws)} 'claw' podgrafov.")

    # --- Krok 7: Vizualizácie (logika presunutá sem) ---
    print("\nGenerujem vizualizácie...")

    fig_main = graph_analyzer.create_main_graph_figure(G)
    if fig_main:
        main_graph_path = os.path.join(output_dir, "main_graph_hierarchical.svg")
        fig_main.savefig(main_graph_path, format="svg", bbox_inches='tight')
        plt.close(fig_main)
        print(f"- Vizualizácia hlavného grafu uložená do: '{main_graph_path}'")

    clique_figures = graph_analyzer.create_clique_figures(G, cliques)
    if clique_figures:
        for size, fig in clique_figures.items():
            clique_path = os.path.join(output_dir, f"cliques_size_{size}.svg")
            fig.savefig(clique_path, format="svg", bbox_inches='tight')
            plt.close(fig)
            print(f"- Vizualizácia klík veľkosti {size} uložená do: '{clique_path}'")

    fig_claws = graph_analyzer.create_claw_subgraphs_figure(G, claws)
    if fig_claws:
        claws_path = os.path.join(output_dir, "claw_subgraphs.svg")
        fig_claws.savefig(claws_path, format="svg", bbox_inches='tight')
        plt.close(fig_claws)
        print(f"- Vizualizácia 'claw' podgrafov uložená do: '{claws_path}'")

    # --- Uloženie podgrafov ako JSON ---
    print("\nUkladám podgrafy ako JSON...")

    # Konvertuj kliky a claw podgrafy do zoznamov uzlov
    cliques_json = [list(clique) for clique in cliques]
    claws_json = [list(claw) for claw in claws]

    cliques_path = os.path.join(output_dir, "cliques.json")
    claws_path = os.path.join(output_dir, "claws.json")

    with open(cliques_path, 'w', encoding='utf-8') as f:
        json.dump(cliques_json, f, indent=2, ensure_ascii=False)
        print(f"- Kliky uložené do: '{cliques_path}'")

    with open(claws_path, 'w', encoding='utf-8') as f:
        json.dump(claws_json, f, indent=2, ensure_ascii=False)
        print(f"- Claw podgrafy uložené do: '{claws_path}'")

    try:
        print("\nPripravujem podgrafy pre LLM analýzu...")

        subgraphs_for_llm = []

        # Pridaj všetky claw podgrafy
        subgraphs_for_llm.extend([list(claw) for claw in claws])

        # Pridaj kliky veľkosti 3 až 5
        for clique in cliques:
            if 3 <= len(clique) <= 5:
                subgraphs_for_llm.append(list(clique))

        # Ulož ich do priečinka analyzed_subgraphs
        os.makedirs("analyzed_subgraphs", exist_ok=True)
        llm_subgraphs_path = os.path.join("analyzed_subgraphs", "subgraphs_to_analyze.json")
        with open(llm_subgraphs_path, 'w', encoding='utf-8') as f:
            json.dump(subgraphs_for_llm, f, indent=2, ensure_ascii=False)
            print(f"- Podgrafy pre LLM analýzu uložené do: '{llm_subgraphs_path}'")
    except Exception as e:
        print(f"Chyba pri ukladaní podgrafov pre LLM analýzu: {e}")

    print("\nProces je dokončený.")


if __name__ == "__main__":
    main()