import pandas as pd
import numpy as np
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Pomocné moduly, ktoré zostávajú
import graph_analyzer
import llm_logic


class GraphAnalyzer:
    """
    Centrálna trieda na analýzu grafov z korelačných dát.
    Zastrešuje načítanie dát, analýzu, LLM spracovanie a ukladanie výsledkov.
    """

    def __init__(self, csv_path: str, json_descriptions_path: str, alpha: float = 0.1):
        """
        Inicializuje analyzátor so vstupnými súbormi a parametrami.

        Args:
            csv_path (str): Cesta k .csv súboru s dátami.
            json_descriptions_path (str): Cesta k .json súboru s popismi atribútov.
            alpha (float): Parameter pre orezanie korelačnej matice.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dátový súbor nebol nájdený: {csv_path}")
        if not os.path.exists(json_descriptions_path):
            raise FileNotFoundError(f"Súbor s popismi nebol nájdený: {json_descriptions_path}")

        self.csv_path = csv_path
        self.alpha = alpha
        self.dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

        with open(json_descriptions_path, 'r', encoding='utf-8') as f:
            self.descriptions = json.load(f)

        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Názov výstupného súboru podľa konvencie
        self.result_filepath = os.path.join(self.results_dir, f"{self.dataset_name}-{self.alpha}.json")

    def _analyze_single_subgraph(self, subgraph_task: dict) -> dict:
        """
        Spracuje jeden podgraf pomocou LLM.
        Táto metóda je navrhnutá na paralelné spúšťanie.
        """
        subgraph_type = subgraph_task['type']
        nodes = subgraph_task['nodes']

        print(f"  -> Spúšťam LLM analýzu pre '{subgraph_type}' s uzlami: {nodes}...")

        final_answer, original_responses = llm_logic.get_synthesized_answer(
            nodes, self.descriptions, subgraph_type
        )

        return {
            "subgraph_type": subgraph_type,
            "nodes_data": nodes,
            "synthesized_analysis": final_answer,
        }

    def analyze(self):
        """
        Hlavná metóda, ktorá spúšťa celý proces analýzy.
        Ak už existuje výsledok, načíta ho z JSON súboru.
        """
        # 1. Perzistencia dát: Kontrola existujúceho výsledku
        if os.path.exists(self.result_filepath):
            print(f"✅ Nájdený existujúci výsledok. Načítavam z '{self.result_filepath}'...")
            with open(self.result_filepath, 'r', encoding='utf-8') as f:
                return json.load(f)

        print(f"🚀 Spúšťam novú analýzu pre dataset '{self.dataset_name}' s alpha={self.alpha}")
        start_time = time.time()

        # 2. Dátová analýza (z pôvodného main_analysis.py)
        print("Krok 1: Príprava dát a výpočet korelácie...")
        df = pd.read_csv(self.csv_path)
        df = df.drop(columns=[col for col in ['date', 'lights'] if col in df.columns])

        correlation_matrix = df.corr()
        trimmed_matrix = self._trim_correlation_matrix(correlation_matrix, self.alpha)

        nodes = trimmed_matrix.columns.tolist()
        G = graph_analyzer.construct_graph(trimmed_matrix.values, nodes)
        print(f"Graf vytvorený s {G.number_of_nodes()} uzlami a {G.number_of_edges()} hranami.")

        claws = graph_analyzer.find_claw_subgraphs(G)
        cliques = graph_analyzer.find_cliques(G)
        print(f"Nájdených {len(cliques)} klík a {len(claws)} 'claw' podgrafov.")

        # 3. Paralelná LLM analýza (Optimalizácia výkonu)
        print("\nKrok 2: Spúšťanie paralelnej LLM analýzy...")
        subgraph_tasks = []
        for clique in cliques:
            subgraph_tasks.append({'type': 'clique', 'nodes': clique})
        for claw in claws:
            subgraph_tasks.append({'type': 'claw', 'nodes': list(claw)})

        llm_analyses = []
        # Použitie ThreadPoolExecutor pre paralelné volania
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # `map` zachováva poradie, čo je užitočné
            llm_analyses = list(executor.map(self._analyze_single_subgraph, subgraph_tasks))

        print("✅ Všetky LLM analýzy dokončené.")

        # 4. Generovanie vizualizácie celého grafu
        print("\nKrok 3: Generovanie vizualizácie grafu...")
        full_graph_viz = graph_analyzer.create_full_graph_viz_base64(G)

        # 5. Zostavenie a uloženie finálneho výsledku
        final_result = {
            "dataset_name": self.dataset_name,
            "alpha": self.alpha,
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "found_cliques_count": len(cliques),
            "found_claws_count": len(claws),
            "full_graph_visualization": full_graph_viz,
            "llm_analyses": llm_analyses
        }

        print(f"\nKrok 4: Ukladanie výsledkov do '{self.result_filepath}'...")
        with open(self.result_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)

        end_time = time.time()
        print(f"🎉 Analýza úspešne dokončená za {end_time - start_time:.2f} sekúnd.")

        return final_result

    def _trim_correlation_matrix(self, correlation_matrix, alpha):
        """Pomocná funkcia na orezanie matice (z pôvodného main_analysis.py)."""
        cor_values = correlation_matrix.values[np.where(~np.eye(correlation_matrix.shape[0], dtype=bool))]
        cor_max = np.max(np.abs(cor_values))
        cor_mean = np.mean(np.abs(cor_values))
        threshold = (cor_max + cor_mean) / 2 + alpha
        print(f"Dynamický prah pre korelácie: {threshold:.4f}")

        trimmed_matrix_np = np.where(np.abs(correlation_matrix) > threshold, correlation_matrix, 0)
        np.fill_diagonal(trimmed_matrix_np, 0)

        return pd.DataFrame(trimmed_matrix_np, index=correlation_matrix.index, columns=correlation_matrix.columns)