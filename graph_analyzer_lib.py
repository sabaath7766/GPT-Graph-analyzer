# Súbor: graph_analyzer_lib.py

import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import networkx as nx

#  importy pre multiprocessing
import multiprocessing as mp

# moje moduly
import graph_utils
import llm_logic

# Globálna premenná pre uloženie inštancie modelu v každom worker procese
worker_llm = None


def init_worker():
    """
    Inicializačná funkcia pre každý worker proces.
    Načíta model do globálnej premennej 'worker_llm'.
    """
    global worker_llm
    print(f"Inicializujem worker proces {os.getpid()}...")
    worker_llm = llm_logic.load_llm_model()


def process_task(task_data):
    """
    Funkcia, ktorú vykonáva každý worker. Spracuje jednu úlohu (jeden podgraf).
    Nemôže byť metódou triedy, lebo by sa zle prenášala medzi procesmi.
    """
    global worker_llm
    nodes, subgraph_type, descriptions, graph_tuple = task_data

    # Rekonštrukcia NetworkX grafu z tuple (jednoduchší prenos medzi procesmi)
    graph = graph_utils.nx.Graph()
    graph.add_nodes_from(graph_tuple[0])
    graph.add_edges_from(graph_tuple[1])

    # LLM Analýza
    final_answer, original_responses = llm_logic.get_synthesized_answer(
        worker_llm, nodes, descriptions, subgraph_type
    )

    # Vizualizácia
    visualization_b64 = None
    if subgraph_type == 'clique':
        visualization_b64 = graph_utils.create_single_clique_viz_base64(graph, nodes)
    elif subgraph_type == 'claw':
        visualization_b64 = graph_utils.create_single_claw_viz_base64(graph, nodes)

    return {
        "subgraph_type": subgraph_type,
        "nodes_data": nodes,
        "synthesized_analysis": final_answer,
        "original_responses": original_responses,
        "visualization_b64": visualization_b64,
        "attributes": {node: descriptions.get(node, "No description.") for node in nodes}
    }


class GraphAnalyzer:
    # Metódy __init__, _load_and_prepare_data, _perform_graph_analysis zostávajú rovnaké
    def __init__(self, csv_path: str, descriptions_json_path: str, alpha: float = 0.1):
        """
        Inicializuje analyzátor so vstupnými súbormi a parametrami.

        Args:
            csv_path (str): Cesta k .csv súboru s dátami.
            descriptions_json_path (str): Cesta k .json súboru s popismi atribútov.
            alpha (float): Parameter pre orezanie korelačnej matice.
        """
        self.csv_path = Path(csv_path)
        self.descriptions_json_path = Path(descriptions_json_path)
        self.alpha = alpha

        if not self.csv_path.exists() or not self.descriptions_json_path.exists():
            raise FileNotFoundError("Jeden zo vstupných súborov (CSV alebo JSON) neexistuje.")

        self.dataset_name = self.csv_path.stem
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.output_path = self.results_dir / f"{self.dataset_name}-alpha{self.alpha}.json"

        self.df = None
        self.attribute_descriptions = None
        self.graph = None

    def _load_and_prepare_data(self):
        """Načíta dáta z CSV a popisy z JSON."""
        print(f"Načítavam dáta z '{self.csv_path}'...")
        self.df = pd.read_csv(self.csv_path)
        if self.df.columns[0].lower() in ['date', 'time', 'unnamed: 0']:
            self.df = self.df.drop(columns=self.df.columns[0])

        print(f"Načítavam popisy atribútov z '{self.descriptions_json_path}'...")
        with open(self.descriptions_json_path, 'r', encoding='utf-8') as f:
            self.attribute_descriptions = json.load(f)

    def _perform_graph_analysis(self):
        """Vykoná korelačnú a grafovú analýzu."""
        print("Vykonávam korelačnú a grafovú analýzu...")
        correlation_matrix = self.df.corr()

        cor_values = correlation_matrix.values[np.where(~np.eye(correlation_matrix.shape[0], dtype=bool))]
        threshold = (np.max(cor_values) + np.mean(cor_values)) / 2 + self.alpha
        print(f"Dynamický prah pre korelácie: {threshold:.4f}")
        trimmed_matrix_np = np.where(np.abs(correlation_matrix) > threshold, correlation_matrix, 0)
        np.fill_diagonal(trimmed_matrix_np, 0)
        trimmed_matrix = pd.DataFrame(trimmed_matrix_np, index=correlation_matrix.index,
                                      columns=correlation_matrix.columns)

        nodes = trimmed_matrix.columns.tolist()
        self.graph = graph_utils.construct_graph(trimmed_matrix.values, nodes)
        print(f"Graf vytvorený s {self.graph.number_of_nodes()} uzlami a {self.graph.number_of_edges()} hranami.")

        isolates = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolates)
        if isolates:
            print(f"Odstraňujem {len(isolates)} izolovaných bodov (bez hrán): {isolates}")

        claws = graph_utils.find_claw_subgraphs(self.graph)
        cliques = graph_utils.find_cliques(self.graph)
        print(f"Nájdených {len(cliques)} klík a {len(claws)} 'claw' podgrafov.")

        return cliques, claws

    def _perform_llm_analysis_multiprocess(self, cliques: list, claws: list, num_workers: int = 2) -> list:
        """Spracuje všetky podgrafy paralelne pomocou multiprocessing.Pool."""
        print(f"\nSpúšťam paralelnú LLM analýzu s {num_workers} procesmi...")

        # Príprava úloh pre workerov
        tasks = []
        # Prevod grafu na jednoduchší formát pre prenos medzi procesmi
        graph_tuple = (list(self.graph.nodes()), list(self.graph.edges()))

        for c in cliques:
            tasks.append((c, 'clique', self.attribute_descriptions, graph_tuple))
        for c in claws:
            tasks.append((c, 'claw', self.attribute_descriptions, graph_tuple))

        all_results = []
        # Vytvorenie a správa pool-u workerov
        with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
            # Použijeme imap_unordered pre priebežné spracovanie výsledkov
            # a tqdm pre zobrazenie priebehu
            results_iterator = pool.imap_unordered(process_task, tasks)

            for result in tqdm(results_iterator, total=len(tasks), desc="Spracúvam podgrafy"):
                all_results.append(result)

        print("Paralelná LLM analýza dokončená.")
        return all_results

    def analyze(self, force_reanalyze: bool = False, num_llm_workers: int = 2):
        """
        Hlavná metóda, ktorá spúšťa celý proces analýzy.

        Args:
            force_reanalyze (bool): Ak je True, analýza sa vykoná znova.
            num_llm_workers (int): Počet paralelných procesov pre LLM analýzu.
        """
        if not force_reanalyze and self.output_path.exists():
            print(f"Nájdený existujúci súbor s výsledkami: '{self.output_path}'. Načítavam ho.")
            with open(self.output_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        print("=" * 20 + " SPUSTENIE NOVEJ ANALÝZY " + "=" * 20)

        self._load_and_prepare_data()
        cliques, claws = self._perform_graph_analysis()

        # Zmenené volanie
        analyzed_subgraphs = self._perform_llm_analysis_multiprocess(cliques, claws, num_workers=num_llm_workers)

        print("Generujem finálny výstupný súbor...")
        final_output = {
            "metadata": {
                "dataset_name": self.dataset_name,
                "csv_path": str(self.csv_path),
                "descriptions_path": str(self.descriptions_json_path),
                "alpha": self.alpha,
                "total_analyses": len(analyzed_subgraphs),
                "attribute_descriptions": self.attribute_descriptions
            },
            "full_graph_visualization_b64": graph_utils.create_full_graph_viz_base64(self.graph),
            "analyses": analyzed_subgraphs
        }

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Analýza úspešne dokončená. Výsledky uložené do '{self.output_path}'.")

        return final_output