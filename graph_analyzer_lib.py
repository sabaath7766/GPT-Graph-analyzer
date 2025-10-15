# Súbor: graph_analyzer_lib.py

import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import multiprocessing as mp
import graph_utils
import llm_logic

worker_llm = None


def init_worker():
    global worker_llm
    worker_llm = llm_logic.load_llm_model()


def process_task(task_data):
    """
    Spracuje jeden podgraf. Generuje už len dáta pre Cytoscape.
    """
    global worker_llm
    nodes, subgraph_type, descriptions, graph_tuple = task_data

    graph = nx.Graph()
    graph.add_nodes_from(graph_tuple[0])
    graph.add_edges_from(graph_tuple[1])

    final_answer, original_responses = llm_logic.get_synthesized_answer(
        worker_llm, nodes, descriptions, subgraph_type
    )

    subgraph = graph.subgraph(nodes)
    cytoscape_data = graph_utils.graph_to_cytoscape_json(subgraph)

    return {
        "subgraph_type": subgraph_type,
        "nodes_data": nodes,
        "synthesized_analysis": final_answer,
        "original_responses": original_responses,
        "visualization_b64": None,
        "cytoscape_data": cytoscape_data,
        "attributes": {node: descriptions.get(node, "No description.") for node in nodes}
    }


class GraphAnalyzer:
    def __init__(self, csv_path: str, descriptions_json_path: str, alpha_normalized: float = 0.5):
        if not 0.0 <= alpha_normalized <= 1.0: raise ValueError(
            "Parameter alpha_normalized musí byť v rozsahu od 0.0 do 1.0.")
        self.alpha_normalized = alpha_normalized
        self.alpha = alpha_normalized * 0.3
        print(f"Inicializácia s normalizovaným alfa = {self.alpha_normalized:.2f} (interná hodnota = {self.alpha:.4f})")
        self.csv_path = Path(csv_path)
        self.descriptions_json_path = Path(descriptions_json_path)
        if not self.csv_path.exists() or not self.descriptions_json_path.exists(): raise FileNotFoundError(
            "Jeden zo vstupných súborov (CSV alebo JSON) neexistuje.")
        self.dataset_name = self.csv_path.stem
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.output_path = self.results_dir / f"{self.dataset_name}-alpha_norm{self.alpha_normalized:.1f}.json"
        self.df, self.attribute_descriptions, self.graph = None, None, None

    def _load_and_prepare_data(self):
        print(f"Načítavam dáta z '{self.csv_path}'...")
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.select_dtypes(include=np.number)
        if self.df.columns[0].lower() in ['date', 'time', 'unnamed: 0']: self.df = self.df.drop(
            columns=self.df.columns[0])
        print(f"Načítavam popisy atribútov z '{self.descriptions_json_path}'...")
        with open(self.descriptions_json_path, 'r', encoding='utf-8') as f: self.attribute_descriptions = json.load(f)

    def _perform_graph_analysis(self):
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
        print(
            f"Graf pôvodne vytvorený s {self.graph.number_of_nodes()} uzlami a {self.graph.number_of_edges()} hranami.")
        isolates = list(nx.isolates(self.graph))
        if isolates:
            print(f"Odstraňujem {len(isolates)} izolovaných bodov (bez hrán): {isolates}")
            self.graph.remove_nodes_from(isolates)
        print(f"Graf po očistení má {self.graph.number_of_nodes()} uzlov a {self.graph.number_of_edges()} hrán.")
        claws = graph_utils.find_claw_subgraphs(self.graph)
        cliques = graph_utils.find_cliques(self.graph)
        print(f"Nájdených {len(cliques)} ptých a {len(claws)} hotspotov.")
        return cliques, claws

    def _perform_llm_analysis_multiprocess(self, cliques: list, claws: list, num_workers: int = 2,
                                           timeout: int = None) -> list:
        print(f"\nSpúšťam paralelnú LLM analýzu s {num_workers} procesmi...")
        if timeout: print(f"UPOZORNENIE: Analýza bude automaticky ukončená po {timeout / 60:.1f} minútach.")
        tasks = []
        graph_tuple = (list(self.graph.nodes()), list(self.graph.edges()))
        for c in cliques: tasks.append((c, 'clique', self.attribute_descriptions, graph_tuple))
        for c in claws: tasks.append((c, 'claw', self.attribute_descriptions, graph_tuple))
        all_results = []
        start_time = time.time()
        with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
            async_results = [pool.apply_async(process_task, (task,)) for task in tasks]
            with tqdm(total=len(tasks), desc="Spracúvam podgrafy") as pbar:
                while async_results:
                    if timeout and (time.time() - start_time) > timeout:
                        print(f"\n!!!!!! Časový limit {timeout / 60:.1f} minút vypršal. Ukončujem analýzu. !!!!!!")
                        print(f"Spracovaných {len(all_results)} z {len(tasks)} podgrafov.")
                        pool.terminate()
                        break
                    remaining_results = []
                    processed_this_loop = 0
                    for res in async_results:
                        if res.ready():
                            all_results.append(res.get())
                            processed_this_loop += 1
                        else:
                            remaining_results.append(res)
                    if processed_this_loop > 0: pbar.update(processed_this_loop)
                    async_results = remaining_results
                    time.sleep(0.5)
        print("\nParalelná LLM analýza dokončená.")
        return all_results

    def analyze(self, force_reanalyze: bool = False, num_llm_workers: int = 2, timeout_minutes: float = None):
        if not force_reanalyze and self.output_path.exists():
            print(f"Nájdený existujúci súbor s výsledkami: '{self.output_path}'. Načítavam ho.")
            with open(self.output_path, 'r', encoding='utf-8') as f: return json.load(f)
        print("=" * 20 + " SPUSTENIE NOVEJ ANALÝZY " + "=" * 20)
        self._load_and_prepare_data()
        cliques, claws = self._perform_graph_analysis()
        timeout_seconds = timeout_minutes * 60 if timeout_minutes is not None else None
        analyzed_subgraphs = self._perform_llm_analysis_multiprocess(cliques, claws, num_workers=num_llm_workers,
                                                                     timeout=timeout_seconds)
        print("\nGenerujem finálny výstupný súbor...")
        final_output = {"metadata": {"dataset_name": self.dataset_name, "csv_path": str(self.csv_path),
                                     "descriptions_path": str(self.descriptions_json_path),
                                     "alpha_internal": self.alpha, "alpha_normalized": self.alpha_normalized,
                                     "total_analyses_found": len(cliques) + len(claws),
                                     "total_analyses_processed": len(analyzed_subgraphs),
                                     "timed_out": (timeout_seconds is not None) and (
                                             len(analyzed_subgraphs) < (len(cliques) + len(claws))),
                                     "attribute_descriptions": self.attribute_descriptions},
                        "full_graph_cytoscape_data": graph_utils.graph_to_cytoscape_json(self.graph),
                        "analyses": analyzed_subgraphs}
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Analýza úspešne dokončená. Výsledky uložené do '{self.output_path}'.")
        return final_output

    def get_structure_counts(self):
        """
        Vykoná iba štrukturálnu analýzu bez zapojenia LLM a vráti počty
        nájdených štruktúr (ptých a hotspotov).
        Je to rýchla metóda určená pre experimenty.
        """
        # Tieto dve metódy sú všetko, čo potrebujeme na nájdenie štruktúr
        self._load_and_prepare_data()
        cliques, claws = self._perform_graph_analysis()

        # Vrátime iba počty
        return len(cliques), len(claws)
