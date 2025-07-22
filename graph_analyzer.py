import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, LinearSegmentedColormap
from itertools import combinations
import math
from typing import Dict, Tuple, List, Set
import io
import base64


# --- Funkcie na analýzu (zostávajú bez zmeny) ---
def construct_graph(trimmed_matrix, nodes):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i in range(len(trimmed_matrix)):
        for j in range(i + 1, len(trimmed_matrix)):
            if trimmed_matrix[i][j] != 0:
                G.add_edge(nodes[i], nodes[j], weight=trimmed_matrix[i][j])
    return G


def find_claw_subgraphs(graph):
    claws = []
    for central_node in graph.nodes():
        neighbors = list(graph.neighbors(central_node))
        if len(neighbors) >= 3:
            for trio in combinations(neighbors, 3):
                if not any(graph.has_edge(u, v) for u, v in combinations(trio, 2)):
                    claws.append((central_node,) + trio)
    return claws


def find_cliques(graph):
    maximal_cliques = list(nx.find_cliques(graph))
    cliques = set()
    for clique in maximal_cliques:
        if 3 <= len(clique) <= 5:
            cliques.add(tuple(sorted(clique)))
        for size in range(3, min(len(clique), 5) + 1):
            for subset in combinations(clique, size):
                cliques.add(tuple(sorted(subset)))
    return [list(clique) for clique in cliques]


# --- Pomocné funkcie pre vizualizáciu ---
def generate_gradient_colors(n=100):
    cmap = LinearSegmentedColormap.from_list("green_blue", ["#21B9DE", "#B9DE21"])
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def draw_stylized_nodes(ax, pos, labels, node_colors, text_size=10, padding=0.4):
    for node, (x, y) in pos.items():
        label = labels.get(node, str(node))
        color = node_colors.get(node, "#cccccc")
        ax.text(x, y, label, fontsize=text_size, ha='center', va='center', color='white', zorder=4,
                bbox=dict(boxstyle=f"round,pad={padding}", facecolor=color, edgecolor="#222222", linewidth=1, zorder=3))


def fig_to_base64(fig):
    """Konvertuje Matplotlib figúru na Base64 reťazec pre použitie v HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)  # Dôležité: zatvoríme figúru, aby sme šetrili pamäť
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


# --- NOVÉ Funkcie na generovanie vizualizácií ako Base64 ---

def create_single_clique_viz_base64(graph, clique_nodes):
    """Vytvorí vizualizáciu pre JEDNU kliku a vráti ju ako Base64 reťazec."""
    if not clique_nodes: return None

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title(f"Clique ({len(clique_nodes)} nodes)", fontsize=10)
    ax.axis("off")

    subgraph = graph.subgraph(clique_nodes)
    pos = nx.circular_layout(subgraph)

    GRADIENT_COLORS = generate_gradient_colors(5)
    node_colors = {node: GRADIENT_COLORS[i % 5] for i, node in enumerate(subgraph.nodes())}

    nx.draw_networkx_edges(subgraph, pos, ax=ax, edge_color="gray", alpha=0.7, width=1.5)
    labels = {node: str(node) for node in subgraph.nodes()}
    draw_stylized_nodes(ax, pos, labels, node_colors, text_size=9, padding=0.5)

    return fig_to_base64(fig)


def create_single_claw_viz_base64(graph, claw_nodes):
    """Vytvorí vizualizáciu pre JEDEN claw podgraf a vráti ju ako Base64 reťazec."""
    if not claw_nodes: return None

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title(f"Claw (Center: {claw_nodes[0]})", fontsize=10)
    ax.axis('off')

    subgraph = graph.subgraph(claw_nodes)
    main_node = claw_nodes[0]
    neighbors = list(claw_nodes[1:])
    pos = {main_node: (0.2, 0.5)}
    for i, neighbor in enumerate(neighbors):
        pos[neighbor] = (0.8, (i + 1) / (len(neighbors) + 1))

    node_colors = {main_node: "#D9534F"}
    node_colors.update({neighbor: "#0275D8" for neighbor in neighbors})

    nx.draw_networkx_edges(subgraph, pos, ax=ax, edge_color="gray", alpha=0.7, width=1)
    labels = {node: str(node) for node in subgraph.nodes()}
    draw_stylized_nodes(ax, pos, labels, node_colors, text_size=9, padding=0.4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return fig_to_base64(fig)


def create_full_graph_viz_base64(graph):
    """
    Vytvorí vizualizáciu celého grafu a vráti ju ako base64 reťazec.
    """
    if graph is None or not isinstance(graph, nx.Graph):
        return None

    pos = nx.spring_layout(graph, seed=42)
    node_labels = {node: node for node in graph.nodes()}

    fig, ax = plt.subplots(figsize=(12, 10))

    # Kreslenie uzlov a hrán
    nx.draw_networkx_nodes(graph, pos, node_color='#A0CBE2', node_size=2000, ax=ax)
    nx.draw_networkx_edges(graph, pos, width=2, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10, font_weight='bold', ax=ax)

    ax.set_title("Full Correlation Graph", fontsize=20)
    ax.axis('off')

    return fig_to_base64(fig)