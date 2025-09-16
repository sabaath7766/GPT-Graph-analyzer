# Súbor: graph_utils.py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, LinearSegmentedColormap
from itertools import combinations
import io
import base64
from matplotlib.patches import FancyArrowPatch

# ZMENA: Importujeme náš nový modul pre pokročilý layout
import advanced_graph_layout


# --- Základné funkcie pre prácu s grafom (bez zmeny) ---
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
                    claws.append(list((central_node,) + trio))
    return claws


def find_cliques(graph):
    maximal_cliques = list(nx.find_cliques(graph))
    cliques = set()
    for clique in maximal_cliques:
        if len(clique) >= 3:
            clique_tuple = tuple(sorted(clique))
            for size in range(3, min(len(clique), 5) + 1):
                for subset in combinations(clique_tuple, size):
                    cliques.add(tuple(sorted(subset)))
    return [list(clique) for clique in cliques]


# --- Pomocné funkcie pre vizualizáciu (bez zmeny) ---
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
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


# --- Vizualizácie podgrafov (bez zmeny) ---
def create_single_clique_viz_base64(graph, clique_nodes):
    # ... kód zostáva rovnaký ...
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
    # ... kód zostáva rovnaký ...
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


# --- HLAVNÁ ZMENA: Plne opravená a funkčná vizualizácia celého grafu ---
def create_full_graph_viz_base64(graph):
    """
    Vytvorí vizualizáciu celého grafu pomocou fyzikálnej simulácie a zakrivených hrán.
    """
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        return None

    print("\nGenerujem vizualizáciu celého grafu pomocou fyzikálneho rozloženia...")

    # Krok 1: Získanie pozícií pomocou novej metódy z advanced_graph_layout.py
    pos = advanced_graph_layout.generate_physics_based_pos(graph)

    node_labels = {node: node for node in graph.nodes()}
    fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
    ax.set_title("Full Correlation Graph (Physics-Based Layout)", fontsize=20)
    ax.axis('off')

    # Krok 2: Kreslenie uzlov
    node_size = 2000 / (1 + 0.05 * len(graph.nodes()))  # Dynamická veľkosť uzlov
    nx.draw_networkx_nodes(graph, pos, node_color='#2B6CB0', node_size=node_size, ax=ax, edgecolors='#EBF8FF')
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, font_weight='bold', font_color='white', ax=ax)

    # Krok 3: Kreslenie hrán so zakrivením
    print("Optimalizujem kreslenie hrán, aby sa neprekrývali s uzlami...")
    for u, v in graph.edges():
        connectionstyle = advanced_graph_layout.find_best_curve_for_edge(graph, pos, u, v)
        edge = FancyArrowPatch(
            posA=pos[u],
            posB=pos[v],
            arrowstyle='-',
            color='gray',
            linewidth=1.2,
            connectionstyle=connectionstyle,
            shrinkA=18,
            shrinkB=18,
            zorder=0
        )
        ax.add_patch(edge)

    # Prispôsobenie hraníc plátna
    if pos:
        x_vals, y_vals = zip(*pos.values())
        ax.set_xlim(min(x_vals) - 2, max(x_vals) + 2)
        ax.set_ylim(min(y_vals) - 2, max(y_vals) + 2)

    fig.tight_layout()
    print("Vizualizácia hotová.")
    return fig_to_base64(fig)