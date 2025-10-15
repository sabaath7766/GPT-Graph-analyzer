# Súbor: graph_utils.py

import base64
import io
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import advanced_graph_layout


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
    return [list(c) for c in cliques]


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


def create_full_graph_viz_base64(graph):
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        return None
    print("\nGenerujem vizualizáciu celého grafu pomocou fyzikálneho rozloženia...")
    pos = advanced_graph_layout.generate_physics_based_pos(graph)
    node_labels = {node: node for node in graph.nodes()}
    fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
    ax.set_title("Full Correlation Graph (Physics-Based Layout)", fontsize=20)
    ax.axis('off')
    node_size = 2000 / (1 + 0.05 * len(graph.nodes()))
    nx.draw_networkx_nodes(graph, pos, node_color='#2B6CB0', node_size=node_size, ax=ax, edgecolors='#EBF8FF')
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, font_weight='bold', font_color='white', ax=ax)
    print("Optimalizujem kreslenie hrán...")
    for u, v in graph.edges():
        connectionstyle = advanced_graph_layout.find_best_curve_for_edge(graph, pos, u, v)
        edge = FancyArrowPatch(posA=pos[u], posB=pos[v], arrowstyle='-', color='gray', linewidth=1.2,
                               connectionstyle=connectionstyle, shrinkA=18, shrinkB=18, zorder=0)
        ax.add_patch(edge)
    if pos:
        x_vals, y_vals = zip(*pos.values())
        ax.set_xlim(min(x_vals) - 2, max(x_vals) + 2)
        ax.set_ylim(min(y_vals) - 2, max(y_vals) + 2)
    fig.tight_layout()
    print("Vizualizácia hotová.")
    return fig_to_base64(fig)


def graph_to_cytoscape_json(graph):
    if not isinstance(graph, nx.Graph):
        return {"nodes": [], "edges": []}
    nodes = [{"data": {"id": node}} for node in graph.nodes()]
    edges = []
    for u, v, data in graph.edges(data=True):
        edge_data = {"source": u, "target": v}
        edge_data.update(data)
        edges.append({"data": edge_data})
    return {"nodes": nodes, "edges": edges}
