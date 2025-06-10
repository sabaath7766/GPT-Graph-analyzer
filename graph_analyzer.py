import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, LinearSegmentedColormap
from itertools import combinations
import math
from typing import Dict, Tuple, List, Set


def construct_graph(trimmed_matrix, nodes):
    """Vytvorí graf z upravenej korelačnej matice."""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i in range(len(trimmed_matrix)):
        for j in range(i + 1, len(trimmed_matrix)):
            if trimmed_matrix[i][j] != 0:
                G.add_edge(nodes[i], nodes[j], weight=trimmed_matrix[i][j])
    return G


def find_claw_subgraphs(graph):
    """Nájde všetky "claw" podgrafy."""
    claws = []
    for central_node in graph.nodes():
        neighbors = list(graph.neighbors(central_node))
        if len(neighbors) >= 3:
            for trio in combinations(neighbors, 3):
                if not any(graph.has_edge(u, v) for u, v in combinations(trio, 2)):
                    claws.append((central_node,) + trio)
    return claws


def find_cliques(graph):
    """Nájde všetky kliky veľkosti 3, 4 a 5."""
    maximal_cliques = list(nx.find_cliques(graph))
    cliques = set()
    for clique in maximal_cliques:
        if 3 <= len(clique) <= 5:
            cliques.add(tuple(sorted(clique)))
        for size in range(3, min(len(clique), 5) + 1):
            for subset in combinations(clique, size):
                cliques.add(tuple(sorted(subset)))
    return [list(clique) for clique in cliques]


def generate_gradient_colors(n=100):
    """Pomocná funkcia na generovanie farieb."""
    cmap = LinearSegmentedColormap.from_list("green_blue", ["#21B9DE", "#B9DE21"])
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def draw_stylized_nodes(ax, pos, labels, node_colors, text_size=10, padding=0.4):
    """Vykreslí štylizované uzly."""
    for node, (x, y) in pos.items():
        label = labels.get(node, str(node))
        color = node_colors.get(node, "#cccccc")
        ax.text(x, y, label, fontsize=text_size, ha='center', va='center', color='white', zorder=4,
                bbox=dict(boxstyle=f"round,pad={padding}", facecolor=color, edgecolor="#222222", linewidth=1, zorder=3))


def improved_bridge_layout(G: nx.Graph, spacing: float = 5.0, component_layout_func=nx.spring_layout,
                           **layout_kwargs) -> Dict[str, Tuple[float, float]]:
    bridges = list(nx.bridges(G))
    G_decomposed = G.copy()
    G_decomposed.remove_edges_from(bridges)
    components: List[Set[str]] = list(nx.connected_components(G_decomposed))
    component_layouts = []
    for component_nodes in components:
        subgraph = G.subgraph(component_nodes)
        pos_subgraph = component_layout_func(subgraph, **layout_kwargs)
        component_layouts.append(pos_subgraph)
    final_pos = {}
    current_offset_x = 0.0
    for pos_subgraph in component_layouts:
        if not pos_subgraph:
            continue
        min_x = min(pos[0] for pos in pos_subgraph.values())
        max_x = max(pos[0] for pos in pos_subgraph.values())
        shift_x = current_offset_x - min_x + spacing
        for node, pos in pos_subgraph.items():
            final_pos[node] = (pos[0] + shift_x, pos[1])
        current_offset_x += (max_x - min_x) + spacing
    articulation_points = set(nx.articulation_points(G))
    for ap in articulation_points:
        neighbor_positions = [final_pos[n] for n in G.neighbors(ap) if n in final_pos]
        if neighbor_positions:
            avg_x = np.mean([p[0] for p in neighbor_positions])
            avg_y = np.mean([p[1] for p in neighbor_positions])
            final_pos[ap] = (avg_x, avg_y)
    return final_pos


def create_hierarchical_layout(G: nx.Graph, grid_spacing: float = 10.0, **layout_kwargs) -> Dict[
    str, Tuple[float, float]]:
    G_filtered = G.copy()
    isolates = [node for node, degree in G.degree() if degree == 0]
    G_filtered.remove_nodes_from(isolates)
    components = list(nx.connected_components(G_filtered))
    if not components:
        return {}
    component_layouts = []
    component_bboxes = []
    for component_nodes in components:
        subgraph = G_filtered.subgraph(component_nodes)
        pos_subgraph = improved_bridge_layout(subgraph, **layout_kwargs)
        component_layouts.append(pos_subgraph)
        if pos_subgraph:
            min_x = min(pos[0] for pos in pos_subgraph.values())
            max_x = max(pos[0] for pos in pos_subgraph.values())
            min_y = min(pos[1] for pos in pos_subgraph.values())
            max_y = max(pos[1] for pos in pos_subgraph.values())
            component_bboxes.append((min_x, max_x, min_y, max_y))
        else:
            component_bboxes.append((0, 0, 0, 0))
    max_width = max((bbox[1] - bbox[0] for bbox in component_bboxes), default=0)
    max_height = max((bbox[3] - bbox[2] for bbox in component_bboxes), default=0)
    cell_width = max_width + grid_spacing
    cell_height = max_height + grid_spacing
    num_components = len(components)
    cols = int(math.ceil(math.sqrt(num_components)))
    final_pos = {}
    for i, (pos_subgraph, bbox) in enumerate(zip(component_layouts, component_bboxes)):
        if not pos_subgraph:
            continue
        row, col = i // cols, i % cols
        target_center_x = col * cell_width
        target_center_y = -row * cell_height
        original_center_x = (bbox[0] + bbox[1]) / 2
        original_center_y = (bbox[2] + bbox[3]) / 2
        shift_x = target_center_x - original_center_x
        shift_y = target_center_y - original_center_y
        for node, pos in pos_subgraph.items():
            final_pos[node] = (pos[0] + shift_x, pos[1] + shift_y)
    return final_pos


def create_main_graph_figure(G):
    """Vytvorí vizualizáciu hlavného grafu a vráti Matplotlib figure objekt."""
    if not G.nodes():
        return None

    fig, ax = plt.subplots(figsize=(20, 16))
    pos = create_hierarchical_layout(G, grid_spacing=0, seed=42, k=8)
    if not pos:
        plt.close(fig)
        return None

    G_to_draw = G.subgraph(pos.keys())
    nodes_list = list(G_to_draw.nodes())
    color_map = plt.get_cmap('viridis')
    node_colors_map = {node: color_map(i / len(nodes_list)) for i, node in enumerate(nodes_list)}

    nx.draw_networkx_edges(G_to_draw, pos, ax=ax, edge_color="gray", alpha=0.6, width=1.0)
    labels_for_nodes = {node: str(node) for node in G_to_draw.nodes()}
    draw_stylized_nodes(ax, pos, labels_for_nodes, node_colors_map, text_size=8, padding=0.4)

    edge_labels = {(u, v): f"{G_to_draw[u][v]['weight']:.2f}" for u, v in G_to_draw.edges()}
    for (u, v), label_text in edge_labels.items():
        if u in pos and v in pos:
            x_mid, y_mid = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            ax.text(x_mid, y_mid, label_text, fontsize=7, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.1"))

    plt.axis("off")
    ax.relim()
    ax.autoscale_view()

    return fig


def create_claw_subgraphs_figure(G, claws):
    """Vytvorí jednu vizualizáciu so všetkými claw podgrafmi a vráti Matplotlib figure."""
    if not claws:
        return None

    num_claws = len(claws)
    fig_cols = min(num_claws, 3)
    fig_rows = (num_claws + fig_cols - 1) // fig_cols
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 5, fig_rows * 4))
    axes = axes.flatten() if num_claws > 1 else [axes]

    for idx, (claw, ax) in enumerate(zip(claws, axes)):
        ax.set_title(f'Claw {idx + 1}')
        ax.axis('off')
        claw_subgraph = G.subgraph(claw)
        main_node = claw[0]
        neighbors = list(claw[1:])
        pos = {main_node: (0.3, 0)}
        spacing = 0.35
        for i, neighbor in enumerate(neighbors):
            y = (i - (len(neighbors) - 1) / 2) * spacing;
            pos[neighbor] = (0.7, y)
        node_colors = {main_node: "#D9534F"}
        node_colors.update({neighbor: "#0275D8" for neighbor in neighbors})
        nx.draw_networkx_edges(claw_subgraph, pos, ax=ax, edge_color="gray", alpha=0.7, width=1)
        labels = {node: str(node) for node in claw_subgraph.nodes()}
        draw_stylized_nodes(ax, pos, labels, node_colors, text_size=8, padding=0.4)
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in claw_subgraph.edges()}
        for (u, v), label in edge_labels.items(): x_mid = (pos[u][0] + pos[v][0]) / 2; y_mid = (pos[u][1] + pos[v][
            1]) / 2; ax.text(x_mid, y_mid, label, fontsize=8, ha="center", va="center",
                             bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0"))
        ax.set_xlim(0, 1)
        y_vals = [p[1] for p in pos.values()]
        ax.set_ylim(min(y_vals) - 0.2, max(y_vals) + 0.2)

    for idx in range(num_claws, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    return fig


def create_clique_figures(graph, cliques):
    """Vytvorí vizualizácie pre kliky, zoskupené podľa veľkosti, a vráti slovník figúr."""
    if not cliques:
        return {}

    cliques_by_size = {3: [], 4: [], 5: []}
    for clique in cliques:
        if len(clique) in cliques_by_size:
            cliques_by_size[len(clique)].append(clique)

    figures = {}

    for size, current_cliques in cliques_by_size.items():
        if not current_cliques:
            continue

        num_cliques = len(current_cliques)
        fig_cols = min(3, num_cliques)
        fig_rows = (num_cliques + fig_cols - 1) // fig_cols
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 6, fig_rows * 6))
        axes = axes.flatten()

        for idx, (clique, ax) in enumerate(zip(current_cliques, axes)):
            # ... (vnútorná logika kreslenia zostáva rovnaká) ...
            ax.set_title(f"Clique {idx + 1} ({size} nodes)")
            ax.axis("off")
            subgraph = graph.subgraph(clique)
            pos = nx.circular_layout(subgraph)
            pos = {node: (x * 0.5, y * 0.5) for node, (x, y) in pos.items()}
            GRADIENT_COLORS = generate_gradient_colors(5)
            node_colors = {node: GRADIENT_COLORS[i % 5] for i, node in enumerate(subgraph.nodes())}
            nx.draw_networkx_edges(subgraph, pos, ax=ax, edge_color="gray", alpha=0.7, width=1)
            labels = {node: str(node) for node in subgraph.nodes()}
            draw_stylized_nodes(ax, pos, labels, node_colors, text_size=8, padding=0.4)
            edge_labels = {(u, v): f"{graph[u][v].get('weight', 1):.2f}" for u, v in subgraph.edges()}
            for (u, v), label in edge_labels.items(): x_mid, y_mid = (pos[u][0] + pos[v][0]) / 2, (
                    pos[u][1] + pos[v][1]) / 2; dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]; ax.text(
                x_mid + dx * 0.1, y_mid + dy * 0.1, label, fontsize=8, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0"))

        for ax_to_hide in axes[num_cliques:]:
            ax_to_hide.set_visible(False)

        plt.tight_layout()
        figures[size] = fig

    return figures
