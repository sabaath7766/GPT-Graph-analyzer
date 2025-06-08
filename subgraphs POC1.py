import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, LinearSegmentedColormap
from itertools import combinations
import datetime
import os
import matplotlib.cm as cm
import math
from typing import Dict, Tuple, List, Set


def load_and_prepare_data():
    """Load and prepare the dataset."""
    try:
        df = pd.read_csv('datasets/energydata.csv')
    except FileNotFoundError:
        print("The dataset file 'abalone.csv' was not found. Please ensure the file exists in the current directory.")
        exit()

    df = df.iloc[:, 1:]
    return df


def compute_correlation_matrix(df):
    """Compute and print the correlation matrix."""
    correlation_matrix = df.corr()
    print("Correlation Matrix:\n", correlation_matrix)
    return correlation_matrix


def trim_correlation_matrix(correlation_matrix, alpha=0.1):
    """Trim the correlation matrix based on dynamic threshold. alpha between 0 & 0.3
    :rtype: pd.DataFrame
    """
    cor_max = np.max(correlation_matrix.values)
    cor_mean = np.mean(correlation_matrix.values)
    threshold = (cor_max + cor_mean) / 2 + alpha
    print(threshold)

    trimmed_matrix = np.where(np.abs(correlation_matrix) > threshold, correlation_matrix, 0)
    print("\nTrimmed Correlation Matrix:\n", trimmed_matrix)

    return pd.DataFrame(trimmed_matrix, index=correlation_matrix.index, columns=correlation_matrix.columns)


def construct_graph(trimmed_matrix, nodes):
    """Construct the graph from the trimmed matrix."""
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for i in range(len(trimmed_matrix)):
        for j in range(i + 1, len(trimmed_matrix)):
            if trimmed_matrix[i][j] != 0:
                G.add_edge(nodes[i], nodes[j], weight=trimmed_matrix[i][j])

    print("\nGraph Summary:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    return G


def find_claw_subgraphs(graph):
    """Find all claw subgraphs in the graph."""
    claws = []
    for central_node in graph.nodes():
        neighbors = list(graph.neighbors(central_node))
        if len(neighbors) >= 3:
            for trio in combinations(neighbors, 3):
                if not any(graph.has_edge(u, v) for u, v in combinations(trio, 2)):
                    claws.append((central_node,) + trio)
    return claws


def find_cliques(graph):
    """
    Finds all cliques of sizes 3, 4, and 5 in the given graph.
    Includes cliques that are subsets of larger ones.

    Parameters:
        graph (networkx.Graph): The graph to analyze.

    Returns:
        list: A list of cliques, each represented as a list of nodes.
    """
    # Find all maximal cliques
    maximal_cliques = list(nx.find_cliques(graph))

    # Collect all cliques of sizes 3-5
    cliques = set()  # Use a set to avoid duplicates
    for clique in maximal_cliques:
        # Include the entire maximal clique if its size is 3-5
        if 3 <= len(clique) <= 5:
            cliques.add(tuple(sorted(clique)))
        # Generate all subsets of sizes 3-5
        for size in range(3, min(len(clique), 5) + 1):
            for subset in combinations(clique, size):
                cliques.add(tuple(sorted(subset)))

    return [list(clique) for clique in cliques]


def generate_gradient_colors(n=100):
    """Generate a list of `n` colors transitioning from green to blue."""
    cmap = LinearSegmentedColormap.from_list("green_blue", ["#21B9DE", "#B9DE21"])
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def draw_stylized_nodes(ax, pos, labels, node_colors, text_size=10, padding=0.4):
    """
    Draws nodes by creating text labels with a styled bounding box.
    This is a robust way to create nodes that fit the text content.

    Args:
        ax: The Matplotlib axes object.
        pos: Dictionary of node positions.
        labels: Dictionary of node labels.
        node_colors: Dictionary of node colors.
        text_size: The font size for the node labels.
        padding: The padding for the bounding box (in points). Adjust for desired fit.
    """
    for node, (x, y) in pos.items():
        label = labels.get(node, str(node))
        color = node_colors.get(node, "#cccccc")

        ax.text(x, y, label,
                fontsize=text_size,
                ha='center',
                va='center',
                color='white',
                zorder=4,  # Ensure text is on top of the box
                bbox=dict(
                    boxstyle=f"round,pad={padding}",  # The style of the box
                    facecolor=color,
                    edgecolor="#222222",
                    linewidth=1,
                    zorder=3  # Ensure box is behind the text
                ))


def bridge_layout(G, spacing=40, subgraph_scale=2.5, repulsion_strength=200.0, repulsion_radius=100.0, iterations=1000):
    """
    DEPRECATED
    Calculates a layout for G with bridges removed and subgraphs spaced out,
    using circular layout and repulsion forces.
    """
    bridges = list(nx.bridges(G))
    articulation_points = set(nx.articulation_points(G))

    G_copy = G.copy()
    G_copy.remove_edges_from(bridges)
    subgraphs = list(nx.connected_components(G_copy))

    pos = {}
    offset_x = 0

    for component in subgraphs:
        subgraph = G.subgraph(component).copy()

        for u, v, d in subgraph.edges(data=True):
            if 'weight' in d:
                d['weight'] = abs(d['weight'])

        num_nodes = len(subgraph)
        radius = subgraph_scale * 10
        center_x = offset_x + radius
        center_y = 0

        sub_layout = {}
        angle_step = 2 * math.pi / max(num_nodes, 1)

        for i, node in enumerate(subgraph.nodes()):
            angle = i * angle_step
            sub_layout[node] = (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle)
            )

        # Scale
        for node, (x, y) in sub_layout.items():
            sub_layout[node] = (x * subgraph_scale, y * subgraph_scale)

        # Shift to the right
        min_x_scaled = min(x for x, y in sub_layout.values())
        max_x_scaled = max(x for x, y in sub_layout.values())
        shift_x = offset_x - min_x_scaled + spacing
        for node, (x, y) in sub_layout.items():
            sub_layout[node] = (x + shift_x, y)

        offset_x = max_x_scaled + shift_x
        pos.update(sub_layout)

    # Place articulation points near their neighbors
    for ap in articulation_points:
        if ap in G.nodes and ap in pos:
            neighbor_positions = [pos[n] for n in G.neighbors(ap) if n in pos]
            if neighbor_positions:
                avg_x = np.mean([p[0] for p in neighbor_positions])
                avg_y = np.mean([p[1] for p in neighbor_positions])
                pos[ap] = (avg_x, avg_y)

    # Apply repulsion
    for _ in range(iterations):
        new_pos = pos.copy()
        for node1, pos1 in pos.items():
            for node2, pos2 in pos.items():
                if node1 != node2:
                    dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    if repulsion_radius > dist > 1e-6:
                        direction = np.array(pos1) - np.array(pos2)
                        direction = direction / np.linalg.norm(direction)
                        force = repulsion_strength / dist
                        shift = direction * force * 0.01
                        new_pos[node1] = tuple(np.array(new_pos[node1]) + shift)
                        new_pos[node2] = tuple(np.array(new_pos[node2]) - shift)
        pos = new_pos

    return pos


def visualize_main_graph_old(G, output_dir, node_size=800, font_size=10, edge_width=1.5, edge_alpha=0.6):
    """
    DEPRECATED
    Visualizes the main graph using the provided layout positions.
    """
    G_filtered = G.edge_subgraph(G.edges()).copy()

    plt.figure(figsize=(16, 10))

    pos = bridge_layout(G_filtered)

    # Draw nodes
    # nx.draw_networkx_nodes(G_filtered, pos, node_size=node_size, node_color='skyblue')

    # Assign a unique color to each subgraph
    num_subgraphs = len(list(nx.connected_components(G_filtered)))
    color_map = cm.get_cmap("tab10", num_subgraphs)

    node_colors = {}
    for i, component in enumerate(nx.connected_components(G_filtered)):
        color = to_hex(color_map(i))
        for node in component:
            node_colors[node] = color

    # Draw nodes with assigned colors
    node_colors_list = [node_colors[node] for node in G_filtered.nodes()]
    nx.draw_networkx_nodes(G_filtered, pos, node_size=node_size, node_color=node_colors_list)

    # Draw edges
    nx.draw_networkx_edges(G_filtered, pos, width=edge_width, alpha=edge_alpha, edge_color='gray')

    # Draw node labels
    nx.draw_networkx_labels(G_filtered, pos, font_size=font_size, font_color='black')

    # Draw edge labels
    edge_labels = {(u, v): f"{G_filtered[u][v]['weight']:.2f}" for u, v in G_filtered.edges()}
    nx.draw_networkx_edge_labels(G_filtered, pos, edge_labels=edge_labels, font_size=font_size - 2)

    plt.axis("off")
    filename = os.path.join(output_dir, "main_graph.svg")
    plt.savefig(filename, format="svg")
    print(f"Main graph visualization saved as '{filename}'.")


def improved_bridge_layout(
        G: nx.Graph,
        spacing: float = 5.0,
        component_layout_func=nx.spring_layout,
        **layout_kwargs
) -> Dict[str, Tuple[float, float]]:
    """
    Calculates a visually separated layout for a single connected graph component with bridges.
    """
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


def create_hierarchical_layout(
        G: nx.Graph,
        grid_spacing: float = 10.0,
        **layout_kwargs
) -> Dict[str, Tuple[float, float]]:
    """
    Creates a hierarchical layout by first filtering isolates, then applying
    the 'improved_bridge_layout' to each component and arranging them in a grid.

    Args:
        G: The input NetworkX graph.
        grid_spacing: The space to maintain between components in the final grid.
        **layout_kwargs: Keyword arguments to pass to the internal layout functions
                         (e.g., seed, iterations, k).

    Returns:
        A dictionary of node positions.
    """
    # 1. PRE-PROCESSING: Filter out all isolated nodes (nodes with no edges)
    G_filtered = G.copy()
    isolates = [node for node, degree in G.degree() if degree == 0]
    G_filtered.remove_nodes_from(isolates)

    # 2. Find all remaining connected components
    components = list(nx.connected_components(G_filtered))
    if not components:
        return {}

    # 3. Calculate the layout for each component using the bridge_layout
    component_layouts = []
    component_bboxes = []
    for component_nodes in components:
        subgraph = G_filtered.subgraph(component_nodes)

        # Apply the bridge layout to each component
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

    # 4. Arrange the fully rendered components in a grid
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


def visualize_main_graph(G, output_dir):
    """Visualize and save the graph using the hierarchical layout."""

    if not G.nodes():
        print("Graph is empty. Skipping visualization.")
        return

    plt.figure(figsize=(20, 16))
    ax = plt.gca()

    # seed for reproducibility of the internal spring layouts
    pos = create_hierarchical_layout(G, grid_spacing=0, seed=42, k=8)

    if not pos:
        print("Layout is empty after filtering isolates. Nothing to visualize.")
        plt.close()
        return

    G_to_draw = G.subgraph(pos.keys())

    # Node coloring
    nodes_list = list(G_to_draw.nodes())
    num_total_nodes = len(nodes_list)
    color_map = plt.get_cmap('viridis')
    node_colors_map = {node: color_map(i / num_total_nodes) for i, node in enumerate(nodes_list)}

    # Draw Edges
    nx.draw_networkx_edges(G_to_draw, pos, ax=ax, edge_color="gray", alpha=0.6, width=1.0)

    # Draw Nodes
    labels_for_nodes = {node: str(node) for node in G_to_draw.nodes()}
    draw_stylized_nodes(ax, pos, labels_for_nodes, node_colors_map, text_size=8, padding=0.4)

    # Draw Edge Labels
    edge_labels = {(u, v): f"{G_to_draw[u][v]['weight']:.2f}" for u, v in G_to_draw.edges()}
    for (u, v), label_text in edge_labels.items():
        if u in pos and v in pos:
            x_mid = (pos[u][0] + pos[v][0]) / 2
            y_mid = (pos[u][1] + pos[v][1]) / 2
            ax.text(x_mid, y_mid, label_text, fontsize=7, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.1"))

    plt.axis("off")
    ax.relim()
    ax.autoscale_view()

    filename = os.path.join(output_dir, "main_graph_hierarchical_layout.svg")
    plt.savefig(filename, format="svg", bbox_inches='tight')
    print(f"Hierarchical graph visualization saved as '{filename}'.")
    plt.close()


def visualize_claw_subgraphs(G, claws, output_dir):
    """
    Visualizes claw sub-graphs in a left-right layout:
      - Main node on the left (fixed x), neighbors on the right (fixed x).
      - Nodes drawn as truly tight, rounded rectangles.
      - Edge labels with a white background and no border.
    """
    if not claws:
        print("\nNo claw subgraphs found to visualize.")
        return

    num_claws = len(claws)
    fig_cols = min(num_claws, 3)
    fig_rows = (num_claws + fig_cols - 1) // fig_cols

    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 5, fig_rows * 4))
    axes = axes.flatten() if num_claws > 1 else [axes]

    # Fixed x-positions to prevent clipping
    left_x = 0.3
    right_x = 0.7

    for idx, (claw, ax) in enumerate(zip(claws, axes)):
        ax.set_title(f'Claw {idx + 1}')
        ax.axis('off')
        claw_subgraph = G.subgraph(claw)

        # Layout: main node at left_x, neighbors at right_x
        main_node = claw[0]
        neighbors = list(claw[1:])
        pos = {main_node: (left_x, 0)}
        spacing = 0.35  # vertical spacing for neighbors
        for i, neighbor in enumerate(neighbors):
            y = (i - (len(neighbors) - 1) / 2) * spacing
            pos[neighbor] = (right_x, y)

        # Colors
        node_colors = {main_node: "#D9534F"}
        node_colors.update({neighbor: "#0275D8" for neighbor in neighbors})

        # Draw edges first
        nx.draw_networkx_edges(claw_subgraph, pos, ax=ax,
                               edge_color="gray", alpha=0.7, width=1)

        # Draw nodes as tight, rounded rectangles
        labels = {node: str(node) for node in claw_subgraph.nodes()}
        draw_stylized_nodes(ax, pos, labels, node_colors, text_size=8, padding=0.4)

        # Edge labels: white background, no border
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in claw_subgraph.edges()}
        for (u, v), label in edge_labels.items():
            x_mid = (pos[u][0] + pos[v][0]) / 2
            y_mid = (pos[u][1] + pos[v][1]) / 2
            ax.text(x_mid, y_mid, label, fontsize=8, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0"))

        # Prevent clipping
        ax.set_xlim(0, 1)
        y_vals = [p[1] for p in pos.values()]
        ax.set_ylim(min(y_vals) - 0.2, max(y_vals) + 0.2)

    # Remove unused subplots
    for idx in range(num_claws, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    filename = os.path.join(output_dir, "claw_subgraphs.svg")
    plt.savefig(filename, format="svg")
    print(f"\nClaw subgraphs (left-right layout) saved as '{filename}'.")
    # print("Claw positions:", pos)


def visualize_cliques(graph, cliques, output_dir, scale_factor=0.5):
    """Visualizes cliques with a predefined gradient color order and fixes empty plots."""
    if not cliques:
        print("No cliques found to visualize.")
        return

    # Group cliques by size
    cliques_by_size = {3: [], 4: [], 5: []}
    for clique in cliques:
        if len(clique) in cliques_by_size:
            cliques_by_size[len(clique)].append(clique)

    for size, current_cliques in cliques_by_size.items():
        if not current_cliques:
            continue

        num_cliques = len(current_cliques)
        fig_cols = min(3, num_cliques)
        fig_rows = (num_cliques + fig_cols - 1) // fig_cols

        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 6, fig_rows * 6))
        axes = axes.flatten()

        for idx, (clique, ax) in enumerate(zip(current_cliques, axes)):
            ax.set_title(f"Clique {idx + 1} ({size} nodes)")
            ax.axis("off")

            subgraph = graph.subgraph(clique)

            # Layout: Scale positions
            pos = nx.circular_layout(subgraph)
            pos = {node: (x * scale_factor, y * scale_factor) for node, (x, y) in pos.items()}

            GRADIENT_COLORS = generate_gradient_colors(5)
            node_colors = {node: GRADIENT_COLORS[i % 5] for i, node in enumerate(subgraph.nodes())}

            # Draw edges first
            nx.draw_networkx_edges(subgraph, pos, ax=ax, edge_color="gray", alpha=0.7, width=1)

            # Draw tight, rounded nodes
            labels = {node: str(node) for node in subgraph.nodes()}
            draw_stylized_nodes(ax, pos, labels, node_colors, text_size=8, padding=0.4)

            # Edge labels
            edge_labels = {(u, v): f"{graph[u][v].get('weight', 1):.2f}" for u, v in subgraph.edges()}
            for (u, v), label in edge_labels.items():
                x_mid, y_mid = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
                dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]

                # Move label along the edge direction
                factor = 0.1  # Adjust this to move more/less along the edge
                new_x = x_mid + dx * factor
                new_y = y_mid + dy * factor

                ax.text(new_x, new_y, label, fontsize=8, ha="center", va="center",
                        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0"))

        # **Fix empty plots** by hiding unused axes
        for ax in axes[num_cliques:]:
            ax.set_visible(False)

        plt.tight_layout()
        filename = os.path.join(output_dir, f"cliques_size_{size}.svg")
        plt.savefig(filename, format="svg")
        print(f"Clique visualizations for size {size} saved as '{filename}'.")


def main():
    """Main function to execute all steps."""
    # Create a timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load and prepare data
    df = load_and_prepare_data()

    # Step 2: Compute correlation matrix
    correlation_matrix = compute_correlation_matrix(df)

    # Step 3: Trim correlation matrix
    trimmed_matrix = trim_correlation_matrix(correlation_matrix, alpha=0.1)

    # Step 4: Construct graph
    nodes = correlation_matrix.columns.tolist()
    G = construct_graph(trimmed_matrix, nodes)

    # Step 5: Find claw subgraphs
    claws = find_claw_subgraphs(G)
    print("\nClaw Subgraphs Found:", claws if claws else "None")

    # Step 6: Find all cliques of sizes 3-5
    cliques = find_cliques(G)

    # Step 7: Visualizations
    visualize_main_graph(G, output_dir)

    if cliques:
        visualize_cliques(G, cliques, output_dir)
    else:
        print("No cliques found in the graph.")

    if claws:
        visualize_claw_subgraphs(G, claws, output_dir)


if __name__ == "__main__":
    main()
