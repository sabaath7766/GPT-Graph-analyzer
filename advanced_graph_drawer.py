import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
import datetime
from collections import deque
from itertools import combinations


# --- POMOCNÉ FUNKCIE PRE GEOMETRIU A METRIKY ---

def do_lines_intersect(p1, p2, p3, p4):
    """Skontroluje, či sa úsečky (p1, p2) a (p3, p4) pretínajú."""

    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0  # Kollinearne
        return 1 if val > 0 else 2  # V smere alebo proti smeru hodinových ručičiek

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True

    # Špeciálne prípady pre kollineárne body
    if o1 == 0 and on_segment(p1, p3, p2): return True
    if o2 == 0 and on_segment(p1, p4, p2): return True
    if o3 == 0 and on_segment(p3, p1, p4): return True
    if o4 == 0 and on_segment(p3, p2, p4): return True

    return False


def calculate_crossing_count(graph, pos):
    """Vypočíta počet prekrížení hrán v grafe."""
    crossings = 0
    # Prejdeme všetky unikátne páry hrán
    for edge1, edge2 in combinations(graph.edges(), 2):
        u1, v1 = edge1
        u2, v2 = edge2
        # Hrany sa nemôžu pretínať, ak zdieľajú uzol
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue

        p1, p2 = pos[u1], pos[v1]
        p3, p4 = pos[u2], pos[v2]

        if do_lines_intersect(p1, p2, p3, p4):
            crossings += 1
    return crossings


def calculate_min_angle(graph, pos):
    """Vypočíta minimálny uhol medzi susednými hranami v celom grafe."""
    min_angle = 360
    for node in graph.nodes():
        if graph.degree(node) < 2:
            continue

        neighbors = list(graph.neighbors(node))
        p0 = np.array(pos[node])

        # Prejdeme všetky páry susedov pre daný uzol
        for n1, n2 in combinations(neighbors, 2):
            p1 = np.array(pos[n1])
            p2 = np.array(pos[n2])

            vec1 = p1 - p0
            vec2 = p2 - p0

            # Vzorec pre uhol pomocou dot product
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

            # Zamedzenie delenia nulou
            if norm_product == 0: continue

            cosine_angle = dot_product / norm_product
            # Orezanie hodnoty kvôli numerickým nepresnostiam
            angle = np.rad2deg(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

            if angle < min_angle:
                min_angle = angle

    return min_angle if min_angle != 360 else 0


# --- KROK 1: Príprava dát (bez zmeny) ---
def load_and_prepare_data(filepath='datasets/energydata.csv'):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Súbor '{filepath}' nebol nájdený.")
        return None
    df = df.drop(columns=['date', 'lights'])
    return df


def get_correlation_graph(df, alpha=0.1):
    correlation_matrix = df.corr()
    threshold = (np.max(correlation_matrix.values[~np.eye(correlation_matrix.shape[0], dtype=bool)]) +
                 np.mean(correlation_matrix.values[~np.eye(correlation_matrix.shape[0], dtype=bool)])) / 2 + alpha
    print(f"Dynamický prah pre korelácie: {threshold:.4f}")
    trimmed_matrix = np.where(np.abs(correlation_matrix) > threshold, 1, 0)
    np.fill_diagonal(trimmed_matrix, 0)
    G = nx.from_numpy_array(trimmed_matrix)
    mapping = {i: col for i, col in enumerate(correlation_matrix.columns)}
    nx.relabel_nodes(G, mapping, copy=False)
    return G


def preprocess_graph(G):
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"Po očistení graf obsahuje {G.number_of_nodes()} uzlov a {G.number_of_edges()} hrán.")
    return G


# --- KROK 2: Vylepšené funkcie pre vizualizáciu ---

def point_to_line_distance(point, line_start, line_end):
    """Calculate the distance from a point to a line segment."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Calculate the distance from point to line segment
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D

    if len_sq == 0:  # Line segment is actually a point
        return math.sqrt(A * A + B * B)

    param = dot / len_sq

    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = px - xx
    dy = py - yy
    return math.sqrt(dx * dx + dy * dy)


def does_edge_intersect_node(edge_start, edge_end, node_pos, node_radius=0.25):
    """Check if a straight edge would intersect with a node (within radius)."""
    return point_to_line_distance(node_pos, edge_start, edge_end) < node_radius


def does_curved_edge_intersect_nodes(p1, p2, rad, graph, pos, u, v, node_radius=0.25):
    """Check if a curved edge intersects any nodes it shouldn't touch."""
    curved_points = get_curved_path_points(p1, p2, rad)

    for node in graph.nodes():
        if node == u or node == v:  # Skip endpoints
            continue

        node_pos = pos[node]

        # Check if any point along the curved path is too close to this node
        for point in curved_points:
            dist = math.sqrt((point[0] - node_pos[0]) ** 2 + (point[1] - node_pos[1]) ** 2)
            if dist < node_radius:
                return True

    return False


def find_best_curve_for_edge(graph, pos, u, v, node_radius=0.25):
    """
    Find the best curve parameters for an edge to avoid intersecting other nodes.
    Returns the connectionstyle string to use.
    """
    p1 = pos[u]
    p2 = pos[v]

    # Check if straight line intersects any other nodes
    intersecting_nodes = []
    for node in graph.nodes():
        if node != u and node != v:  # Skip the endpoints
            if does_edge_intersect_node(p1, p2, pos[node], node_radius):
                intersecting_nodes.append(node)

    # If no intersections, draw straight line
    if not intersecting_nodes:
        return "arc3,rad=0"

    # INCREASED curve options for more spread out edges
    curve_options = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5,  # Positive curves
                     -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -1.0, -1.2, -1.5,  # Negative curves
                     2.0, -2.0, 2.5, -2.5, 3.0, -3.0]  # Extreme curves

    print(f"  Edge {u}-{v}: Testing curves to avoid {len(intersecting_nodes)} nodes...")

    for rad in curve_options:
        # Use the improved collision detection
        if not does_curved_edge_intersect_nodes(p1, p2, rad, graph, pos, u, v, node_radius):
            print(f"    Found solution: rad={rad}")
            return f"arc3,rad={rad}"

    # If all curves still intersect, use a very strong curve as fallback
    print(f"    Using fallback curve: rad=1.5")
    return "arc3,rad=1.5"


def get_curved_path_points(p1, p2, rad, num_points=20):
    """Generate points along a curved path to simulate FancyArrowPatch curve."""
    if rad == 0:
        # Straight line
        t_values = np.linspace(0, 1, num_points)
        return [(p1[0] * (1 - t) + p2[0] * t, p1[1] * (1 - t) + p2[1] * t) for t in t_values]

    # Calculate curve points more accurately
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    line_length = math.sqrt(dx * dx + dy * dy)

    if line_length == 0:
        return [p1]

    # Perpendicular vector (normalized)
    perp_x = -dy / line_length
    perp_y = dx / line_length

    # INCREASED maximum curve offset for more pronounced curves
    max_offset = rad * line_length * 0.8  # Increased from 0.5 to 0.8

    t_values = np.linspace(0, 1, num_points)
    curved_points = []

    for t in t_values:
        # Linear interpolation along straight line
        straight_x = p1[0] * (1 - t) + p2[0] * t
        straight_y = p1[1] * (1 - t) + p2[1] * t

        # Curve strength (parabolic - strongest at middle)
        curve_strength = 4 * t * (1 - t)  # 0 at ends, 1 at middle
        offset = max_offset * curve_strength

        # Add perpendicular offset
        curved_x = straight_x + offset * perp_x
        curved_y = straight_y + offset * perp_y

        curved_points.append((curved_x, curved_y))

    return curved_points


def draw_graph_with_metrics(graph, pos, title, filepath):
    """
    Improved function with intelligent edge routing and dynamic canvas sizing:
    - Calculates and displays metrics.
    - Draws edges with smart curve detection to avoid nodes.
    - Dynamically sizes canvas based on layout bounds.
    """
    print(f"Počítam metriky pre '{title}'...")
    crossings = calculate_crossing_count(graph, pos)
    min_angle = calculate_min_angle(graph, pos)
    full_title = (f"{title}\n"
                  f"Metrics: Edge Crossings = {crossings} | Min Angle = {min_angle:.2f}°")

    # Calculate layout bounds for dynamic canvas sizing
    if not pos:
        fig, ax = plt.subplots(figsize=(18, 16))
    else:
        x_coords = [x for x, y in pos.values()]
        y_coords = [y for x, y in pos.values()]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        x_range = x_max - x_min
        y_range = y_max - y_min

        # Add padding
        x_range_padded = max(x_range * 1.3, 10)  # At least 10 units wide
        y_range_padded = max(y_range * 1.3, 8)  # At least 8 units tall

        # Calculate figure size (scale factor to convert to inches)
        scale_factor = 1.2
        fig_width = min(max(x_range_padded * scale_factor, 12), 30)  # Between 12-30 inches
        fig_height = min(max(y_range_padded * scale_factor, 10), 25)  # Between 10-25 inches

        print(f"Layout bounds: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}]")
        print(f"Dynamic canvas size: {fig_width:.1f} × {fig_height:.1f} inches")

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Smart edge drawing with collision avoidance
    print("Drawing edges with smart collision avoidance...")
    for u, v in graph.edges():
        p1 = pos[u]
        p2 = pos[v]

        # Find the best curve for this edge
        connection_style = find_best_curve_for_edge(graph, pos, u, v)

        arc = patches.FancyArrowPatch(
            p1, p2,
            connectionstyle=connection_style,
            color="#666666",
            linewidth=1.2,
            alpha=0.7
        )
        ax.add_patch(arc)

    # Draw nodes and labels
    node_colors = [graph.degree(node) for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap=plt.cm.plasma,
                           node_size=2000, alpha=0.9, ax=ax, edgecolors='black')
    nx.draw_networkx_labels(graph, pos, font_size=9, font_weight='bold',
                            font_color='white', ax=ax)

    ax.set_title(full_title, fontsize=22, pad=20)
    ax.margins(0.15)
    plt.axis('off')

    try:
        plt.savefig(filepath, format='png', bbox_inches='tight', dpi=120)
        print(f"✅ Vizualizácia uložená do: {filepath}")
    except Exception as e:
        print(f"❌ Nepodarilo sa uložiť obrázok: {e}")
    finally:
        plt.close()


def create_improved_radial_layout(graph, filepath):
    title = "1. Improved Degree-Based Radial Layout"
    degrees = dict(graph.degree())
    nodes_sorted = sorted(graph.nodes(), key=lambda n: degrees[n], reverse=True)
    num = len(nodes_sorted)
    s1 = nodes_sorted[:max(1, int(num * 0.05))]
    s2 = nodes_sorted[max(1, int(num * 0.05)):max(2, int(num * 0.3))]
    s3 = nodes_sorted[max(2, int(num * 0.3)):]
    shells = [s for s in [s1, s2, s3] if s]
    pos = nx.shell_layout(graph, nlist=shells, scale=2)
    draw_graph_with_metrics(graph, pos, title, filepath)


def create_sorted_grid_layout(graph, filepath):
    title = "2. Sorted Grid Layout"
    nodes_sorted = sorted(graph.nodes(), key=lambda n: graph.degree(n), reverse=True)
    side = int(math.ceil(math.sqrt(graph.number_of_nodes())))
    pos = {node: ((i % side), side - (i // side) - 1) for i, node in enumerate(nodes_sorted)}
    draw_graph_with_metrics(graph, pos, title, filepath)


def create_component_internal_layout(component, spread_factor):
    """Create internal node layout for a single component."""
    component_pos = {}

    if component.number_of_nodes() == 1:
        node = list(component.nodes())[0]
        component_pos[node] = (0, 0)
        return component_pos

    # Find diameter (longest shortest path) in this component
    try:
        ecc = nx.eccentricity(component)
        end_node1 = max(ecc, key=ecc.get)
        distances = nx.shortest_path_length(component, source=end_node1)
        end_node2 = max(distances, key=distances.get)
        backbone = nx.shortest_path(component, source=end_node1, target=end_node2)
    except:
        backbone = list(component.nodes())[:min(3, component.number_of_nodes())]

    # Position backbone vertically with size-based spreading
    local_height = len(backbone) * 1.0 * spread_factor
    main_y_coords = np.linspace(local_height / 2, -local_height / 2, len(backbone))
    for i, node in enumerate(backbone):
        component_pos[node] = (0, main_y_coords[i])

    # Position remaining nodes using BFS from backbone
    processed = set(backbone)
    queue = deque(backbone)
    base_offset = 1.2 * spread_factor
    side_offsets = {node: {'left': base_offset, 'right': base_offset} for node in backbone}

    while queue and len(processed) < component.number_of_nodes():
        parent = queue.popleft()
        if parent not in component_pos:
            continue

        px, py = component_pos[parent]
        neighbors = sorted([n for n in component.neighbors(parent) if n not in processed])

        for i, node in enumerate(neighbors):
            side = 'right' if i % 2 == 0 else 'left'
            offset_x = side_offsets.get(parent, {'left': base_offset, 'right': base_offset})[side]
            sign = 1 if side == 'right' else -1
            component_pos[node] = (px + sign * offset_x, py)

            if parent in side_offsets:
                side_offsets[parent][side] += 1.0 * spread_factor

            processed.add(node)
            queue.append(node)

    # Handle any remaining unprocessed nodes
    remaining_in_component = set(component.nodes()) - processed
    if remaining_in_component:
        max_x_in_comp = max(x for x, y in component_pos.values()) if component_pos else 0
        for i, node in enumerate(remaining_in_component):
            component_pos[node] = (max_x_in_comp + 2, i - len(remaining_in_component) / 2)

    return component_pos


def physics_based_component_positioning(component_info):
    """Use physics simulation to position components without overlap."""
    print("  Inicializujem pozície komponentov...")

    # Initial positioning - place larger components first
    for i, info in enumerate(component_info):
        if i == 0:
            # Largest component at center
            info['center_x'] = 0
            info['center_y'] = 0
        else:
            # Place other components in a rough circle around center
            angle = 2 * math.pi * i / len(component_info)
            initial_radius = 8
            info['center_x'] = initial_radius * math.cos(angle)
            info['center_y'] = initial_radius * math.sin(angle)

        # Store original centers for translation
        info['original_center_x'] = info['center_x']
        info['original_center_y'] = info['center_y']

    # Physics simulation parameters
    max_iterations = 200
    dt = 0.1
    damping = 0.95
    repulsion_strength = 50

    for iteration in range(max_iterations):
        total_force = 0

        for i, info in enumerate(component_info):
            force_x = 0
            force_y = 0

            # Calculate repulsion from other components
            for j, other_info in enumerate(component_info):
                if i == j:
                    continue

                dx = info['center_x'] - other_info['center_x']
                dy = info['center_y'] - other_info['center_y']
                distance = math.sqrt(dx * dx + dy * dy)
                min_distance = info['radius'] + other_info['radius'] + 0.5

                if distance < min_distance and distance > 0:
                    # Normalize direction
                    nx_dir = dx / distance
                    ny_dir = dy / distance

                    # Calculate overlap and force
                    overlap = min_distance - distance
                    force_magnitude = repulsion_strength * overlap

                    force_x += force_magnitude * nx_dir
                    force_y += force_magnitude * ny_dir

            # Update velocity and position
            info['velocity_x'] = (info['velocity_x'] + force_x * dt) * damping
            info['velocity_y'] = (info['velocity_y'] + force_y * dt) * damping

            info['center_x'] += info['velocity_x'] * dt
            info['center_y'] += info['velocity_y'] * dt

            total_force += abs(force_x) + abs(force_y)

        # Check for convergence
        if total_force < 1.0:
            print(f"  Physics simulation converged po {iteration + 1} iteráciách")
            break
    else:
        print(f"  Physics simulation skončila po {max_iterations} iteráciách")

    # Final positions
    for i, info in enumerate(component_info):
        print(f"  Finálna pozícia komponentu {i + 1}: ({info['center_x']:.1f}, {info['center_y']:.1f})")

    return component_info


def create_orthogonal_layout_from_scratch(graph, filepath):
    """PHYSICS-BASED VERSION: Uses force collision to arrange components optimally."""
    title = "3. Orthogonal Layout - Physics-Based Component Arrangement"
    pos = {}

    if nx.is_connected(graph):
        components = [graph]
    else:
        components = [graph.subgraph(cc).copy() for cc in nx.connected_components(graph)]
        # Sort components by size (largest first)
        components = sorted(components, key=lambda c: c.number_of_nodes(), reverse=True)
        print(f"Graf má {len(components)} disconnected komponentov.")

    print("Komponenty (zoradené podľa veľkosti):")
    for i, comp in enumerate(components):
        print(f"  {i + 1}. {comp.number_of_nodes()} uzlov")

    # Phase 1: Create internal layouts for each component
    component_layouts = []
    component_info = []

    for comp_idx, component in enumerate(components):
        print(f"Vytváram layout pre komponent {comp_idx + 1}...")

        # Calculate desired spread based on component size
        component_sizes = [c.number_of_nodes() for c in components]
        max_nodes = max(component_sizes) if component_sizes else 1
        size_ratio = component.number_of_nodes() / max_nodes
        spread_factor = 0.8 + (size_ratio * 1.5)  # Range from 0.8 to 2.3

        component_pos = create_component_internal_layout(component, spread_factor)

        # Calculate bounding circle for this component
        if component_pos:
            x_coords = [x for x, y in component_pos.values()]
            y_coords = [y for x, y in component_pos.values()]

            # Center of component
            center_x = (min(x_coords) + max(x_coords)) / 2
            center_y = (min(y_coords) + max(y_coords)) / 2

            # Radius (distance to furthest node + padding)
            max_dist = 0
            for x, y in component_pos.values():
                dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                max_dist = max(max_dist, dist)

            radius = max_dist + 1.5  # Extra padding
        else:
            center_x, center_y, radius = 0, 0, 2

        component_layouts.append(component_pos)
        component_info.append({
            'component': component,
            'center_x': center_x,
            'center_y': center_y,
            'radius': radius,
            'velocity_x': 0,
            'velocity_y': 0,
            'size': component.number_of_nodes()
        })

        print(f"  Komponent {comp_idx + 1}: center=({center_x:.1f}, {center_y:.1f}), radius={radius:.1f}")

    # Phase 2: Physics-based positioning to avoid overlaps
    print("Spúšťam physics simulation pre rozloženie komponentov...")
    component_info = physics_based_component_positioning(component_info)

    # Phase 3: Combine all layouts with final positions
    for comp_idx, (component_pos, info) in enumerate(zip(component_layouts, component_info)):
        final_center_x = info['center_x']
        final_center_y = info['center_y']
        original_center_x = info.get('original_center_x', 0)
        original_center_y = info.get('original_center_y', 0)

        # Translate each node position
        for node, (local_x, local_y) in component_pos.items():
            global_x = final_center_x + (local_x - original_center_x)
            global_y = final_center_y + (local_y - original_center_y)
            pos[node] = (global_x, global_y)

        # Handle single isolated nodes
        if component.number_of_nodes() == 1:
            node = list(component.nodes())[0]
            pos[node] = (center_x, center_y)
            continue

        # For components with multiple nodes, create local layout
        component_pos = {}

        # Find diameter (longest shortest path) in this component
        try:
            ecc = nx.eccentricity(component)
            end_node1 = max(ecc, key=ecc.get)
            distances = nx.shortest_path_length(component, source=end_node1)
            end_node2 = max(distances, key=distances.get)
            backbone = nx.shortest_path(component, source=end_node1, target=end_node2)
        except:
            # Fallback if eccentricity fails
            backbone = list(component.nodes())[:min(3, component.number_of_nodes())]

        # Position backbone vertically in local coordinates with size-based spreading
        local_height = len(backbone) * 1.0 * spread_factor  # More spread for bigger components
        main_y_coords = np.linspace(local_height / 2, -local_height / 2, len(backbone))
        for i, node in enumerate(backbone):
            component_pos[node] = (0, main_y_coords[i])  # Local coordinates

        # Position remaining nodes using BFS from backbone
        processed = set(backbone)
        queue = deque(backbone)
        # Scale side offsets based on spread factor
        base_offset = 1.2 * spread_factor
        side_offsets = {node: {'left': base_offset, 'right': base_offset} for node in backbone}

        while queue and len(processed) < component.number_of_nodes():
            parent = queue.popleft()
            if parent not in component_pos:
                continue

            px, py = component_pos[parent]
            neighbors = sorted([n for n in component.neighbors(parent) if n not in processed])

            for i, node in enumerate(neighbors):
                side = 'right' if i % 2 == 0 else 'left'
                offset_x = side_offsets.get(parent, {'left': 1.5, 'right': 1.5})[side]
                sign = 1 if side == 'right' else -1
                component_pos[node] = (px + sign * offset_x, py)

                if parent in side_offsets:
                    side_offsets[parent][side] += 1.0 * spread_factor

                processed.add(node)
                queue.append(node)

        # Handle any remaining unprocessed nodes in this component
        remaining_in_component = set(component.nodes()) - processed
        if remaining_in_component:
            print(f"  Pridávam {len(remaining_in_component)} zostávajúcich uzlov...")
            max_x_in_comp = max(x for x, y in component_pos.values()) if component_pos else 0
            for i, node in enumerate(remaining_in_component):
                component_pos[node] = (max_x_in_comp + 2, i - len(remaining_in_component) / 2)

        # Convert local coordinates to global coordinates
        for node in component.nodes():
            if node in component_pos:
                local_x, local_y = component_pos[node]
                pos[node] = (center_x + local_x, center_y + local_y)

    print(f"Celkovo umiestnených {len(pos)} z {graph.number_of_nodes()} uzlov.")

    # Verify all nodes are positioned
    missing_nodes = set(graph.nodes()) - set(pos.keys())
    if missing_nodes:
        print(f"WARNING: Chýbajú pozície pre uzly: {missing_nodes}")
        # Add missing nodes at the end
        max_x = max(x for x, y in pos.values()) if pos else 0
        for i, node in enumerate(missing_nodes):
            pos[node] = (max_x + 3, i)

    draw_graph_with_metrics(graph, pos, title, filepath)


# --- KROK 3: Hlavná funkcia ---
def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", "advanced_layouts", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Výstupy sa budú ukladať do priečinka: '{output_dir}'")

    df = load_and_prepare_data()
    if df is None: return

    G_original = get_correlation_graph(df, alpha=0.1)
    G_clean = preprocess_graph(G_original.copy())

    if G_clean.number_of_nodes() == 0:
        print("Graf je po očistení prázdny, nie je čo kresliť.")
        return

    print("\n--- Generujem pokročilé vizualizácie grafu s metrikami ---")
    create_improved_radial_layout(G_clean, os.path.join(output_dir, "layout_1_radial.png"))
    create_sorted_grid_layout(G_clean, os.path.join(output_dir, "layout_2_grid.png"))
    create_orthogonal_layout_from_scratch(G_clean, os.path.join(output_dir, "layout_3_orthogonal.png"))

    print("\nProces je dokončený. 👍")


if __name__ == "__main__":
    main()