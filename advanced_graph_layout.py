# Súbor: advanced_graph_layout.py
import math
import numpy as np
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def physics_based_component_positioning(component_info):
    """Použije fyzikálnu simuláciu na usporiadanie komponentov bez prekrývania."""
    print("  Inicializujem pozície komponentov...")
    for i, info in enumerate(component_info):
        if i == 0:
            info['center_x'], info['center_y'] = 0, 0
        else:
            angle = 2 * math.pi * (i - 1) / (len(component_info) - 1) if len(component_info) > 1 else 0
            initial_radius = sum(c['radius'] for c in component_info[:i]) / (i * 0.8) if i > 0 else 8
            info['center_x'] = initial_radius * math.cos(angle)
            info['center_y'] = initial_radius * math.sin(angle)
        info['original_center_x'], info['original_center_y'] = info['center_x'], info['center_y']

    max_iterations, dt, damping, repulsion_strength = 200, 0.1, 0.95, 50

    for iteration in range(max_iterations):
        total_force = 0
        for i, info in enumerate(component_info):
            force_x, force_y = 0, 0
            for j, other_info in enumerate(component_info):
                if i == j: continue
                dx, dy = info['center_x'] - other_info['center_x'], info['center_y'] - other_info['center_y']
                distance = math.sqrt(dx * dx + dy * dy)
                min_distance = info['radius'] + other_info['radius'] + 0.5
                if 0 < distance < min_distance:
                    overlap = min_distance - distance
                    force_magnitude = repulsion_strength * overlap
                    force_x += force_magnitude * (dx / distance)
                    force_y += force_magnitude * (dy / distance)
            info['velocity_x'] = (info['velocity_x'] + force_x * dt) * damping
            info['velocity_y'] = (info['velocity_y'] + force_y * dt) * damping
            info['center_x'] += info['velocity_x'] * dt
            info['center_y'] += info['velocity_y'] * dt
            total_force += abs(force_x) + abs(force_y)
        if total_force < 1.0:
            print(f"  Fyzikálna simulácia skonvergovala po {iteration + 1} iteráciách.")
            break
    else:
        print(f"  Fyzikálna simulácia skončila po {max_iterations} iteráciách.")
    return component_info


def create_component_internal_layout(component, spread_factor):
    """Vytvorí interný layout pre jeden komponent grafu."""
    if component.number_of_nodes() <= 2:
        return nx.spring_layout(component, scale=spread_factor, seed=42)

    pos = {}
    try:
        # Nájdenie "chrbtice" komponentu
        ecc = nx.eccentricity(component)
        end_node1 = max(ecc, key=ecc.get)
        distances = nx.shortest_path_length(component, source=end_node1)
        end_node2 = max(distances, key=distances.get)
        backbone = nx.shortest_path(component, source=end_node1, target=end_node2)
    except Exception:
        backbone = sorted(list(component.nodes()))[:min(5, component.number_of_nodes())]

    main_y_coords = np.linspace(len(backbone) / 2.0, -len(backbone) / 2.0, len(backbone)) * spread_factor
    for i, node in enumerate(backbone):
        pos[node] = (0, main_y_coords[i])

    processed = set(backbone)
    queue = deque([(node, 0) for node in backbone])
    side_offsets = {node: {'left': spread_factor, 'right': spread_factor} for node in backbone}

    while queue:
        parent, level = queue.popleft()
        px, py = pos[parent]
        neighbors = sorted([n for n in component.neighbors(parent) if n not in processed])

        for i, node in enumerate(neighbors):
            side = 'right' if i % 2 == 0 else 'left'
            offset = side_offsets[parent][side]
            sign = 1 if side == 'right' else -1
            pos[node] = (px + sign * offset, py)
            side_offsets[parent][side] += spread_factor
            processed.add(node)
            queue.append((node, level + 1))

    return pos


def generate_physics_based_pos(graph):
    """Hlavná funkcia, ktorá vygeneruje pozície uzlov pomocou fyzikálnej simulácie."""
    if nx.is_connected(graph):
        components = [graph]
    else:
        components = sorted([graph.subgraph(c).copy() for c in nx.connected_components(graph)], key=len, reverse=True)

    component_layouts, component_info = [], []
    max_nodes = len(components[0]) if components else 1

    for comp_idx, component in enumerate(components):
        spread_factor = 0.8 + (len(component) / max_nodes * 1.5)
        component_pos = create_component_internal_layout(component, spread_factor)

        if component_pos:
            x_coords, y_coords = zip(*component_pos.values())
            center_x, center_y = (np.mean(x_coords), np.mean(y_coords))
            radius = max(math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for x, y in component_pos.values()) + 0.8
        else:
            center_x, center_y, radius = 0, 0, 2

        component_layouts.append(component_pos)
        component_info.append({'center_x': 0, 'center_y': 0, 'radius': radius, 'velocity_x': 0, 'velocity_y': 0,
                               'original_pos': component_pos})

    component_info = physics_based_component_positioning(component_info)

    final_pos = {}
    for info in component_info:
        dx, dy = info['center_x'], info['center_y']
        for node, (x, y) in info['original_pos'].items():
            final_pos[node] = (x + dx, y + dy)

    return final_pos


# --- Funkcie pre kreslenie zakrivených hrán (skopírované) ---
def point_to_line_distance(point, line_start, line_end):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx, dy = x2 - x1, y2 - y1
    line_mag_sq = dx * dx + dy * dy
    if line_mag_sq == 0: return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / line_mag_sq))
    closest_x, closest_y = x1 + t * dx, y1 + t * dy
    return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


def does_edge_intersect_node(edge_start, edge_end, node_pos, node_radius=0.5):
    return point_to_line_distance(node_pos, edge_start, edge_end) < node_radius


def get_curved_path_points(p1, p2, rad, num_points=20):
    if rad == 0:
        t = np.linspace(0, 1, num_points)
        return [(p1[0] * (1 - v) + p2[0] * v, p1[1] * (1 - v) + p2[1] * v) for v in t]

    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    line_length = math.sqrt(dx * dx + dy * dy)
    if line_length == 0: return [p1] * num_points

    perp_x, perp_y = -dy / line_length, dx / line_length
    max_offset = rad * line_length * 0.5

    t = np.linspace(0, 1, num_points)
    points = []
    for v in t:
        straight_x, straight_y = p1[0] * (1 - v) + p2[0] * v, p1[1] * (1 - v) + p2[1] * v
        curve_strength = 4 * v * (1 - v)
        offset = max_offset * curve_strength
        points.append((straight_x + offset * perp_x, straight_y + offset * perp_y))
    return points


def does_curved_edge_intersect_nodes(p1, p2, rad, graph, pos, u, v, node_radius=0.5):
    path_points = get_curved_path_points(p1, p2, rad)
    for node in graph.nodes():
        if node in (u, v): continue
        for point in path_points:
            if math.sqrt((point[0] - pos[node][0]) ** 2 + (point[1] - pos[node][1]) ** 2) < node_radius:
                return True
    return False


def find_best_curve_for_edge(graph, pos, u, v, node_radius=0.5):
    p1, p2 = pos[u], pos[v]
    if not any(does_edge_intersect_node(p1, p2, pos[n], node_radius) for n in graph.nodes() if n not in (u, v)):
        return "arc3,rad=0"

    for rad in np.concatenate([np.linspace(0.3, 1.5, 7), np.linspace(-0.3, -1.5, 7)]):
        if not does_curved_edge_intersect_nodes(p1, p2, rad, graph, pos, u, v, node_radius):
            return f"arc3,rad={rad}"
    return "arc3,rad=1.0"  # Fallback