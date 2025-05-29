import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import datetime

# Step 1: Load the dataset
try:
    df = pd.read_csv('datasets/energydata.csv')
except FileNotFoundError:
    print("The dataset file 'abalone.csv' was not found. Please ensure the file exists in the current directory.")
    exit()

df = df.iloc[:, 1:]

# Step 2: Compute correlation matrix
correlation_matrix = df.corr()
print("Correlation Matrix:\n", correlation_matrix)

# Step 3: Trim the correlation matrix
alpha = 0.01  # Adjust this value as needed

# Compute the threshold dynamically
cor_max = np.max(correlation_matrix.values)  # Maximum correlation
cor_mean = np.mean(correlation_matrix.values)  # Mean correlation
threshold = (cor_max + cor_mean) / 2 + alpha

trimmed_matrix = np.where(np.abs(correlation_matrix) > threshold, correlation_matrix, 0)
print("\nTrimmed Correlation Matrix:\n", trimmed_matrix)

# Step 4: Construct the graph
G = nx.Graph()
nodes = correlation_matrix.columns.tolist()
G.add_nodes_from(nodes)

for i in range(len(trimmed_matrix)):
    for j in range(i + 1, len(trimmed_matrix)):
        if trimmed_matrix[i][j] != 0:
            G.add_edge(nodes[i], nodes[j], weight=trimmed_matrix[i][j])

print("\nGraph Summary:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Function to find claw subgraphs
def find_claw_subgraphs(graph):
    claws = []
    for central_node in graph.nodes():
        neighbors = list(graph.neighbors(central_node))
        if len(neighbors) >= 3:
            for trio in combinations(neighbors, 3):
                if not any(graph.has_edge(u, v) for u, v in combinations(trio, 2)):
                    claws.append((central_node,) + trio)
    return claws

claws = find_claw_subgraphs(G)
print("\nClaw Subgraphs Found:", claws if claws else "None")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Step 5: Visualize and save the main graph
plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G)
# pos = nx.kamada_kawai_layout(G)  # Evenly spreads out nodes
# pos = nx.fruchterman_reingold_layout(G)  # Similar to spring but more balanced
pos = nx.circular_layout(G)  # Arranges nodes in a circle
# pos = nx.spectral_layout(G)  # Uses matrix eigenvalues for positioning

edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}

nx.draw(G, pos, with_labels=True, node_size=800, font_size=10, edge_color="gray", alpha=0.8)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
# nx.draw(G, pos=pos, node_size=700, alpha=0.8, edge_color='gray', linewidths=1, font_size=10)
plt.savefig(f'outputs/main_graph_{timestamp}.png')
print("\nMain graph visualization saved as 'main_graph.png'.")

# Step 6: Visualize and save 3-to-5-node cliques
cliques = list(nx.find_cliques(G))  # Find all cliques
filtered_cliques_3 = [c for c in cliques if len(c) == 3]
filtered_cliques_4 = [c for c in cliques if len(c) == 4]
filtered_cliques_5 = [c for c in cliques if len(c) == 5]
# filtered_cliques = [c for c in cliques if 3 <= len(c) <= 5]  # Keep only 3 to 5-node cliques
print("\n3 Node Cliques Found:", filtered_cliques_3 if filtered_cliques_3 else "None")
print("\n4 Node Cliques Found:", filtered_cliques_4 if filtered_cliques_4 else "None")
print("\n5 Node Cliques Found:", filtered_cliques_5 if filtered_cliques_5 else "None")

plt.figure(figsize=(12, 8))
colors = [plt.cm.tab10(i) for i in range(len(filtered_cliques_3))]
# Draw base graph with low opacity
nx.draw(G, pos=pos, node_size=700, alpha=0.1, edge_color='gray', linewidths=1, font_size=10)

for i, clique in enumerate(filtered_cliques_3):
    subgraph = G.subgraph(clique)
    nx.draw_networkx_nodes(subgraph, pos=pos, node_size=700, node_color='orange', alpha=1)
    nx.draw_networkx_edges(subgraph, pos=pos, edge_color=colors[i], width=2)

plt.savefig(f'outputs/3cliques_{timestamp}.png')
print("\n3-Node cliques visualization saved as 'three_node_cliques.png'.")

# # Step 7: Visualize and save Claw Subgraphs
# if claws:
#     plt.figure(figsize=(12, 8))
#     nx.draw(G, pos=pos, node_size=700, alpha=0.3, edge_color='gray', linewidths=1, font_size=10)
#     for claw in claws:
#         subgraph = G.subgraph(claw)
#         nx.draw(subgraph, pos=pos, node_size=700, alpha=1, edge_color='blue', linewidths=2, font_size=10)
#     plt.savefig(f'outputs/claw_subgraphs_{timestamp}.png')
#     print("\nClaw subgraphs visualization saved as 'claw_subgraphs.png'.")
# else:
#     print("\nNo claw subgraphs found to visualize.")

# Step 7: Visualize Claw Subgraphs in Separate Subplots within One Figure
if claws:
    # Calculate the number of rows and columns for subplots
    num_claws = len(claws)
    fig_cols = 3  # Adjust based on how many you want per row
    fig_rows = (num_claws + fig_cols - 1) // fig_cols

    plt.figure(figsize=(fig_cols * 6, fig_rows * 4))

    for idx, claw in enumerate(claws):
        plt.subplot(fig_rows, fig_cols, idx + 1)

        # Extract nodes for this claw
        central_node = claw[0]
        peripheral_nodes = claw[1:]

        # Create a subgraph for this claw
        claw_subgraph = G.subgraph(claw)

        # Separate the central node from others for visualization
        pos = nx.spring_layout(claw_subgraph)

        # Draw nodes and edges
        nx.draw_networkx_nodes(claw_subgraph, pos, node_size=700, alpha=1)
        edge_labels_sub = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in claw_subgraph.edges()}
        nx.draw_networkx_edge_labels(claw_subgraph, pos, alpha=0.7, edge_labels=edge_labels_sub)

        # Highlight the central node with a different color
        nx.draw_networkx_nodes(claw_subgraph.subgraph([central_node]),
                               pos, node_size=700, alpha=1, node_color='red')

        # Draw labels
        nx.draw_networkx_edges(claw_subgraph, pos, alpha=0.7, edge_color="gray", width=1)
        nx.draw_networkx_labels(claw_subgraph, pos, font_size=10)

        plt.title(f'Claw {idx + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'outputs/claw_subgraphs_{timestamp}.png')
    print("\nClaw subgraphs visualization saved as 'claw_subgraphs.png'.")
else:
    print("\nNo claw subgraphs found to visualize.")
