import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from cdlib import algorithms
import random
import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def get_community_color_map(communities):
    color_map = {}
    num_coms = len(communities)
    colors = cm.get_cmap('tab20', num_coms)  # Or any colormap
    for idx, community in enumerate(communities):
        for node in community:
            color_map[node] = colors(idx)
    return color_map

# Load dataset
df = pd.read_csv('Datasets/twitter_combined.txt', sep=' ', header=None, names=['user1', 'user2'])
df['weight'] = 1
df = df.groupby(['user1', 'user2'], as_index=False).count()

# Build directed graph more efficiently
print("Building graph...")
G = nx.from_pandas_edgelist(df, source='user1', target='user2', edge_attr='weight', create_using=nx.DiGraph())
print("Graph built!")

# Compute PageRank
print("Calculating PageRank...")
pagerank = nx.pagerank(G, alpha=0.85)
sorted_pagerank = dict(sorted(pagerank.items(), key=lambda item: item[1], reverse=True))
print("Top 10 PageRank nodes:")
print(dict(list(sorted_pagerank.items())[:10]))

# Community detection using Louvain (via CDlib, scalable & efficient)
print("Detecting communities with louvain (detects communities assuming undirected graph)...")
communities = algorithms.louvain(G.to_undirected())
print(f"Found {len(communities.communities)} communities")

#print(communities.communities)
'''
print("Detecting communities with Infomap (detects communities in native directed graph)...")
communities2 = algorithms.infomap(G)
print(f"Found {len(communities2.communities)} communities")
'''
#print(communities2.communities)

# may need to be removed if too big in the terminal
#print(communities)
#print(communities2)

# Visualize only a small subgraph (top-k nodes by PageRank)
print("Preparing graph visualization...")
top_nodes = list(dict(list(sorted_pagerank.items())[:1000]).keys())  # Top 200 nodes
H = G.subgraph(top_nodes).copy()

# Choose community result (Louvain or Infomap)
coms = communities.communities  # or louvain_coms.communities
color_map = get_community_color_map(coms)

weights = np.array([d['weight'] for _, _, d in H.edges(data=True)])
min_w, max_w = weights.min(), weights.max()

norm = mcolors.Normalize(vmin=min_w, vmax=max_w)
cmap = cm.get_cmap('RdYlGn_r')  # green (low) to red (high)
edge_colors = [cmap(norm(d['weight'])) for _, _, d in H.edges(data=True)]

#pos = nx.spring_layout(H, seed=42)
pos = nx.kamada_kawai_layout(H)

# Assign node colors
node_colors = [color_map.get(node, (0.5, 0.5, 0.5)) for node in H.nodes()]

print("Drawing Subgraph (Top 1000 Nodes by PageRank)...")
plt.figure(figsize=(12, 8))
nx.draw(
    H,
    pos,
    node_color=node_colors,
    with_labels=False,
    node_size=30,
    edge_color=edge_colors,
    alpha=0.7
)
#edge_labels = nx.get_edge_attributes(H, 'weight')
#nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels)
plt.title("Top 1000 Nodes by PageRank (louvain communities highlighted)")
plt.show()