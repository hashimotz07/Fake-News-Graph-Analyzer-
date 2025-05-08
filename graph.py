import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    bar = chr(9608) * int(percent) + chr(9617) * (100 - int(percent))
    print(f"\r|{bar}| {percent:.2f}%", end="\r")
    if progress >= total:
       print("\n")
       print("Succesfully Created Graph!")
       return 0


# Load dataset (example format: user1, user2, weight)
df = pd.read_csv('Datasets/twitter_combined2.txt', sep=' ', header=None, names=['user1', 'user2'])

# Create a directed graph


print("Criando Grafo...")
G = nx.DiGraph()  # grafo direcionado (retweets/mentions têm direção)
start = 0
for _, row in df.iterrows():
    u1 = str(row['user1'])
    u2 = str(row['user2'])
    if G.has_edge(u1, u2):
        G[u1][u2]['weight'] += 1
    else:
        G.add_edge(u1, u2, weight=1)
    #progress_bar(start + 1,2420766)
    progress_bar(start + 1,543095)
    start += 1


# Compute PageRank
pagerank = nx.pagerank(G, alpha=0.85)
sorted_pagerank = dict(sorted(pagerank.items(), key=lambda item: item[1], reverse=True))
print(sorted_pagerank)

# Detect communities
from networkx.algorithms.community import greedy_modularity_communities
communities = greedy_modularity_communities(G)
print(communities)


print("Created Communities")

pos = nx.spring_layout(G, seed=42)  # This uses a force-directed layout (change seed for different results)

# Draw Graph
#plt.figure(figsize=(10, 7))
print("Figure Size Set")
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=False, node_size=20)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
print("Drew Graph")
plt.show()
