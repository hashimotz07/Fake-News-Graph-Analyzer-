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

# Carregando o dataset
df = pd.read_csv('Datasets/twitter_combined.txt', sep=' ', header=None, names=['user1', 'user2'])
df['weight'] = 1
df = df.groupby(['user1', 'user2'], as_index=False).count()

# --------------------------
# Construindo o Grafo
# --------------------------

print("Construindo grafo...")
G = nx.from_pandas_edgelist(df, source='user1', target='user2', edge_attr='weight', create_using=nx.DiGraph())
print("Grafo construido!")

# --------------------------
# Lendo dados do Grafo
# --------------------------

print(f"Especificações sobre o Grafo criado:")
print(f"Número de retweets/mentions: {G.number_of_edges()}")
print(f"Número de usuários: {len(G)}")
print(f"Peso total do Grafo: {G.size(weight='weight')}")
# Encontrar a aresta de maior peso
max_edge = max(G.edges(data=True), key=lambda x: x[2].get('weight', 0))

# max_edge será uma tupla (u, v, attr_dict), por exemplo: ('A', 'C', {'weight': 5})
maior_aresta = (max_edge[0], max_edge[1])
maior_peso = max_edge[2]['weight']

print(f"Aresta de maior peso: {maior_aresta}, com peso {maior_peso}")

# Vértice com maior grau total (entrada + saída)
max_node = max(G.nodes(), key=lambda node: G.in_degree(node) + G.out_degree(node))
max_total_degree = G.in_degree(max_node) + G.out_degree(max_node)

print(f"Vértice de maior grau total: {max_node} (grau {max_total_degree})")

# Vértice com maior grau de entrada
max_in_node = max(G.nodes(), key=lambda node: G.in_degree(node))
max_in_degree = G.in_degree(max_in_node)

# Vértice com maior grau de saída
max_out_node = max(G.nodes(), key=lambda node: G.out_degree(node))
max_out_degree = G.out_degree(max_out_node)

print(f"Vértice com maior grau de entrada: {max_in_node} (grau de entrada {max_in_degree})")
print(f"Vértice com maior grau de saída: {max_out_node} (grau de saída {max_out_degree})")

# ---------------------------------
# Calcular Medidas de centralidade
# ---------------------------------

print("Calculando Grau de centralidade...")
degree_centrality = nx.degree_centrality(G)


print("Calculando Conexão de centralidade (betweenness) de 100 nós selecionados aleatóriamente...")
approx_betweenness = nx.betweenness_centrality(G, k=100, seed=42, normalized=True)

k = 100
# Amostra aleatória de k nós
sampled_nodes = random.sample(list(G.nodes()), k)

# Subgrafo com os nós amostrados
F = G.subgraph(sampled_nodes)

# Calculando a centralidade de proximidade apenas para os nós amostrados
aprox_closeness = nx.closeness_centrality(F)


# Exibir os top 5 para cada métrica
def print_top_centrality(centrality_dict, name):
    sorted_centrality = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"Top 5 por {name}:")
    for node, value in sorted_centrality[:5]:
        print(f"{node}: {value:.4f}")
    print()

print_top_centrality(degree_centrality, "Grau de centralidade")
#print_top_centrality(approx_betweenness, "centralidade de intermediação")
print_top_centrality(approx_betweenness, "Conexão de centralidade (aproximada 100 nós)")
print_top_centrality(aprox_closeness, "Proximidade de centralidade (aproximada 100 nós)")

# --------------------------
# Calcular o pagerank
# --------------------------

print("Computando PageRank...")
pagerank = nx.pagerank(G, alpha=0.85)
sorted_pagerank = dict(sorted(pagerank.items(), key=lambda item: item[1], reverse=True))
print("Top 10 nós do PageRank:")
print(dict(list(sorted_pagerank.items())[:10]))

# --------------------------
# Calcular as comunidades
# --------------------------

# Detecção de comunidade usando Louvain
print("Detectando comunidades com o Louvain (detecta comunidades assumindo grafo não direcionado)...")
communities = algorithms.louvain(G.to_undirected())
print(f"Encontradas {len(communities.communities)} comunidades")


# Detecção de comunidade usando Infomap
print("Detectando comunidades com o Infomap (detecta comunidades no grafo direcionado nativo)...")
communities2 = algorithms.infomap(G)
print(f"Encontradas {len(communities2.communities)} comunidades")



# --------------------------
# Gerar um Subgrafo
# --------------------------

# Visualize only a small subgraph (top-k nodes by PageRank)
print("Preparando visualização do grafo...")
top_nodes = list(dict(list(sorted_pagerank.items())[:1000]).keys())  # Top 200 nodes
H = G.subgraph(top_nodes).copy()

# Pega as cores conforme as comunidades Louvain
coms = communities.communities
color_map = get_community_color_map(coms)

weights = np.array([d['weight'] for _, _, d in H.edges(data=True)])
min_w, max_w = weights.min(), weights.max()

norm = mcolors.Normalize(vmin=min_w, vmax=max_w)
cmap = cm.get_cmap('RdYlGn_r')  # green (low) to red (high)
edge_colors = [cmap(norm(d['weight'])) for _, _, d in H.edges(data=True)]

pos = nx.kamada_kawai_layout(H)

# Assinalar cores dos nós
node_colors = [color_map.get(node, (0.5, 0.5, 0.5)) for node in H.nodes()]

# --------------------------
# Plotar o Subgrafo
# --------------------------

print("Desenhando Subgrafo (Top 1000 nós do PageRank)...")
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

plt.title("Top 1000 nós do PageRank (comunidades louvain separadas em cores)")
#plt.show()


# -----------------------------
# Plotar distribuição de graus
# -----------------------------

print("Pegando informações sobre graus...")
# Grau total (entrada + saída)
degree_sequence = [G.degree(n) for n in G.nodes()]
# Grau de entrada
in_degree_sequence = [G.in_degree(n) for n in G.nodes()]
# Grau de saída
out_degree_sequence = [G.out_degree(n) for n in G.nodes()]

print("Desenhando Histogramas para plotar distribuição de graus...")
plt.figure(figsize=(15, 5))

# Grau total
plt.subplot(1, 3, 1)
plt.hist(degree_sequence, bins=100, color='skyblue', edgecolor='black', log=True)
plt.title("Distribuição de Grau Total")
plt.xlabel("Grau")
plt.ylabel("Frequência (log)")

# Grau de entrada
plt.subplot(1, 3, 2)
plt.hist(in_degree_sequence, bins=100, color='lightgreen', edgecolor='black', log=True)
plt.title("Distribuição de Grau de Entrada")
plt.xlabel("Grau de Entrada")
plt.ylabel("Frequência (log)")

# Grau de saída
plt.subplot(1, 3, 3)
plt.hist(out_degree_sequence, bins=100, color='salmon', edgecolor='black', log=True)
plt.title("Distribuição de Grau de Saída")
plt.xlabel("Grau de Saída")
plt.ylabel("Frequência (log)")

plt.tight_layout()
plt.show()