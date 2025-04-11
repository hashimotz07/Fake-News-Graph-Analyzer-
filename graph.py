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


#df = pd.read_csv('Datasets/twitter_combined.txt', sep=' ', header=None, names=['user1', 'user2'])
df = pd.read_csv('Datasets/twitter_combined2.txt', sep=' ', header=None, names=['user1', 'user2'])



#print(df.head())
# Create a directed graph

'''
G = nx.DiGraph()
start = 0
for _, row in df.iterrows():
 G.add_edge(row['user1'], row['user2'])#, weight=row['weight'])
 #print("Added Edge!")
 progress_bar(start + 1,2420766)
 start += 1
'''

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
    progress_bar(start + 1,14)
    start += 1


# Compute PageRank
pagerank = nx.pagerank(G, alpha=0.85)


# Detect communities
#from networkx.algorithms.community import greedy_modularity_communities
#communities = greedy_modularity_communities(G)


print("Created Communities")

# Draw Graph
#plt.figure(figsize=(10, 7))
print("Figure Size Set")
nx.draw(G, with_labels=False, node_size=20)
print("Drew Graph")
plt.show()
