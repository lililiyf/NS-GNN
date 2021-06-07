import dgl
import torch as th
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import scipy.sparse as sp
from loader import BioDataset
import numpy as np

 #set up dataset
root_unsupervised = 'dataset/unsupervised'
dataset = BioDataset(root_unsupervised, data_type='unsupervised')
edge_attr = dataset[0].edge_attr.clone().detach()
edge_index = dataset[0].edge_index.clone().detach()
center_node_idx = dataset[0].center_node_idx.clone().detach()
x = dataset[0].x.clone().detach()
species_id = dataset[0].species_id.clone().detach()

indices = edge_index.numpy().copy()
values = np.ones(len(edge_index.clone().detach().t()))
nodes_num = len(x)
graph_adj = sp.csr_matrix((values, indices), shape=[nodes_num, nodes_num])
G = nx.from_scipy_sparse_matrix(graph_adj)

colors=[]
for i in range(nodes_num):
    colors.append(0.3)

colors[center_node_idx]=1

nodeLabel=th.tensor(list(range(len(x))))

pos = nx.spring_layout(G)
nx.draw(G, pos, edge_color="grey",node_color=colors, node_size=500, with_labels=True)  # 画图，设置节点大小
plt.savefig("1.png")
edge_labels = {}

i_1=0
for edge in G.edges:
    edge_labels[edge] = edge_attr[i_1].tolist()
    i_1=i_1+1 # G[edge[0]][edge[1]] will return all attributes of edge

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.savefig("2.png")
plt.show()

plt.savefig("3.png")