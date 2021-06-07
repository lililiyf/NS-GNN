import metis
import torch
import random
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import collections
# from ica.utils import load_data, pick_aggregator, create_map, build_graph
# from ica.classifiers import LocalClassifier, RelationalClassifier, ICA
# from scipy.stats import sem
# from sklearn.metrics import accuracy_score
# import sklearn
# from ssl_utils import encode_onehot
# from sklearn_extra.cluster import KMedoids
# from utils import row_normalize
import numpy as np
import numba
from numba import njit
import os
import time
import sys
np.set_printoptions(threshold=sys.maxsize)


class ClusteringMachine(object):
    """
    Clustering the graph, feature set and target.
    """
    def __init__(self, adj, cluster_number=5, clustering_method='metis'):
        """
        :param graph: Networkx Graph.
        """
        self.adj = adj
        #self.features = features.cpu().numpy()
        self.graph = nx.from_scipy_sparse_matrix(adj)
        self.clustering_method = clustering_method
        self.cluster_number = cluster_number

    def decompose(self):
        """
        Decomposing the graph, partitioning, creating Torch arrays.
        """
        if self.clustering_method == "metis":
            #print("\nMetis graph clustering started.\n")
            #print('c1')
            self.metis_clustering()
            #print('c2')
            central_nodes = self.get_central_nodes()
            #print('c3')
            self.shortest_path_to_clusters(central_nodes)
            #print('c3')

        self.dis_matrix = torch.FloatTensor(self.dis_matrix)
        # self.transfer_edges_and_nodes()

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        """
        Clustering the graph with Metis. For details see:
        """
        #print('m1')
        (st, parts) = metis.part_graph(self.graph, self.cluster_number)
        #print('debug:adj',self.adj.A)
        #print('debug: after metis.part_graph ')
        #print('debug: parts:',parts)
        self.clusters = list(set(parts))
        #print('debug: clusters:', self.clusters)
        #print('m3')
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}
        #print('debug: cluster_membership：', self.cluster_membership)
        #print('m4')

    # def kmedoids_clustering(self):
    #     from sklearn.decomposition import PCA, TruncatedSVD
    #     # pca = TruncatedSVD(n_components=600, n_iter=5000, algorithm='arpack')
    #     # pca = PCA(n_components=256)
    #     # self.features_pca = pca.fit_transform(self.features)
    #     # from sklearn import preprocessing
    #     # scaler = preprocessing.StandardScaler()
    #     # self.features_pca = scaler.fit_transform(self.features_pca)
    #     self.features_pca = self.features
    #     kmedoids = KMedoids(n_clusters=self.cluster_number,
    #             random_state=0).fit(self.features_pca)
    #     self.clusters = list(set(kmedoids.labels_))
    #     self.cluster_membership = kmedoids.labels_
    #
    #     return self.find_kmedoids_center(kmedoids.cluster_centers_)

    # def find_kmedoids_center(self, centers_ori):
    #     centers = []
    #     for ii, x in enumerate(self.features_pca):
    #         # if self.features.shape[1] in (centers_ori == x).sum(1):
    #         idx = np.where((centers_ori == x).sum(1) == self.features_pca.shape[1])[0]
    #         if len(idx):
    #             centers_ori[idx, :] = -1
    #             centers.append(ii)
    #
    #     assert len(centers) == self.cluster_number, "duplicate features"
    #     return centers

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        for cluster in self.clusters:
            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            #print('debug:cluster:',cluster)
            # print('debug:subgraph.edges()',subgraph.edges())
            # print('debug:subgraph.nodes()',subgraph.nodes())
            #print('debug:sg_nodes[cluster]:',self.sg_nodes[cluster])
            #print('debug:sg_nodes[cluster][0]:',self.sg_nodes[cluster][0])
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            #self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()), test_size = 0.8)
            #self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
            #self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])
        #print('Number of nodes in clusters:', {x: len(y) for x,y in self.sg_nodes.items()})

    def get_central_nodes(self):
        """
        set the central node as the node with highest degree in the cluster
        """
        self.general_data_partitioning()
        central_nodes = {}
        for cluster in self.clusters:
            counter = {}
            #print('debug:cluster:',cluster)
            #print('debug:self.sg_edges[cluster]',self.sg_edges[cluster])
            if len(self.sg_edges[cluster])==0:
                central_nodes[cluster] = self.sg_nodes[cluster][0]
            else:
                for node, _ in self.sg_edges[cluster]:
                    counter[node] = counter.get(node, 0) + 1
                #print('debug:counter.items:',counter.items())

                sorted_counter = sorted(counter.items(), key=lambda x:x[1])
                #print('debug:cluster:',cluster)
                #print('debug:sorted_counter:',sorted_counter)
                #print('debug:sorted_counter[-1][0]:',sorted_counter[-1][0])
                central_nodes[cluster] = sorted_counter[-1][0]
            #print('debug:central_nodes[cluster]:',central_nodes[cluster])
        return central_nodes

    def transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format.
        """
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])

    def transform_depth(self, depth):
        return 1 / depth

    def shortest_path_to_clusters(self, central_nodes, transform=True):
        """
        Do BFS on each central node, then we can get a node set for each cluster
        which is within k-hop neighborhood of the cluster.
        """
        # self.distance = {c:{} for c in self.clusters}
        self.dis_matrix = -np.ones((self.adj.shape[0], self.cluster_number))
        for cluster in self.clusters:
            node_cur = central_nodes[cluster]
            visited = set([node_cur])
            q = collections.deque([(x, 1) for x in self.graph.neighbors(node_cur)])
            while q:
                node_cur, depth = q.popleft()
                if node_cur in visited:
                    continue
                visited.add(node_cur)
                # if depth not in self.distance[cluster]:
                #     self.distance[cluster][depth] = []
                # self.distance[cluster][depth].append(node_cur)
                if transform:
                    self.dis_matrix[node_cur][cluster] = self.transform_depth(depth)
                else:
                    self.dis_matrix[node_cur][cluster] = depth
                for node_next in self.graph.neighbors(node_cur):
                    q.append((node_next, depth+1))

        if transform:
            self.dis_matrix[self.dis_matrix==-1] = 0
        else:
            self.dis_matrix[self.dis_matrix==-1] = self.dis_matrix.max() + 2
        return self.dis_matrix

    def dfs(self, node_cur, visited, path, cluster_id):
        for node_next in self.graph.neighbors(node_cur):
            if node_next in visited:
                continue
            visited.add(node_next)
            if path not in self.distance[cluster_id]:
                self.distance[cluster_id][path] = []
            self.distance[cluster_id][path].append(node_next)
            self.bfs(node_next, visited, path+1, cluster_id)




