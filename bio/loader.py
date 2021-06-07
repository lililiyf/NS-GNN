import os
import torch
import random
import networkx as nx
import pandas as pd
import numpy as np
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
from collections import Counter, deque
from networkx.algorithms.traversal.breadth_first_search import generic_bfs_edges

import os.path as osp
import scipy.sparse as sp

#from distance import ClusteringMachine

def nx_to_graph_data_obj(g, center_id, allowable_features_downstream=None,
                         allowable_features_pretrain=None,
                         node_id_to_go_labels=None):
   
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

    # nodes
    nx_node_ids = [n_i for n_i in g.nodes()]  # contains list of nx node ids
    # in a particular ordering. Will be used as a mapping to convert
    # between nx node ids and data obj node indices

    x = torch.tensor(np.ones(n_nodes).reshape(-1, 1), dtype=torch.float)
    # we don't have any node labels, so set to dummy 1. dim n_nodes x 1

    center_node_idx = nx_node_ids.index(center_id)
    center_node_idx = torch.tensor([center_node_idx], dtype=torch.long)

    # edges
    edges_list = []
    edge_features_list = []
    for node_1, node_2, attr_dict in g.edges(data=True):
        edge_feature = [attr_dict['w1'], attr_dict['w2'], attr_dict['w3'],
                        attr_dict['w4'], attr_dict['w5'], attr_dict['w6'],
                        attr_dict['w7'], 0, 0]  # last 2 indicate self-loop
        # and masking
        edge_feature = np.array(edge_feature, dtype=int)
        # convert nx node ids to data obj node index
        i = nx_node_ids.index(node_1)
        j = nx_node_ids.index(node_2)
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
        edges_list.append((j, i))
        edge_features_list.append(edge_feature)

    # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

    # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    edge_attr = torch.tensor(np.array(edge_features_list),
                             dtype=torch.float)

    try:
        species_id = int(nx_node_ids[0].split('.')[0])  # nx node id is of the form:
        # species_id.protein_id
        species_id = torch.tensor([species_id], dtype=torch.long)
    except:  # occurs when nx node id has no species id info. For the extract
        # substructure context pair transform, where we convert a data obj to
        # a nx graph obj (which does not have original node id info)
        species_id = torch.tensor([0], dtype=torch.long)    # dummy species
        # id is 0

    # construct data obj
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.species_id = species_id
    data.center_node_idx = center_node_idx

    if node_id_to_go_labels:  # supervised case with go node labels
        # Construct a dim n_pretrain_go_classes tensor and a
        # n_downstream_go_classes tensor for the center node. 0 is no data
        # or negative, 1 is positive.
        downstream_go_node_feature = [0] * len(allowable_features_downstream)
        pretrain_go_node_feature = [0] * len(allowable_features_pretrain)
        if center_id in node_id_to_go_labels:
            go_labels = node_id_to_go_labels[center_id]
            # get indices of allowable_features_downstream that match with elements
            # in go_labels
            _, node_feature_indices, _ = np.intersect1d(
                allowable_features_downstream, go_labels, return_indices=True)
            for idx in node_feature_indices:
                downstream_go_node_feature[idx] = 1
            # get indices of allowable_features_pretrain that match with
            # elements in go_labels
            _, node_feature_indices, _ = np.intersect1d(
                allowable_features_pretrain, go_labels, return_indices=True)
            for idx in node_feature_indices:
                pretrain_go_node_feature[idx] = 1
        data.go_target_downstream = torch.tensor(np.array(downstream_go_node_feature),
                                        dtype=torch.long)
        data.go_target_pretrain = torch.tensor(np.array(pretrain_go_node_feature),
                                        dtype=torch.long)

    return data

def graph_data_obj_to_nx(data):
    """
    Converts pytorch geometric Data obj to network x data object.
    :param data: pytorch geometric Data object
    :return: nx graph object
    """
    G = nx.Graph()

    # edges
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    n_edges = edge_index.shape[1]
    for j in range(0, n_edges, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        w1, w2, w3, w4, w5, w6, w7, _, _ = edge_attr[j].astype(bool)
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5,
                       w6=w6, w7=w7)

    # # add center node id information in final nx graph object
    # nx.set_node_attributes(G, {data.center_node_idx.item(): True}, 'is_centre')

    return G


def change_existed(dataset_name, cluster_number):
    """
    本函数用于处理数据集，因为数据集没有process函数
    有可能会爆内存。

    """
    root = "dataset/" + dataset_name
    dataset = BioDataset(root, data_type=dataset_name, need_regenerate=False)

    datalist = []

    success_num = 0

    print('change_existed', success_num)

    # 循环所有图，生成新的数据集
    for i in range(len(dataset)):
        try:
            edge_attr = dataset[i].edge_attr.clone().detach()
            edge_index = dataset[i].edge_index.clone().detach()
            center_node_idx = dataset[i].center_node_idx.clone().detach()
            x = dataset[i].x.clone().detach()
            species_id = dataset[i].species_id.clone().detach()

            indices = edge_index.numpy().copy()
            values = np.ones(len(edge_index.clone().detach().t()))
            nodes_num = len(x)

            graph_adj = sp.csr_matrix((values, indices), shape=[nodes_num, nodes_num])

            cluster_agent = ClusteringMachine(graph_adj, cluster_number)
            cluster_agent.decompose()
            pseudo_labels = cluster_agent.dis_matrix

            # 生成新的图
            _d = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, species_id=species_id,
                  center_node_idx=center_node_idx, graph_cluster_distance=pseudo_labels)

            print('first succeed')
            success_num += 1
            # print("total num: {}".format(i))
            if success_num %1000 ==0 :
                print("success num: {}".format(success_num))
            # if success_num > 500:
            #    break
            datalist.append(_d)
        except:
            continue

    data, slices = InMemoryDataset.collate(datalist)

    # 生成新的pt文件
    target_path = "dataset/" + dataset_name + "/processed/geometric_data_processed_new.pt"
    torch.save((data, slices), target_path)


class BioDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 data_type,
                 empty=False,
                 transform=None,
                 pre_transform=None,
                 cluster_number=4,
                 need_regenerate=False,
                 pre_filter=None):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: the data directory that contains a raw and processed dir
        :param data_type: either supervised or unsupervised
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        :param transform:
        :param pre_transform:
        :param pre_filter:
        """
        self.root = root
        self.data_type = data_type
        self.cluster_number = cluster_number

        super(BioDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if need_regenerate:
            change_existed(self.data_type, self.cluster_number)
        
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
                 

    @property
    def raw_file_names(self):
        #raise NotImplementedError('Data is assumed to be processed')
        if self.data_type == 'supervised': # 8 labelled species
            file_name_list = ['3702', '6239', '511145', '7227', '9606', '10090', '4932', '7955']
        else: # unsupervised: 8 labelled species, and 42 top unlabelled species by n_nodes.
            file_name_list = ['3702', '6239', '511145', '7227', '9606', '10090',
            '4932', '7955', '3694', '39947', '10116', '443255', '9913', '13616',
            '3847', '4577', '8364', '9823', '9615', '9544', '9796', '3055', '7159',
            '9031', '7739', '395019', '88036', '9685', '9258', '9598', '485913',
            '44689', '9593', '7897', '31033', '749414', '59729', '536227', '4081',
            '8090', '9601', '749927', '13735', '448385', '457427', '3711', '479433',
            '479432', '28377', '9646']
        return file_name_list


    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return 
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        raise NotImplementedError('Data is assumed to be processed')

if __name__ == "__main__":
    # root_supervised = 'dataset/supervised'
    #
    # d_supervised = BioDataset(root_supervised, data_type='supervised')

    # print(d_supervised)

    root_unsupervised = 'dataset/unsupervised'
    d_unsupervised = BioDataset(root_unsupervised, data_type='unsupervised',cluster_number=4, need_regenerate=True)

    print(d_unsupervised)






