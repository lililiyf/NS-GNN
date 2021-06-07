import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        #return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(2*num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(2*num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x_node = self.x_embedding1(x[:,0])+self.x_embedding2(x[:,1])
        x_struct=x_node

        #h_list = [x]
        h_node_list = [x_node]
        h_struct_list = [x_struct]
        for layer in range(self.num_layer):
            h_node = self.gnns[layer](h_node_list[layer], edge_index, edge_attr)
            h_node = self.batch_norms[layer](h_node)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h_node = F.dropout(h_node, self.drop_ratio, training = self.training)
            else:
                h_node = F.dropout(F.relu(h_node), self.drop_ratio, training = self.training)
            h_node_list.append(h_node)

        for layer in range(2*self.num_layer):
            h_struct = self.gnns[layer](h_struct_list[layer], edge_index, edge_attr)
            h_struct = self.batch_norms[layer](h_struct)
            if layer == 2*self.num_layer - 1:
                # remove relu for the last layer
                h_struct = F.dropout(h_struct, self.drop_ratio, training=self.training)
            else:
                h_struct = F.dropout(F.relu(h_struct), self.drop_ratio, training=self.training)
            h_struct_list.append(h_struct)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_node_list, dim = 1)
            struct_representation = torch.cat(h_struct_list, dim=1)
        elif self.JK == "last":
            node_representation = h_node_list[-1]
            struct_representation = h_struct_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_node_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
            h_list = [h.unsqueeze_(0) for h in h_struct_list]
            struct_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_node_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]
            h_list = [h.unsqueeze_(0) for h in h_struct_list]
            struct_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation,struct_representation


class GNN_graphpred(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(2*(self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(2*emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set(2*(self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(2*emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * 2*(self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * 2*self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        #self.gnn.load_state_dict(torch.load(model_file))
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation,struct_representation = self.gnn(x, edge_index, edge_attr)

        node_pooled = self.pool(node_representation, batch)
        struct_pooled = self.pool(struct_representation, batch)
        graph_rep = torch.cat([node_pooled, struct_pooled], dim=1)

        return self.graph_pred_linear(graph_rep)

class GNN_structpred(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, cluster_number, JK="last", drop_ratio=0,  gnn_type="gin"):
        super(GNN_structpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.cluster_number = cluster_number

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        if self.JK == "concat":
            self.strcut_pred_linear = torch.nn.Linear((self.num_layer*2 + 1) * self.emb_dim,
                                                     self.cluster_number)
        else:
            self.strcut_pred_linear = torch.nn.Linear(self.emb_dim, self.cluster_number)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        _,struct_representation= self.gnn(x, edge_index, edge_attr)

        return self.strcut_pred_linear(struct_representation)

class GNN_structpred_node(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, cluster_number, JK="last", drop_ratio=0,  gnn_type="gin"):
        super(GNN_structpred_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.cluster_number = cluster_number

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        if self.JK == "concat":
            self.strcut_pred_linear = torch.nn.Linear((self.num_layer*2 + 1) * self.emb_dim,
                                                     self.cluster_number)
        else:
            self.strcut_pred_linear = torch.nn.Linear(self.emb_dim, self.cluster_number)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation,_= self.gnn(x, edge_index, edge_attr)

        return self.strcut_pred_linear(node_representation)
if __name__ == "__main__":
    pass

