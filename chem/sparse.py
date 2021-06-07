import torch

graph_adj = torch.sparse_coo_tensor(indices=dataset[i].graph_adj.clone().detach().t(),
                                    values=torch.ones((len(dataset[i].graph_adj)), dtype=torch.int),
                                    size=(dataset[i].num_nodes, dataset[i].num_nodes))

graph_adj.