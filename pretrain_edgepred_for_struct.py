import argparse

from loader import BioDataset
from dataloader import DataLoaderAE
from util import NegativeEdge

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model_ns import GNN
import pandas as pd

criterion = nn.BCEWithLogitsLoss()

def train(args, model, device, loader, optimizer):
    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        _,struct_emb = model(batch.x, batch.edge_index, batch.edge_attr)

        positive_score = torch.sum(struct_emb[batch.edge_index[0, ::2]] * struct_emb[batch.edge_index[1, ::2]], dim = 1)
        negative_score = torch.sum(struct_emb[batch.negative_edge_index[0]] * struct_emb[batch.negative_edge_index[1]], dim = 1)

        optimizer.zero_grad()
        loss = criterion(positive_score, torch.ones_like(positive_score)) + criterion(negative_score, torch.zeros_like(negative_score))
        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum/(step+1), train_loss_accum/(step + 1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--input_model_file', type=str, default='param_struct_6', help='filename to input the pre-trained model')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 0, help='number of workers for dataset loading')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up dataset
    root_unsupervised = 'dataset/unsupervised'
    dataset = BioDataset(root_unsupervised, data_type='unsupervised', transform = NegativeEdge())

    print(dataset)

    loader = DataLoaderAE(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up model
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    model.load_state_dict(torch.load(args.input_model_file+ ".pth", map_location=lambda storage, loc: storage))
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    #optimizer = optim.Adam(model.graph_pred_linear.parameters(), lr=args.lr, weight_decay=args.decay)    
    print(optimizer)


    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train_acc, train_loss = train(args, model, device, loader, optimizer)

        print(train_acc)
        print(train_loss)

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")



if __name__ == "__main__":
    main()