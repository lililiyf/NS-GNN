import argparse


from loader import BioDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_structpred,GNN_graphpred

from sklearn.metrics import roc_auc_score

import pandas as pd

from util import combine_dataset

criterion = nn.MSELoss()


def train(args, model, device, loader, optimizer):
    model.train()
    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        # y = batch.go_target_pretrain.view(pred.shape).to(torch.float64)
        y = batch.graph_cluster_distance.view(pred.shape).to(torch.float64)

        optimizer.zero_grad()
        loss = criterion(pred.double(), y)
        # loss = struct_task_nn.make_loss(pred)
        loss.backward()

        optimizer.step()
        loss_accum += float(loss.cpu().item())
    print('loss:',loss_accum/(step + 1))



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=2,
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
    parser.add_argument('--cluster_number', type=int, default=5,
                        help='strcut info task：cluster_number (default:5)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--input_model_file', type=str, default='param_struct_1',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type=str, default='param_struct_struct_1',
                        help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    #parser.add_argument('--seed', type=int, default=42, help="Seed for splitting dataset.")
    #parser.add_argument('--split', type=str, default="species", help='Random or species split')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    #print('start data')
    # set up dataset
    root_unsupervised = 'dataset/unsupervised'
    #dataset = BioDataset(root_unsupervised, data_type='unsupervised', cluster_number=args.cluster_number, need_regenerate=True)
    dataset = BioDataset(root_unsupervised, data_type='unsupervised', cluster_number=args.cluster_number)
    print(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # print('end load')
    # print('label:',dataset[0].graph_cluster_distance)
    # set up model
    model = GNN_structpred(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                            gnn_type=args.gnn_type,cluster_number=args.cluster_number)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file + ".pth")

    model.to(device)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train(args, model, device, loader, optimizer)

    if not args.output_model_file == "":
        torch.save(model.gnn.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
