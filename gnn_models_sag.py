import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dpdgat import dpdGATConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_mean_pool
import pickle


class dpdGAT(nn.Module):
    def __init__(self, in_channels=128, out_channels=64, heads=2, dropout=0.4):
        super(dpdGAT, self).__init__()
        self.init_x= nn.Linear(in_channels, out_channels)
        self.conv1 = dpdGATConv(out_channels, out_channels, heads, dropout)
        self.conv2 = dpdGATConv(out_channels, out_channels, heads, dropout)
        self.conv3 = dpdGATConv(out_channels, out_channels, heads, dropout)
        self.conv4 = dpdGATConv(out_channels, out_channels, heads, dropout)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.bn4 = nn.BatchNorm1d(out_channels)
        self.fc1 = nn.Linear(heads * out_channels, out_channels)
        self.fc2 = nn.Linear(heads * out_channels, out_channels)
        self.fc3 = nn.Linear(heads * out_channels, out_channels)
        self.fc4 = nn.Linear(heads * out_channels, out_channels)
        self.fc = nn.Linear(out_channels, 7) 

    def forward(self, x, edge_index, train_edge_id, p=0.5):
        x = self.init_x(x)

        x = self.conv1(x, edge_index)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.conv3(x, edge_index)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.conv4(x, edge_index)
        x = F.relu(self.fc4(x))
        x = self.bn4(x)

        x = F.dropout(x, p, training=self.training)
        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]
        x = torch.mul(x1, x2)
        y = self.fc(x)

        return y




class TransGCN(nn.Module):
    def __init__(self, hidden = 128, heads=2, num_layers=1, weight=0.1):
        super(TransGCN, self).__init__()
        self.transformer_layer1 = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads,dim_feedforward=hidden,dropout=0.1,activation="relu")
        self.transformer1 = nn.TransformerEncoder(self.transformer_layer1, num_layers)
        self.transformer_layer2 = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads,dim_feedforward=hidden,dropout=0.1,activation="relu")
        self.transformer2 = nn.TransformerEncoder(self.transformer_layer1, num_layers)
        self.transformer_layer3 = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads,dim_feedforward=hidden,dropout=0.1,activation="relu")
        self.transformer3 = nn.TransformerEncoder(self.transformer_layer1, num_layers)
        self.transformer_layer4 = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads,dim_feedforward=hidden,dropout=0.1,activation="relu")
        self.transformer4 = nn.TransformerEncoder(self.transformer_layer1, num_layers)
        self.a = nn.Parameter(torch.tensor(weight))


        self.lin  =  nn.Linear(7,hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)
        self.sag1 = SAGPooling(hidden,0.5)
        self.sag2 = SAGPooling(hidden,0.5)
        self.sag3 = SAGPooling(hidden,0.5)
        self.sag4 = SAGPooling(hidden,0.5)

        self.tc1 = nn.Linear(heads * hidden, hidden)
        self.tc2 = nn.Linear(heads * hidden, hidden)
        self.tc3 = nn.Linear(heads * hidden, hidden)
        self.tc4 = nn.Linear(heads * hidden, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(0.5)
        # for param in self.parameters():
        #     print(type(param), param.size())


    def forward(self, x, edge_index, batch):
        x=self.lin(x)

        x_t1 = x
        x = self.transformer1(x.unsqueeze(1)).squeeze(1)
        x = self.a*x + (1-self.a)*x_t1
        x = self.conv1(x, edge_index)
        x = F.relu(self.fc1(x)) 
        x = self.bn1(x)
        y = self.sag1(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1] 


        x_t2 = x
        x = self.transformer2(x.unsqueeze(1)).squeeze(1)
        x = self.a*x + (1-self.a)*x_t2
        x = self.conv2(x, edge_index)
        x = F.relu(self.fc2(x)) 
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]  
        
        x_t3 = x
        x = self.transformer3(x.unsqueeze(1)).squeeze(1)
        x = self.a*x + (1-self.a)*x_t3
        x = self.conv3(x, edge_index)
        x = F.relu(self.fc3(x)) 
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        # x_t4 = x
        # x = self.transformer4(x.unsqueeze(1)).squeeze(1)
        # x = self.a*x + (1-self.a)*x_t4
        x = self.conv4(x, edge_index)
        x = F.relu(self.fc4(x)) 
        x = self.bn4(x)
        y = self.sag4(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]
        return global_mean_pool(y[0], y[3])

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class ppi_model(nn.Module):
    def __init__(self):
        super(ppi_model,self).__init__()
        self.BGNN = TransGCN()
        self.TGNN = dpdGAT()

    def forward(self, batch, p_x_all, p_edge_all, edge_index, train_edge_id, p=0.5):
        edge_index = edge_index.to(device)
        batch = batch.to(torch.int64).to(device)
        x = p_x_all.to(torch.float32).to(device)
        edge = torch.LongTensor(p_edge_all.to(torch.int64)).to(device)
        embs = self.BGNN(x, edge, batch-1)
        final = self.TGNN(embs, edge_index, train_edge_id, p=0.5)
        return final





