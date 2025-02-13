import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, h, adj):
        '''
                adj图邻接矩阵，维度[N,N]非零即一
                h.shape: (N, in_features), self.W.shape:(in_features,out_features)
                Wh.shape: (N, out_features)
        '''

        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e) #将没有链接的边设置为负无穷
        attention = torch.where(adj > 0, e, zero_vec) #(N,N)
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留
        # 否则需要mask设置为非常小的值，因为softmax的时候这个最小值会不考虑
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变[N,N]，得到归一化的注意力全忠！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout,防止过拟合
        h_prime = torch.matmul(attention, Wh)  # [N,N].[N,out_features]=>[N,out_features]

        #
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        #add multi-head
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x #去掉了softmax

class MultiGAT(nn.Module):
    def __init__(self, gat1, gat2, gat3, seq_len,heads,output_dim, hidden_dim):
        super(MultiGAT, self).__init__()
        self.gat1 = gat1
        self.gat2 = gat2
        self.gat3 = gat3
        # self.attention_layer = SelfAttention(seq_len,heads)

        self.fc2 =nn.Linear(
            self.gat1.attentions[0].in_features +
                            self.gat2.attentions[0].in_features
                            +self.gat3.attentions[0].in_features
                            +seq_len
                            ,512) ##  +++ self.gat2.attentions[0].in_features ++self.gat2.out_att.out_features
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x1, adj1, x2, adj2, x3, adj3):
        out1 = self.gat1(x1, adj1)
        out2 = self.gat2(x2, adj2)
        out3 = self.gat3(x3, adj3)
        out_concat = torch.cat([out1,out2,out3], dim=1) # out2,out3

        #
        # out_concat = out_concat.unsqueeze(-1) # seq_len= 30+128+128
        # out_concat = self.attention_layer(out_concat,out_concat,out_concat,None)
        out_hidden_concat = torch.cat([x1,x2,x3,out_concat], dim=1) #,,out_hidden
        out_hidden_concat = F.relu(self.fc2(out_hidden_concat))
        out_hidden_concat = F.relu(self.fc3(out_hidden_concat))
        out_hidden_concat = F.relu(self.fc4(out_hidden_concat))
        out_final = self.out(out_hidden_concat)
        return out_final

def read_adjacency_matrix_from_txt(file_path, num_nodes,device):
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        adjacency_matrix[i,i]=1
    with open(file_path, 'r') as f:
        for line in f:
            i, j = map(int, line.strip().split(','))
            adjacency_matrix[i, j] = 1
    return torch.tensor(adjacency_matrix).to(device)
def load_data(dataset_dir,city,datasets_file,device,neigh_num):
    df_dir = os.path.join(dataset_dir, f'{city}/{city}_{datasets_file}.csv')
    POI_df = pd.read_csv(df_dir)
    labels = POI_df['label'].astype(np.float32)
    labels= torch.tensor(labels).to(device)
    popu = POI_df['0.5'].apply(lambda x: np.fromstring(x[1:-1], sep=' ').astype(np.float32))
    # popu = POI_df['0.5'].apply(lambda x: np.array(eval(x)).astype(np.float32))
    popu = torch.tensor(popu).to(device)
    review = POI_df['embedding'].apply(lambda x: np.array(eval(x)).astype(np.float32))
    review = torch.tensor(review).to(device)
    rating = POI_df['rating'].apply(lambda x: np.array(eval(x)).astype(np.float32))
    rating = torch.tensor(rating).to(device)

    idx_train = POI_df[POI_df['split']==2].index.tolist()
    idx_val = POI_df[POI_df['split'] == 1].index.tolist()
    idx_test = POI_df[POI_df['split'] == 0].index.tolist()

    num_nodes = len(POI_df)
    del POI_df
    adjpop = read_adjacency_matrix_from_txt(os.path.join(dataset_dir, f'{city}/{city}_location_adj{neigh_num}.txt'), num_nodes,device)
    adjra = read_adjacency_matrix_from_txt(os.path.join(dataset_dir, f'{city}/{city}_rating_adj{neigh_num}.txt'), num_nodes,device)
    adjre = read_adjacency_matrix_from_txt(os.path.join(dataset_dir, f'{city}/{city}_review_adj{neigh_num}.txt'), num_nodes,device)

    return labels,popu,adjpop,review,adjre,rating,adjra,idx_train,idx_val,idx_test