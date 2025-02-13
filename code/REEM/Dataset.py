import pandas as pd
from torch.utils import data

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLP1stage(nn.Module):
    def __init__(self, input_dim,hidden_dim1,hidden_dim2,hidden_dim3,output_dim1): #768,512,256,128,1/#45,256,128,64,1
        super(MLP1stage, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc5 = nn.Linear(hidden_dim3, output_dim1)
    def forward(self, A):
        x = F.relu(self.fc1(A))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc5(x)
        return out,x

class ForzenEmbDataset(data.Dataset):
    def __init__(self, csv_dir, popu_column_name,embedding_column_name):
        POI_df = pd.read_csv(csv_dir)
        POI_df['review_embedding'] = POI_df['review_embedding']
        self.len = POI_df.shape[0]
        self.embedding_series = POI_df[embedding_column_name].apply(lambda x: np.fromstring(x[1:-1], sep=" ")) #array
        self.target_series = POI_df['racial_segregation_index'].astype(np.float32)
        self.popu_series = POI_df[popu_column_name].apply(lambda x:np.fromstring(x[1:-1],sep=' ')) #array
        self.placekey_series = POI_df['placekey']
    def __getitem__(self, index):
        review_emb = self.embedding_series[index].astype(np.float32)
        target = self.target_series[index].astype(np.float32)
        popu=self.popu_series[index].astype(np.float32)
        placekey = self.placekey_series[index]
        return review_emb, target,popu,placekey
    def __len__(self):
        return self.len
class BothDataset(data.Dataset):
    def __init__(self, csv_dir, popu_column_name,review_column_name):
        POI_df = pd.read_csv(csv_dir)
        self.len = POI_df.shape[0]
        self.review_series = POI_df[review_column_name].apply(lambda x:str(x))
        self.target_series = POI_df['racial_segregation_index'].astype(np.float32)
        self.popu_series = POI_df[popu_column_name].apply(lambda x:np.fromstring(x[1:-1],sep=' '))
        self.placekey_series = POI_df['placekey']

    def __getitem__(self, index):
        review = self.review_series[index]
        target = self.target_series[index].astype(np.float32)
        popu=self.popu_series[index].astype(np.float32)
        placekey = self.placekey_series[index]
        return review, target,popu,placekey

    def __len__(self):
        return self.len

class EmbPoporiDataset(data.Dataset): #存档复现stage2用
    def __init__(self, csv_dir, popu_column_name):
        POI_df = pd.read_csv(csv_dir)
        POI_df.rename(columns={'label':'racial_segregation_index'},inplace=True)
        self.len = POI_df.shape[0]
        self.target_series = POI_df['racial_segregation_index'].astype(np.float32)
        self.popu_series = POI_df[popu_column_name].apply(lambda x:np.fromstring(x[1:-1],sep=' '))
        self.placekey_series = POI_df['placekey']
        self.embedding_series = POI_df['embedding'].apply(lambda x:np.array(eval(x)))
        # self.rating_series = POI_df['ratings'].apply(lambda x:np.array(eval(x)))

    def __getitem__(self, index):
        target = self.target_series[index].astype(np.float32)
        popu=self.popu_series[index].astype(np.float32)
        embedd = self.embedding_series[index].astype(np.float32)
        # ratings = self.rating_series[index].astype(np.float32)
        # placekey =self.placekey_series[index]

        return target,popu,embedd#,ratings,placekey

    def __len__(self):
        return self.len


class EmbPopDataset(data.Dataset): #stage3
    def __init__(self, csv_dir, popu_column_name):
        POI_df = pd.read_csv(csv_dir)
        # POI_df.rename(columns={'label':'racial_segregation_index'},inplace=True)
        self.len = POI_df.shape[0]
        self.target_series = POI_df['racial_segregation_index'].astype(np.float32)
        self.popu_series = POI_df[popu_column_name].apply(lambda x:np.fromstring(x[1:-1],sep=' '))
        self.placekey_series = POI_df['placekey']
        self.embedding_series = POI_df['embedding'].apply(lambda x:np.array(eval(x)))
        self.rating_series = POI_df['rating'].apply(lambda x:np.array(eval(x)))

    def __getitem__(self, index):
        target = self.target_series[index].astype(np.float32)
        popu=self.popu_series[index].astype(np.float32)
        embedd = self.embedding_series[index].astype(np.float32)
        ratings = self.rating_series[index].astype(np.float32)
        placekey =self.placekey_series[index]

        return target,popu,embedd,ratings,placekey

    def __len__(self):
        return self.len



class MLP(nn.Module):
    def __init__(self, input_dim,hidden_layers,output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_layers[0])
        for i,h in enumerate(hidden_layers[1:],1):
            setattr(self, 'fc'+str(i+1), nn.Linear( hidden_layers[i-1], hidden_layers[i] ))
        self.fc_out = nn.Linear(hidden_layers[-1], output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        for i in range(1,len(self.hidden_layers)-1):
            x = F.relu( getattr(self, 'fc'+str(i+1))(x) ) #根据变量名字字符串获取对象属性
        x1 = F.relu(getattr(self, 'fc'+str(len(self.hidden_layers)))(x))
        x1 = self.fc_out(x1)
        return x1,x
