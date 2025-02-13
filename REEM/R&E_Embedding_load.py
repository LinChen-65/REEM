import os
import argparse
import time
import torch
import pandas as pd
import ast
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from Dataset import MLP1stage,BothDataset
from functions import average_pool,test_reg_func

parser = argparse.ArgumentParser(description="REEM trained-GTE &trained-MLP")
parser.add_argument("--city", type=str, required=True, help="city")
# parser.add_argument("--dataset_name", type=str,required=True,help='dataset_file_lastname')
args = parser.parse_args()
city = args.city

train_batch_size=8
val_batch_size=8
test_batch_size=8
num_epochs=200
patience = 10
random_seed = 2

lr_e=1e-6
lr_m=5e-6
weight_decay_e = 5e-4
weight_decay_m = 1e-4


train_pro = 0.4
num_layer=0

reviewMLP_dim = [768,512,256,128,1]
MLPsize = str(reviewMLP_dim)

datasets_file = 'popu+allreview_load' 
popu_columns_name = '0.5' 
review_columns_name = 'review' 
data_used_name = f'popu:{popu_columns_name}+expandtrain+loadget'
time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
test_sample_time = 10
test_sample_proportion = 0.5
device = 'cuda'

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)

current_dir = '/code/model/REEM/'
dataset_dir = '/data/train-dataset/'
pretrain_dir = '/code/REEM'
pretrained_model_dir=os.path.join(pretrain_dir, 'model_pretrain')
model_dir=os.path.join(current_dir, f'trained-emb/{city}')

prefix = "popu:0.5+expandreview_lre5e-06_lrm1e-05_[768, 512, 256, 128, 1]_" #folder name example
all_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
matching_dirs = [d for d in all_dirs if d.startswith(prefix)]
if matching_dirs:
    ckpt_dir = os.path.join(model_dir, matching_dirs[0])
else:
    raise FileNotFoundError(f"No directory found in {model_dir} starting with prefix '{prefix}'")

train_dir = os.path.join(dataset_dir,f'{city}/{city}_popu+allreview_load_traindata.csv')
val_dir = os.path.join(dataset_dir,f'{city}/{city}_popu+allreview_load_valdata.csv')
train_data = BothDataset(train_dir,popu_columns_name,review_columns_name)
val_data = BothDataset(val_dir,popu_columns_name,review_columns_name)
train_loader = DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_data, batch_size=val_batch_size, shuffle=False)

model_emb = AutoModel.from_pretrained(os.path.join(pretrained_model_dir,"thenlper/gte-base"))
tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_model_dir,"thenlper/gte-base"))
model_emb=torch.nn.DataParallel(model_emb) 
model_emb=model_emb.to(device)

for name in model_emb.module.embeddings.state_dict():
    layer=eval('model_emb.module.embeddings.'+name)
    layer.requires_grad=False
for i in range(12-num_layer):
    for name in model_emb.module.encoder.layer[i].state_dict():
        layer=eval(f'model_emb.module.encoder.layer[(i)].'+name)
        layer.requires_grad=False
model_clas = MLP1stage(reviewMLP_dim[0],reviewMLP_dim[1],reviewMLP_dim[2],reviewMLP_dim[3],reviewMLP_dim[4]).to(device)

test_prec_record = list()
test_dir = os.path.join(dataset_dir,f'{city}/{city}_popu+allreview_load_testdata.csv')
test_data = BothDataset(test_dir,popu_columns_name,review_columns_name)
test_loader = DataLoader(dataset=test_data, batch_size=test_batch_size, shuffle=False)

##Load best epoch model param
model_emb.load_state_dict(torch.load(os.path.join(ckpt_dir,f'model_emb_best.pth')))
model_clas.load_state_dict(torch.load(os.path.join(ckpt_dir,f'model_clas_best.pth')))
model_emb.eval()
model_clas.eval()

predicts = []
labels = []
with torch.no_grad(): 
    for batch_idx, (review_list, label, _, placekey) in tqdm(enumerate(test_loader)):
        review_list = tuple(ast.literal_eval(s) for s in review_list)
        len_list = [len(sublist) for sublist in review_list]
        review_list = [item for sublist in review_list for item in sublist]

        token_list = tokenizer(review_list, max_length=512, padding=True, truncation=True, return_tensors='pt').to(
            device)
        outputs_list = model_emb(**token_list)
        embeddings_list_temp = average_pool(outputs_list.last_hidden_state,
                                            token_list['attention_mask'])

        embeddings_list = torch.split(embeddings_list_temp, len_list)
        embeddings_list = [torch.mean(t, dim=0) for t in embeddings_list]
        embeddings_list = torch.stack(embeddings_list)

        predict = model_clas(embeddings_list)
        predict = torch.squeeze(predict, dim=-1)
        predict = predict.cpu()

        predicts.append(predict)
        labels.append(label)


        embeddings_save = embeddings_list.detach().cpu().numpy()
        df = pd.DataFrame(
            {'placekey': placekey, 'embedding': [list(emb) for emb in embeddings_save], 'label': label.numpy()})
        df.to_csv(os.path.join(ckpt_dir, f'embedding_test_result_{batch_idx}.csv'), index=False) #embedding-result/

    full_predict = torch.cat(predicts, dim=0).cpu()
    full_label = torch.cat(labels, dim=0).cpu()

    mse_mean, mse_std, rmse_mean, rmse_std, mae_mean, mae_std, r2_mean, r2_std, evs_mean, evs_std = test_reg_func(
        full_predict, full_label, test_sample_time, test_sample_proportion)
    np.random.seed(random_seed)
    test_prec_record.append(
        [mse_mean, mse_std, rmse_mean, rmse_std, mae_mean, mae_std, r2_mean, r2_std, evs_mean, evs_std])

    print(
        f'Test-MSE=%.4f, Test-RMSE=%.4f, Test-MAE=%.4f,Test-R2=%.4f,Test-EVS=%.4f,' % (
        mse_mean, rmse_mean, mae_mean, r2_mean, evs_mean))



    for batch_idx, (review_list, label,_,placekey) in tqdm(enumerate(train_loader)):
        review_list = tuple(ast.literal_eval(s) for s in review_list)
        len_list = [len(sublist) for sublist in review_list]
        review_list = [item for sublist in review_list for item in sublist]

        token_list = tokenizer(review_list, max_length=512, padding=True, truncation=True, return_tensors='pt').to(
            device)
        outputs_list = model_emb(**token_list)
        embeddings_list_temp = average_pool(outputs_list.last_hidden_state,
                                            token_list['attention_mask'])

        embeddings_list = torch.split(embeddings_list_temp, len_list)
        embeddings_list = [torch.mean(t, dim=0) for t in embeddings_list]
        embeddings_list = torch.stack(embeddings_list)

        predict = model_clas(embeddings_list)
        predict = torch.squeeze(predict, dim=-1)
        predict = predict.cpu()

        embeddings_save = embeddings_list.detach().cpu().numpy()
        df = pd.DataFrame(
            {'placekey': placekey, 'embedding': [list(emb) for emb in embeddings_save], 'label': label.numpy()})
        df.to_csv(os.path.join(ckpt_dir, f'/embedding_result/embedding_train_result_{batch_idx}.csv'), index=False) #embedding-result/



    for batch_idx, (review_list, label, _,placekey) in tqdm(enumerate(val_loader)):
        review_list = tuple(ast.literal_eval(s) for s in review_list)
        len_list = [len(sublist) for sublist in review_list]
        review_list = [item for sublist in review_list for item in sublist]

        token_list = tokenizer(review_list, max_length=512, padding=True, truncation=True, return_tensors='pt').to(
            device)
        outputs_list = model_emb(**token_list)
        embeddings_list_temp = average_pool(outputs_list.last_hidden_state,
                                            token_list['attention_mask'])

        embeddings_list = torch.split(embeddings_list_temp, len_list)
        embeddings_list = [torch.mean(t, dim=0) for t in embeddings_list]
        embeddings_list = torch.stack(embeddings_list)

        predict = model_clas(embeddings_list)
        predict = torch.squeeze(predict, dim=-1)
        predict = predict.cpu()

        embeddings_save = embeddings_list.detach().cpu().numpy()
        df = pd.DataFrame(
            {'placekey': placekey, 'embedding': [list(emb) for emb in embeddings_save], 'label': label.numpy()})
        df.to_csv(os.path.join(ckpt_dir, f'embedding_val_result_{batch_idx}.csv'), index=False) #embedding-result/