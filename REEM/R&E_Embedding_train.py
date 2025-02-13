import os
import argparse
import time
import torch
import torch.nn as nn
import ast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader,Subset
from transformers import AutoTokenizer, AutoModel
from Dataset import MLP1stage,BothDataset
from functions import average_pool,test_reg_func
def random_subset(dataset, percentage):
    indices = torch.randperm(len(dataset)).tolist()[:int(len(dataset) * percentage)]
    return Subset(dataset, indices)

parser = argparse.ArgumentParser(description="REEM trained-GTE&trained-EmbAdapter")
parser.add_argument("--city", type=str, required=True, help="city")
# parser.add_argument("--dataset_name", type=str,required=True,help='dataset_file_lastname')
args = parser.parse_args()
city = args.city


train_batch_size=64
val_batch_size=64
test_batch_size=4
num_epochs=200
patience = 10
random_seed=2

lr_e=5e-6
lr_m=1e-5
weight_decay_e = 5e-4
weight_decay_m = 1e-4

train_pro = 0.4
num_layer=2
reviewMLP_dim = [768,512,256,128,1]
MLPsize = str(reviewMLP_dim)

datasets_file = 'popu+allreview'
popu_columns_name = '0.5' 
review_columns_name = 'review'
data_used_name = f'popu:{popu_columns_name}+expandreview'
time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
test_sample_time = 10
test_sample_proportion = 0.5
device = 'cuda'

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)

# current_dir = os.getcwd() #ATTENTION
current_dir = './code/model/REEM/'
dataset_dir = './data/train-dataset/'
pretrain_dir = './code/REEM/'
pretrained_model_dir=os.path.join(pretrain_dir, 'model_pretrain')
model_dir=os.path.join(current_dir, f'trained-emb/{city}')
ckpt_dir=os.path.join(model_dir,f'{data_used_name}_lre{lr_e}_lrm{lr_m}_{MLPsize}_{time_data}')
os.makedirs(ckpt_dir,exist_ok=True)
config_dict={'test_sample_time': test_sample_time,'test_sample_proportion':test_sample_proportion,'train_batch_size':train_batch_size,
             'val_batch_size':val_batch_size,'test_batch_size':test_batch_size,'num_epochs':num_epochs,'lr_e':lr_e,'lr_m':lr_m,'num_layer':num_layer,
             'data_used_name':data_used_name,'datasets_file':datasets_file,'weight_decay_e':weight_decay_e,'weight_decay_m':weight_decay_m,'random_seed':random_seed,'train_proportion':train_pro,
             'MLPsize':MLPsize,'time_data':time_data}
with open(os.path.join(ckpt_dir,'config.txt'),'w') as f:
    f.write(str(config_dict))


train_dir = os.path.join(dataset_dir,f'{city}/{city}_popu+allreview_traindata.csv')
val_dir = os.path.join(dataset_dir,f'{city}/{city}_popu+allreview_valdata.csv')
train_data = BothDataset(train_dir,popu_columns_name,review_columns_name)
val_data = BothDataset(val_dir,popu_columns_name,review_columns_name)
train_loader = DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True) 
val_loader = DataLoader(dataset=val_data, batch_size=val_batch_size, shuffle=True)


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
criterion = nn.MSELoss()
optimizer = torch.optim.Adam([
    {'params': model_emb.module.encoder.layer[-num_layer:].parameters(), 'lr': lr_e,'weight_decay':weight_decay_e},
    {'params': model_clas.parameters(), 'lr': lr_m,'weight_decay':weight_decay_m}
])

best_val_loss=float('inf')
train_loss_record = list()
val_loss_record = list()

for i in range(num_epochs):
    subset = random_subset(train_data, 0.4)
    epoch_train_loader = DataLoader(dataset=subset, batch_size=train_batch_size, shuffle=True)
    model_emb.train()
    model_clas.train()
    train_loss=0.0

    for batch_idx, (review_list, label,_,placekey) in tqdm(enumerate(epoch_train_loader)):
        token_list = tokenizer(review_list, max_length=512, padding=True, truncation=True, return_tensors='pt').to(
            device)
        outputs_list = model_emb(**token_list)
        embeddings_list = average_pool(outputs_list.last_hidden_state, token_list['attention_mask'])  # (batch_size*768)
        predict = model_clas(embeddings_list)
        predict = torch.squeeze(predict, dim=-1)
        loss=criterion(predict,label.to(device))  


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().item()*train_batch_size

    train_loss_epoch=train_loss/len(train_data)
    train_loss_record.append(train_loss_epoch)
    print('Epoch %d, Train-Loss(MSE)=%.3f'%(i,train_loss_epoch))



    model_emb.eval()
    model_clas.eval()
    val_loss=0.0

    with torch.no_grad():
        for batch_idx, (review_list, label, _,placekey) in tqdm(enumerate(val_loader)):

            token_list = tokenizer(review_list, max_length=512, padding=True, truncation=True, return_tensors='pt').to(
                device)
            outputs_list = model_emb(**token_list)
            embeddings_list = average_pool(outputs_list.last_hidden_state,
                                                token_list['attention_mask'])

            predict = model_clas(embeddings_list)
            predict = torch.squeeze(predict, dim=-1)
            loss = criterion(predict, label.to(device))

    
            val_loss += loss.cpu().item() * val_batch_size

      

        val_loss_epoch = val_loss / len(val_data)
        val_loss_record.append(val_loss_epoch)
        print('Epoch %d, Val-Loss(MSE)=%.3f'%(i,val_loss_epoch))




        #early stopping
        if round(val_loss_epoch,6) < round(best_val_loss,6):
            best_val_loss = val_loss_epoch
            torch.save(model_emb.state_dict(), os.path.join(ckpt_dir, f'model_emb_best.pth'))
            torch.save(model_clas.state_dict(), os.path.join(ckpt_dir, f'model_clas_best.pth'))
            no_improvement_count = 0  

        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping triggered.")
                break

np.save(os.path.join(ckpt_dir,'train_loss_record.npy'),np.array(train_loss_record))
np.save(os.path.join(ckpt_dir,'val_loss_record.npy'),np.array(val_loss_record))

best_epoch  = val_loss_record.index(min(val_loss_record))
print('best_epoch:',best_epoch)

plt.figure(figsize=(12, 8))
plt.plot(train_loss_record, label='Train Loss')
plt.plot(val_loss_record, label='val Loss')

min_train_loss = train_loss_record[best_epoch]
min_val_loss = val_loss_record[best_epoch]
plt.scatter(train_loss_record.index(min_train_loss), min_train_loss, color='red')
plt.scatter(val_loss_record.index(min_val_loss), min_val_loss, color='red')
plt.text(train_loss_record.index(min_train_loss), min_train_loss, f'{min_train_loss:.4f}', fontsize=12,
         color='red')
plt.text(val_loss_record.index(min_val_loss), min_val_loss, f'{min_val_loss:.4f}', fontsize=12, color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Val Loss over Epochs')
plt.legend()
plt.savefig(os.path.join(ckpt_dir,'plot_loss.png'))
print("Training completed.")

#============================
# test
test_prec_record = list()
test_dir = os.path.join(dataset_dir,f'{city}/{city}_popu+allreview_testdata.csv')
test_data = BothDataset(test_dir,popu_columns_name,review_columns_name)

test_loader = DataLoader(dataset=test_data, batch_size=test_batch_size, shuffle=False)
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


#mse_mean, mse_std,rmse_mean, rmse_std, mae_mean, mae_std, r2_mean, r2_std, evs_mean, evs_std
result_dict = {'best-epoch': best_epoch,'report_result':test_prec_record[0]}
with open(os.path.join(ckpt_dir, 'result.txt'), 'w') as f:
    f.write(str(result_dict))

