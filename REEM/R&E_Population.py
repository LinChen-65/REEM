import os
import time
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from Dataset import *
from functions import test_reg_func

parser = argparse.ArgumentParser(description="REEM trained-popu&trained-fuse1")
parser.add_argument("--city", type=str, required=True, help="city")

args = parser.parse_args()
city = args.city

train_batch_size=64
val_batch_size=64
test_batch_size=64
num_epochs=200
random_seed=2023


lr_pop=3e-3
weight_decay_pop = 1e-5


patience = 10
pop_dim = 5
pop_hidden_layers = [100,30,10]


popu_columns_name = '0.5'
MLPsize = '[10,100,30;768,512,256,128; 30+128,30,1]'
time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
test_sample_time = 5
test_sample_proportion = 0.5
datasets_file = f'allembedding&popu'
data_used_name = f'allembedding&popu{popu_columns_name}'

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)


current_dir = 'code/model/REEM/'
dataset_dir = 'data/train-dataset/'

model_dir = os.path.join(current_dir, f'trained-popu/{city}')
ckpt_dir=os.path.join(model_dir,f'{data_used_name}_lrp{lr_pop}__{MLPsize}_{time_data}')
os.makedirs(ckpt_dir,exist_ok=True)
config_dict={'patience':patience,'test_sample_time': test_sample_time,'test_sample_proportion':test_sample_proportion,'train_batch_size':train_batch_size,'pop_hidden_layers':pop_hidden_layers,'emb_hidden_layers':emb_hidden_layers,'fuse_hidden_layers':fuse_hidden_layers,
             'val_batch_size':val_batch_size,'test_batch_size':test_batch_size,'num_epochs':num_epochs,'lr_pop':lr_pop,
             'data_used_name':data_used_name,'weight_decay_pop':weight_decay_pop,
             'MLPsize':MLPsize,'time_data':time_data,'random_seed':random_seed,'datasets_file':datasets_file}
with open(os.path.join(ckpt_dir,'config.txt'),'w') as f:
    f.write(str(config_dict))


train_dir = os.path.join(dataset_dir,f'{city}/{city}_{datasets_file}_traindata.csv')
val_dir = os.path.join(dataset_dir,f'{city}/{city}_{datasets_file}_valdata.csv')
test_dir = os.path.join(dataset_dir,f'{city}/{city}_{datasets_file}_testdata.csv')
train_data = EmbPoporiDataset(train_dir,popu_columns_name)
val_data = EmbPoporiDataset(val_dir,popu_columns_name)
test_data = EmbPoporiDataset(test_dir,popu_columns_name)
train_loader = DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=val_batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=test_batch_size, shuffle=False)

criterion = nn.MSELoss()
pop_model = MLP(pop_dim, pop_hidden_layers,1).to(device)
optimizer_pop = torch.optim.Adam(pop_model.parameters(), lr=lr_pop,weight_decay=weight_decay_pop) #weight_decay=weight_decay,betas=betas

best_val_loss=float('inf')
train_loss_record=list()
val_loss_record = list()
test_prec_record = list()
for i in range(num_epochs):
    pop_model.train()
    train_loss=0.0
    for batch_idx, (label,input,embedd) in tqdm(enumerate(train_loader)):
        predict = pop_model(input.to(device))
        predict = torch.squeeze(predict, dim=-1)
        loss=criterion(predict,label.to(device)) 

        total_loss = loss #+ l1_regularization * L1_lam

        optimizer_pop.zero_grad()
        total_loss.backward()
        optimizer_pop.step()
        train_loss += loss.cpu().item()*train_batch_size

    train_loss_epoch=train_loss/len(train_data)
    train_loss_record.append(train_loss_epoch)
    print('Epoch %d, Train-Loss(MSE)=%.3f'%(i,train_loss_epoch))

    pop_model.eval()
    val_loss=0.0
    with torch.no_grad():
        for batch_idx, (label, input,embedd) in tqdm(enumerate(val_loader)):
            predict = pop_model(input.to(device))
            predict = torch.squeeze(predict, dim=-1)
            loss = criterion(predict, label.to(device))  # 返回一个tensor
            val_loss += loss.cpu().item() * val_batch_size

        val_loss_epoch = val_loss / len(val_data)
        val_loss_record.append(val_loss_epoch)
        print('Epoch %d, Val-Loss(MSE)=%.3f'%(i,val_loss_epoch))
        if round(val_loss_epoch,6) < round(best_val_loss,6):
            best_val_loss = val_loss_epoch
            no_improvement_count = 0
            torch.save(pop_model.state_dict(), os.path.join(ckpt_dir, f'model_pop_best.pth'))
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping triggered.")
                break


np.save(os.path.join(ckpt_dir,'pop_train_loss_record.npy'),np.array(train_loss_record))
np.save(os.path.join(ckpt_dir,'pop_val_loss_record.npy'),np.array(val_loss_record))

best_epoch  = val_loss_record.index(min(val_loss_record))
print('popu_best_epoch:',best_epoch)

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
plt.savefig(os.path.join(ckpt_dir,'pop_plot_loss.png'))
print("Training completed.")


#============
# test
predicts = []
labels = []
test_prec_record = list()
pop_model.load_state_dict(torch.load(os.path.join(ckpt_dir,f'model_pop_best.pth')))
pop_model.eval()
with torch.no_grad():
    for batch_idx, (label, input, embedd) in tqdm(enumerate(test_loader)):
        predict = pop_model(input.to(device))
        predict = torch.squeeze(predict, dim=-1)

        predict = predict.cpu()
        predicts.append(predict)
        labels.append(label)

    full_predict = torch.cat(predicts, dim=0)
    full_label = torch.cat(labels, dim=0)
    mse_mean, mse_std, rmse_mean, rmse_std, mae_mean, mae_std, r2_mean, r2_std, evs_mean, evs_std = test_reg_func(
        full_predict, full_label, test_sample_time, test_sample_proportion)
    np.random.seed(random_seed)

    test_prec_record.append(
        [mse_mean, mse_std, rmse_mean, rmse_std, mae_mean, mae_std, r2_mean, r2_std, evs_mean, evs_std])

    print(
        f'Test-MSE=%.4f, Test-RMSE=%.4f, Test-MAE=%.4f,Test-R2=%.4f,Test-EVS=%.4f,' % (
        mse_mean, rmse_mean, mae_mean, r2_mean, evs_mean))


result_dict = {'best-epoch': best_epoch,'report_result':test_prec_record[0]}
with open(os.path.join(ckpt_dir, 'pop_result.txt'), 'w') as f:
    f.write(str(result_dict))
