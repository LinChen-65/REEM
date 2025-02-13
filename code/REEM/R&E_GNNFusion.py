import time
import argparse
import matplotlib.pyplot as plt
from Dataset import *
from GATutils import *
from functions import test_reg_func


parser = argparse.ArgumentParser(description="REEM trained-GNN")
parser.add_argument("--city", type=str, required=True, help="city")

args = parser.parse_args()
city = args.city

num_epochs=200
random_seed=42

lr_gat = 5e-4
weight_decay_gat = 1e-4
lr_fc = 5e-4
weight_decay_fc = 5e-5


patience =20
emb_dim = 768
pop_dim = 5
rate_dim = 45
pop_hidden_layers = [100,30,10]
emb_hidden_layers = [512,256,128]
rat_hidden_layers = [512,128,64]
output_dim = 1

datasets_file = 'GNNGraph'
popu_columns_name = '0.5' 
MLPsize = '-2,-1,-1;30,128,128;512,128,64,1'
time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
test_sample_time = 5
test_sample_proportion = 0.5
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)

current_dir = '/code/model/REEM/'
dataset_dir = '/data/train-dataset/'
load_model_dir=os.path.join(current_dir, f'best-model/{city}/')
model_dir = os.path.join(current_dir, f'trained-Fusion/{city}')
ckpt_dir=os.path.join(model_dir,f'{datasets_file}_lr{lr_gat}{lr_fc}_{MLPsize}_{time_data}')
os.makedirs(ckpt_dir,exist_ok=True)
config_dict={'patience':patience,'test_sample_time': test_sample_time,'test_sample_proportion':test_sample_proportion,
             'lr_gat':lr_gat,'weight_decay_gat':weight_decay_gat, 'lr_fc':lr_fc,'weight_decay_fc':weight_decay_fc,
             'num_epochs':num_epochs, 'MLPsize':MLPsize,'time_data':time_data,'random_seed':random_seed,'datasets_file':datasets_file}
with open(os.path.join(ckpt_dir,'config.txt'),'w') as f:
    f.write(str(config_dict))

pop_model = MLP(pop_dim, pop_hidden_layers,1).to(device)
pop_model.load_state_dict(torch.load(os.path.join(load_model_dir,f'model_pop_best.pth'),map_location=device,weights_only=True))
emb_model = MLP1stage(emb_dim, emb_hidden_layers[0],emb_hidden_layers[1],emb_hidden_layers[2],1).to(device)
emb_model.load_state_dict(torch.load(os.path.join(load_model_dir,f'model_clas_best.pth'),map_location=device,weights_only=True))
rate_model = MLP1stage(rate_dim,rat_hidden_layers[0],rat_hidden_layers[1],rat_hidden_layers[2],1).to(device)
rate_model.load_state_dict(torch.load(os.path.join(load_model_dir,f'model_best_rating.pth'),map_location=device,weights_only=True))

##GAT层都已注释
PoGAT = GAT(nfeat=pop_hidden_layers[-2], #30 #pop_dim,#5
                nhid=pop_hidden_layers[1], #30 #int(pop_hidden_layers[-2]/2),#15
                dropout=0.2,
                nheads=1,
                alpha=0.2).to(device)
ReGAT = GAT(nfeat=emb_hidden_layers[-1], #30 #pop_dim,#5
                nhid=emb_hidden_layers[1], #30 #int(pop_hidden_layers[-2]/2),#15
                dropout=0.2,
                nheads=1,
                alpha=0.2).to(device)
RaGAT = GAT(nfeat=rat_hidden_layers[-1], #64 #rate_dim,#45
                nhid=rat_hidden_layers[1],#128#int(rat_hidden_layers[-1]/2), #32
                dropout=0.2,
                nheads=1,
                alpha=0.2).to(device)

seq_len = PoGAT.out_att.out_features+ReGAT.out_att.out_features+RaGAT.out_att.out_features
heads = 1
Allmodel = MultiGAT(PoGAT,ReGAT,RaGAT,seq_len,heads,output_dim=1,hidden_dim=32).to(device)

gat_params = [Allmodel.gat1.parameters(),#Allmodel.attention_layer.parameters(),
              Allmodel.gat3.parameters(), Allmodel.gat2.parameters()]
fc_params = [Allmodel.fc2.parameters(),
             Allmodel.fc3.parameters(),Allmodel.fc4.parameters(), Allmodel.out.parameters()]
optimizer = torch.optim.Adam([
    {'params': param, 'lr': lr_gat, 'weight_decay': weight_decay_gat} for param in gat_params
] + [
    {'params': param, 'lr': lr_fc,'weight_decay': weight_decay_fc} for param in fc_params
])
criterion = nn.MSELoss()


Neigh_num = 5
labels,ori_popu,adjpop,ori_review,adjre,ori_rating,adjra,idx_train,idx_val,idx_test= load_data(dataset_dir,city,datasets_file,device,Neigh_num)

best_val_loss = float('inf')
train_loss_record = []
val_loss_record = []
for epoch in range(num_epochs):
    Allmodel.train()

    _, popu = pop_model(ori_popu)
    _, review = emb_model(ori_review)
    _, rating = rate_model(ori_rating)
    output = Allmodel(popu, adjpop, review, adjre, rating, adjra)

    loss_train = criterion(output[idx_train].squeeze(),labels[idx_train])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    train_loss_record.append(loss_train.item())
    print(f'Epoch {epoch}, Train-Loss(MSE)={loss_train.item():.3f}')


    Allmodel.eval()
    with torch.no_grad():
        _, popu = pop_model(ori_popu)
        _, review = emb_model(ori_review)
        _, rating = rate_model(ori_rating)
        output = Allmodel(popu, adjpop, review, adjre, rating, adjra)
        loss_val = criterion(output[idx_val].squeeze(), labels[idx_val])

        val_loss_record.append(loss_val.item())
        print('Epoch %d, Val-Loss(MSE)=%.3f' % (epoch, loss_val))

        if round(loss_val.item(),6) < round(best_val_loss,6):
            best_val_loss = loss_val.item()
            no_improvement_count = 0  
            torch.save(Allmodel.state_dict(), os.path.join(ckpt_dir, f'model_GAT_best.pth'))

        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping triggered.")
                break


np.save(os.path.join(ckpt_dir, 'GAT_train_loss_record.npy'), np.array(train_loss_record))
np.save(os.path.join(ckpt_dir, 'GAT_val_loss_record.npy'), np.array(val_loss_record))


best_epoch = val_loss_record.index(min(val_loss_record))
print('best_epoch:',best_epoch)
plt.figure(figsize=(12, 8))
plt.plot(train_loss_record, label='Train Loss')
plt.plot(val_loss_record, label='Val Loss')
min_train_loss = train_loss_record[best_epoch]
min_val_loss = val_loss_record[best_epoch]
plt.scatter(train_loss_record.index(min_train_loss), min_train_loss, color='red')
plt.scatter(val_loss_record.index(min_val_loss), min_val_loss, color='red')
plt.text(train_loss_record.index(min_train_loss), min_train_loss, f'{min_train_loss:.4f}', fontsize=12, color='red')
plt.text(val_loss_record.index(min_val_loss), min_val_loss, f'{min_val_loss:.4f}', fontsize=12, color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Val Loss over Epochs')
plt.legend()
plt.ylim(0.0, 0.02)
plt.savefig(os.path.join(ckpt_dir, 'stage3_plot_loss.png'))
print("Training completed.")


# Test set evaluation
Allmodel.load_state_dict(torch.load(os.path.join(ckpt_dir, f'model_GAT_best.pth'),weights_only=True))
Allmodel.eval()
test_prec_record = list()
with torch.no_grad():
    popu_predict, popu = pop_model(ori_popu)
    _, review = emb_model(ori_review)
    _, rating = rate_model(ori_rating)
    output = Allmodel(popu, adjpop, review, adjre, rating, adjra)
    loss_test = criterion(output[idx_test].squeeze(), labels[idx_test])
    full_predict = output[idx_test].cpu()
    full_label = labels[idx_test].cpu()

    mse_mean, mse_std, rmse_mean, rmse_std, mae_mean, mae_std, r2_mean, r2_std, evs_mean, evs_std = test_reg_func(
        full_predict, full_label, test_sample_time, test_sample_proportion)
    np.random.seed(random_seed)
    test_prec_record.append(
        [mse_mean, mse_std, rmse_mean, rmse_std, mae_mean, mae_std, r2_mean, r2_std, evs_mean, evs_std])
    print(f'Test-MSE=%.4f, Test-RMSE=%.4f, Test-MAE=%.4f, Test-R2=%.4f, Test-EVS=%.4f' % (
    mse_mean, rmse_mean, mae_mean, r2_mean, evs_mean))

result_dict = {
      'best-epoch': best_epoch,
    'report_result': test_prec_record[0]
}
with open(os.path.join(ckpt_dir, 'GAT_result.txt'), 'w') as f:
    f.write(str(result_dict))
