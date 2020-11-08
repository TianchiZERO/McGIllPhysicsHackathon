import torch.nn as nn
import torch
import SRCNNModel
from torch.utils.data import Dataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
#Parameters
path_train_dataset=''
path_trained_model = ''
lr_layer1_2 = 1e-4
lr_layer3 = 1e-5 
total_epoch = 100
batch_size = 100
save_every_x_epochs = 10

# Define class Train_dataset here in order to use torch.utils.data.dataloader later
class Train_dataset(Dataset):
    def __init__(self, train_dateset_h5_file):
        super(Train_dataset, self).__init__()
        self.train_dateset_h5_file = train_dateset_h5_file

    def __getitem__(self, index):
        with h5py.File(self.train_dateset_h5_file, 'r') as f:
            return ((f['input'][index]/255.)), ((f['output'][index]/255.))
        
    def __len__(self):
        with h5py.File(self.train_dateset_h5_file, 'r') as f:
            return (f['input']).shape[0]


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
model = SRCNNModel.SRCNN().to(device)
#model = torch.load(path_trained_model+'epoch_{}'.format(0)+'.pt')
criterion = nn.MSELoss()
optimizer = torch.optim.SGD([{'params': model.conv1.parameters()},
                       {'params': model.conv2.parameters()},
                       {'params': model.conv3.parameters(),'lr':lr_layer3}], 
                      lr=lr_layer1_2, momentum=0.9)

mean_loss = []
for epoch in range(total_epoch):
    train_dataloader = torch.utils.data.DataLoader(dataset=Train_dataset(path_train_dataset),batch_size=batch_size,shuffle=True,drop_last=True,num_workers=0)
    total_loss_for_each_epoch = 0
    model.train()
    count = 0
    for train_data in train_dataloader:
        train_input,train_output = train_data
        train_input = train_input.to(device)
        train_output = train_output.to(device)
        predict = model.forward(train_input)
        loss = criterion(predict,train_output)
        total_loss_for_each_epoch += loss.item()*train_input.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += train_input.shape[0]
    print(epoch,total_loss_for_each_epoch/count)
    mean_loss.append(total_loss_for_each_epoch/count)
    if epoch % save_every_x_epochs == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mean_loss,
            }, path_trained_model+'/epoch'+str(epoch)+'_model.pt')    



plt.figure()
plt.plot(mean_loss)
plt.title("Mean Loss over Epochs")
plt.ylabel("Mean Loss")
plt.xlabel("Epoch")
plt.savefig("mean_loss.png")
#print(mean_loss)