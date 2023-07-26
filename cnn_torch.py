#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# In[25]:


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding="same")
        self.conv2=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding="same")
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding="same")
        self.conv4=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding="valid")
        self.conv5=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding="same")
        self.conv6=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding="same")
        self.linear1=nn.Linear(3584, 512)
        self.linear2=nn.Linear(512,4)
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.batch_norm5 = nn.BatchNorm2d(64)
        self.batch_norm_lin = nn.BatchNorm2d(512)
        self.flatten=torch.nn.Flatten()
    
    def forward(self, x):
        #here input will be of shape(N, C_in, H_in, W_in) ~~ (batch_size, #channels, height, width)
        #x shape=(64, 1,32,20)
        x = x.view(-1, 1, 32, 20)
        # input shape= torch.Size([64, 1, 32, 20])
        
        
        #conv layer 1
        x=self.conv1(x)
        x=F.leaky_relu(x)#activation
        # shape= torch.Size([64, 32, 32, 20])
        x=self.dropout(x)
        #shape= torch.Size([64, 32, 32, 20])
        
        
        #conv layer 2
        x2=self.conv2(x)
        x2=self.batch_norm2(x2)
        x2=F.leaky_relu(x2)#activation
        # shape= torch.Size([64, 32, 32, 20])
        
        
        
        
        #concatenation
        x=torch.concat([x,x2],dim=1)
        #shape= torch.Size([64, 64, 32, 20])
        
        
        
        
        #conv layer 3
        x=self.conv3(x)
        #x=self.batch_norm3(x)
        x=F.leaky_relu(x)#activation
        x=self.dropout(x)
        #shape= torch.Size([64, 64, 32, 20])
        x=F.max_pool2d(x,kernel_size=2, stride=2)
        #shape= torch.Size([64, 64, 16, 10])
        
        
        
        #conv layer 4
        x=self.conv4(x)
        x=self.batch_norm4(x)
        x=F.leaky_relu(x)#activation
        x=self.dropout(x)
        #shape= torch.Size([64, 64, 14, 8])
        
        
        
        #conv layer 5
        x2=self.conv5(x)
        x2=self.batch_norm5(x2)
        x2=F.leaky_relu(x2)#activation
        #shape= torch.Size([64, 64, 14, 8])
        
        
        
        #concatenation
        x=torch.concat([x,x2],dim=1)
        #shape= torch.Size([64, 128, 14, 8])
        
        
        
        
        #conv layer 6
        x=self.conv6(x)
        x=F.leaky_relu(x)#activation
        x=self.dropout(x)
        #shape= torch.Size([64, 128, 14, 8])
        x=F.max_pool2d(x,kernel_size=2, stride=2)
        #shape= torch.Size([64, 128, 7, 4])
        
        
        
        x=self.flatten(x)#flattening
        #shape= torch.Size([64, 3584])
        
        
        
        #fully connected layer 1
        x=self.linear1(x)
        x = x.view(-1, x.size(1), 1, 1) # converting 2D tensor to 4D tensor because 
                                        # pytorch batch normalization expect a 4D tensor
        x=self.batch_norm_lin(x)
        #restoring to the previous 2D shape
        x=torch.squeeze(x,-1)
        x=torch.squeeze(x,-1)
        x=F.leaky_relu(x) #activation
        x=self.dropout(x)
        #shape= torch.Size([64, 512])
        
        
        
        
        #fully connected layer 2
        x=self.linear2(x)
        x=F.leaky_relu(x)
        
        return x
        
        


# In[40]:


def calculate_accuracy(outputs, one_hot_labels):
    _, predicted_labels = torch.max(outputs, dim=1)
    one_hot_predicted_labels = torch.nn.functional.one_hot(predicted_labels, num_classes=4).float()
    
    correct_predictions = torch.all(torch.eq(one_hot_predicted_labels, one_hot_labels), dim=1)
    accuracy = torch.sum(correct_predictions).item() / len(correct_predictions)
    return accuracy


# In[46]:


def train_model(model, device, train_loader, test_data, test_labels, num_epochs=10, lr=1e-5):
    criterion=nn.MSELoss().to(device)
    optimizer=optim.Adam(model.parameters(), lr=lr)

    num_epochs=2000
    

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss=0
        for i,(inputs,labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            inputs=inputs.unsqueeze(1)
            # print("input_size=",inputs.size())
            outputs=model(inputs)
            loss=criterion(outputs,torch.nn.functional.one_hot(torch.tensor(labels), 4).squeeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        avg_loss=running_loss/ len(train_loader)
        
        # evaluating after each epoch
        model.eval()
        with torch.no_grad():
            #making a dummy batch of all test data
            test_data=torch.tensor(test_data).unsqueeze(1).to(device)
            test_output=model(test_data)
            one_hot_test_labels=torch.nn.functional.one_hot(torch.tensor(test_labels), 4).squeeze(1).float().to(device)
            test_loss=criterion(test_output, one_hot_test_labels)
            test_accuracy=calculate_accuracy(test_output, one_hot_test_labels)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}")


# In[48]:


def main():
    DIR="/N/u/mdprin/Carbonate/EEG/EEG-DL-master/Models/DatasetAPI/EEG-Motor-Movement-Imagery-Dataset/"

    train_data= pd.read_csv(DIR+"training_data.csv", header=None)
    train_data= np.array(train_data).astype("float32")
    # train_data=train_data.reshape((-1,32,20))
    
    train_labels= pd.read_csv(DIR+"training_labels.csv", header=None)
    train_labels= np.array(train_labels)
    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
    train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    test_data= pd.read_csv(DIR+"test_data.csv", header=None)
    test_data= np.array(test_data).astype("float32")
    # test_data=test_data.reshape((-1,32,20))

    test_labels= pd.read_csv(DIR+"test_labels.csv", header=None)
    test_labels= np.array(test_labels)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=CNNmodel().to(device)
    train_model(model, device,train_loader,test_data, test_labels,num_epochs=100, lr=1e-5)


# In[49]:


if __name__=="__main__":
    main()

