import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
from torchsummary import summary
from torch.utils.data import TensorDataset,DataLoader
def create_dataset():
    train_data=pd.read_csv("day03/train.csv")
    test_data=pd.read_csv("day03/test.csv")

    train_x,train_y=train_data.iloc[:,:-1],train_data.iloc[:,-1]
    test_x=test_data.iloc[:,1:]


    train_x,val_x,train_y,val_y=train_test_split(
        train_x,
        train_y,
        test_size=0.2,
        random_state=3,
        stratify=train_y
    )

    train_x=train_x.astype(np.float32).values
    train_y=train_y.astype(np.int64).values
    val_x=val_x.astype(np.float32).values
    val_y=val_y.astype(np.int64).values
    test_x=test_x.astype(np.float32).values

    train_x=torch.from_numpy(train_x)
    train_y=torch.from_numpy(train_y)
    val_x=torch.from_numpy(val_x)
    val_y=torch.from_numpy(val_y)
    test_x=torch.from_numpy(test_x)

    train_dataset=TensorDataset(train_x,train_y)
    val_dataset=TensorDataset(val_x,val_y)
    test_dataset=TensorDataset(test_x)

    return train_dataset,val_dataset,test_dataset,train_x.shape[1],len(np.unique(train_y))
class PhonePriceModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear1=nn.Linear(input_dim,128)
        self.linear2=nn.Linear(128,256)
        self.linear3=nn.Linear(256,output_dim)
    def forward(self,x):
        x=self.linear1(x)
        x=torch.relu(x)
        x=self.linear2(x)
        x=torch.relu(x)
        x=self.linear3(x)
        return x
def train(train_dataset,val_dataset,input_dim,output_dim):
    train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    model=PhonePriceModel(input_dim,output_dim)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.01)
    epochs=50
    for epoch in range(epochs):
        total_loss=0.0
        batch_sum=0
        start=time.time()
        for x,y in train_loader:
            model.train()
            y_pred=model(x)
            loss=criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            batch_sum+=1
        end=time.time()
        print(f"Epoch:{epoch+1}/{epochs},Loss:{total_loss/batch_sum:.4f},Time:{end-start:.2f}s")
    torch.save(model.state_dict(),"day03/phone_price_model.pth")
    # print(f"""模型结构:\n{model.state_dict()}""")
    print("模型训练完成并保存为phone_price_model.pth")

def evaluate(val_dataset,input_dim,output_dim):
    model=PhonePriceModel(input_dim,output_dim)
    model.load_state_dict(torch.load("day03/phone_price_model.pth",weights_only=True))
    val_loader=DataLoader(val_dataset,batch_size=16,shuffle=False)
    model.eval()
    correct=0
    with torch.no_grad():
        for x,y in val_loader:
            y_pred=model(x)
            y_pred=torch.argmax(y_pred,dim=1)
            print(f"预测结果:{y_pred},真实结果:{y}")
            correct+=(y_pred==y).sum().item()
    print(f"验证集准确率:{correct/len(val_dataset):.4f}")


if __name__ == "__main__":
    train_dataset,val_dataset,test_dataset,input_dim,output_dim=create_dataset()
    # print(f'训练集对象:{train_dataset}')
    # print(f'验证集对象:{val_dataset}')
    # print(f'测试集对象:{test_dataset}')
    # print(f'输入维度:{input_dim}')
    # print(f'输出维度:{output_dim}')
    train(train_dataset,val_dataset,input_dim,output_dim)
    evaluate(val_dataset,input_dim,output_dim)
