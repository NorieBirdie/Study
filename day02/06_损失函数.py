"""
多分类交叉熵损失
Loss=-sigmalog(s(fx))
损失函数结果=正确概率类别的对数最小化
CrossEntropyLoss=sofemax()+损失计算
"""
import torch
import torch.nn as nn

def dm01():
    y_ture=torch.tensor([[0,1,0],[1,0,0]],dtype=torch.float32)
    y_pred=torch.tensor([[0.05,0.95,0],[0.1,0.8,0.1]],dtype=torch.float32,requires_grad=True)
    criterion=nn.CrossEntropyLoss()
    loss=criterion(y_pred,y_ture)
    print(f'loss:{loss}')

"""
二分类
Loss=-ylog(s(fx))-(1-y)log(1-s(fx))
没有包含sigmoid激活，所以还需要手动sigmoid
BCEloss
"""
def dm02():
    y_ture=torch.tensor([0,1,0],dtype=torch.float32)
    y_pred=torch.tensor([0.1,0.8,0.2],dtype=torch.float32,requires_grad=True)
    criterion=nn.BCELoss()
    loss=criterion(y_pred,y_ture)
    print(f'loss:{loss}')
"""
MAE，SmoothL1Loss，MSELoss，SmoothL1Loss
三个回归损失函数

"""
def dm03():
    y_ture=torch.tensor([[0.1,0.4,0.9],[0.4,0.5,0.6]],dtype=torch.float32)
    y_pred=torch.tensor([[0.1,0.2,0.3],[0.4,0.5,0.6]],dtype=torch.float32,requires_grad=True)
    criterion=nn.SmoothL1Loss()
    loss=criterion(y_pred,y_ture)
    print(f'loss:{loss}')
if __name__ == "__main__":
    dm01()
    dm02()
    dm03()