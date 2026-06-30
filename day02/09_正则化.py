import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""
正则化
L1正则化：L1正则化是指在损失函数中加入权重参数的绝对值之和作为惩罚项，从而鼓励模型产生稀疏的权重参数。
L2正则化：L2正则化是指在损失函数中加入权重参数的平方和作为惩罚项，从而鼓励模型产生较小的权重参数。
Dropout正则化：Dropout正则化是一种在训练过程中随机丢弃神经网络中的一部分神经元的技术，从而减少模型对特定神经元的依赖，降低过拟合的风险。
BN正则化：BN正则化是指在神经网络中使用批量归一化（Batch Normalization）技术，通过对每一层的输入进行归一化处理，从而加速模型的训练过程，并提高模型的泛化能力。


"""
def dm01():
    t1=torch.randint(0,10,(1,4)).float()
    print(t1)
    # 加权求和计算
    linear1=nn.Linear(4,4)
    l1=linear1(t1)
    print(l1)
    #激活函数
    output=torch.relu(l1)
    print(output)
    #随机失活
    dropout=nn.Dropout(0.4)
    output=dropout(output)
    print(output)
def dm02():
    input_2d=torch.randn((1,2,3,4))
    print(input_2d)
    bn2d=nn.BatchNorm2d(num_features=2,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
    output=bn2d(input_2d)
    print(output)
    





if __name__ == "__main__":
    dm02()
