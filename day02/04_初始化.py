import torch.nn as nn
import torch

def dm01():
    linear=nn.Linear(5,3)
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)

def dm02():
    linear=nn.Linear(5,3)
    nn.init.kaiming_normal_(linear.weight)
    
    print(linear.weight.data)
   


if __name__=="__main__":
    dm01()
    dm02()