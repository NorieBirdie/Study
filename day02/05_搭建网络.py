import torch
import torch.nn as nn
from torchsummary import summary
class modeldemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(3,3)
        self.linear2=nn.Linear(3,2)
        self.output=nn.Linear(2,2)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)



    def forward(self, x):
        x=torch.sigmoid(self.linear1(x))
        x=torch.relu(self.linear2(x))
        x=torch.softmax(self.output(x),dim=-1)
        return x
def train():
    model=modeldemo()
    print(model)
    data=torch.rand(5,3)
    output=model(data)
    print(output)
    print(f'output shape: {output.shape}')
    print(f'output require_grad: {output.requires_grad}')
    print(f'===查看模型参数===')
    summary(model, input_size=(5,3))
    for name, param in model.named_parameters():
        print(f'参数名: {name}, 参数值: {param.data}, 参数梯度: {param.grad}')
if __name__ == "__main__":
    train()