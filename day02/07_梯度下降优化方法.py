"""
问题：平缓区域权重下降慢，鞍点，局部最小值
动量法momentum优化方法
St=beta*St-1+(1-beta)*gt

"""
import torch
import torch.nn as nn
import torch.optim as optim

def dm01_momentum():
    w=torch.tensor([1.0], requires_grad=True,dtype=torch.float32)
    criterion=((w**2)/2.0)
    optimizer=optim.SGD([w], lr=0.1, momentum=0.9)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    criterion=((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
"""
自适应学习率优化方法
adagrad
St=St-1+gt^2
学习率=学习率/（sqrt(St)+小常数）
小常数=1e-10
缺点：学习率下降过快，导致提前停止学习
"""
def dm02_adagrad():
    w=torch.tensor([1.0], requires_grad=True,dtype=torch.float32)
    criterion=((w**2)/2.0)
    optimizer=optim.Adagrad([w], lr=0.01)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    criterion=((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
"""
自适应学习率优化方法
RMSprop
St=beta*St-1+(1-beta)*gt^2
学习率=学习率/（sqrt(St)+小常数）
小常数=1e-10
优点：解决了adagrad学习率下降过快的问题
"""
def dm03_rmsprop():
    w=torch.tensor([1.0], requires_grad=True,dtype=torch.float32)
    criterion=((w**2)/2.0)
    optimizer=optim.RMSprop([w], lr=0.01, alpha=0.9, eps=1e-10)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    criterion=((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
"""
综合优化方法
Adam
既优化学习率，也优化梯度
mt=beta*Mt-1+(1-beta)*gt    梯度
St=beta*St-1+(1-beta)*gt^2  学习率
"""
def dm04_rmsprop():
    w=torch.tensor([1.0], requires_grad=True,dtype=torch.float32)
    criterion=((w**2)/2.0)
    optimizer=optim.Adam([w],0.01,(0.9,0.999))
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    criterion=((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

if __name__=="__main__":
    
    dm01_momentum()
    dm02_adagrad()
    dm03_rmsprop()
    dm04_rmsprop()


