"""
张量与numpy数组的转换
涉及到的api:
张量转numpy
numpy() 共享内存
from_numpy() 
numpy.copy()不共享内存
numpy转张量
from_numpy() 共享内存
torch.tensor() 不共享内存
标量张量提取内容
item() 共享内存

"""
import torch
import numpy as np
#定义函数
def dm01():
        t1 =torch.tensor([1,2,3,4,5])
       # n1 = t1.numpy()
        n1=t1.numpy().copy() #不共享内存
        print(f"张量t1: {t1}", f"类型: {type(t1)}")
        print(f"numpy数组n1: {n1}", f"类型: {type(n1)}")
        n1[0]=100 #不共享内存
        print(f"张量t1: {t1}", f"类型: {type(t1)}")
        print(f"numpy数组n1: {n1}", f"类型: {type(n1)}")
        print('-'*30)
def dm02():
    n2 =np.array([11,22,33,44,55])
    print(f"numpy数组n2: {n2}", f"类型: {type(n2)}")
    t2 = torch.from_numpy(n2) #共享内存
    print(f"张量t2: {t2},dtype: {t2.dtype}", f"类型: {type(t2)}")
    n2[0]=100
    print(f"numpy数组n2: {n2}", f"类型: {type(n2)}")
    print(f"张量t2: {t2}", f"类型: {type(t2)}")
def dm03():
    t3 = torch.tensor(100)
    a=t3.item() #共享内存
    print(f"张量t3: {t3},dtype: {t3.dtype}", f"类型: {type(t3)}")
    print(f"提取的标量值: {a},类型: {type(a)}")


if __name__ == "__main__":
    #dm01()
    #dm02()
    dm03()