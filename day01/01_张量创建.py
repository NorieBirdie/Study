import torch
import numpy as np
def dm01():
    # 标量
    tq=torch.tensor(1)
    print(f't1:{tq}, type: {type(tq)}')
    print('-'*30)

    data=[[1,2,3],[4,5,6]]
    t2=torch.tensor(data)
    print(f't2:{t2}, type: {type(t2)}')
    print('-'*30)

    data=np.array([[14,2,3],[4,5,6]])
    t3=torch.tensor(data)
    print(f't3:{t3}, type: {type(t3)}')

    t4=torch.Tensor(2,3)
    print(f't4:{t4}, type: {type(t4)}')
    print('-'*30)
    
if __name__ == "__main__":
    dm01()