import torch
def dm01():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = a * b
    print(c)
    print('-'*30    )
def dm02():
    a=torch.tensor([[1,2,3],[4,5,6]])
    b=torch.tensor([[7,8],[10,11],[12,13]])
    c=a@b
    print(c)
if __name__ == '__main__':    
    dm01()
    dm02()