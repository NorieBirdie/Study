import torch
t1=torch.tensor([1,2,3])
t2=t1.add(10)
t3=t1.add_(10)
print(f't1={t1},t2={t2},t3={t3}')
