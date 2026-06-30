import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def dm01():
    lr, epoch, iterattion, step_size = 0.1, 200, 10, 50
    # 创建数据集
    y_ture = torch.tensor([0], dtype=torch.float32)
    x = torch.tensor([1.0], dtype=torch.float32)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 创建优化器
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    # 创建学习率衰减对象
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    lr_list = []
    epoch_list = []

    # 循环遍历训练轮数，进行具体的训练
    for i in range(epoch):
        # 获取当前轮数和学习率，并保存到列表中
        epoch_list.append(i)
        lr_list.append(scheduler.get_last_lr()[0])

        # 循环遍历，每轮每批次进行训练
        for batch in range(iterattion):
            # 前向传播
            y_pred = x * w
            # 计算损失
            loss = nn.MSELoss()(y_pred, y_ture)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
        # 更新学习率
        scheduler.step()

    # 打印结果
    print(f"lr_list: {lr_list}")

    # 可视化
    plt.plot(epoch_list, lr_list, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dm01()
