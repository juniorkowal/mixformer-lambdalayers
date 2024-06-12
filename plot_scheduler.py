import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import SGD
import matplotlib.pyplot as plt

def plot_scheduler(steps, optimizer, scheduler):

    lrs = []
    for _ in range(steps):
        optimizer.step()
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    plt.plot(lrs)
    plt.show()