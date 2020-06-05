import torch
import torch.optim as optim

CIFAR10_MILESTONES = [60, 120, 160]
ImageNet_MILESTONES = [30, 60]


def get_optim_sche(lr, opt, net, dataset, retrain=False):
    if opt == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    if retrain:
        scheduler = None
    else:
        MILESTONES = []
        gamma = 0
        if dataset == 'cifar10':
            MILESTONES = CIFAR10_MILESTONES
            gamma = 0.2
        elif dataset == 'imagenet':
            MILESTONES = ImageNet_MILESTONES
            gamma = 0.1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=gamma, last_epoch=-1)
    return optimizer, scheduler
