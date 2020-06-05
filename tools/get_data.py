import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


CIFAR10_TRAIN_MEAN = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
CIFAR10_TRAIN_STD = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)

ImageNet_TRAIN_MEAN = (0.485, 0.456, 0.406)
ImageNet_TRAIN_STD = (0.229, 0.224, 0.225)


def get_train_loader(args):
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_TRAIN_MEAN, std=CIFAR10_TRAIN_STD)
        ])

        traindata = torchvision.datasets.CIFAR10(root='../../../share/jianheng/data', train=True, download=False, transform=transform_train)
        trainloader = DataLoader(traindata, batch_size=args.b, shuffle=True, num_workers=2)
        return trainloader
    elif args.dataset == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=ImageNet_TRAIN_MEAN, std=ImageNet_TRAIN_STD)
        ])

        traindata = torchvision.datasets.ImageFolder(root='../../../share/jianheng/data/imagenet/train',
                                                     transform=transform_train)
        trainloader = DataLoader(traindata, batch_size=args.b, shuffle=True, num_workers=2)
        return trainloader


def get_test_loader(args):
    if args.dataset == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_TRAIN_MEAN, std=CIFAR10_TRAIN_STD)
        ])

        testdata = torchvision.datasets.CIFAR10(root='../../../share/jianheng/data', train=False, download=False, transform=transform_test)
        testloader = DataLoader(testdata, batch_size=args.b, shuffle=True, num_workers=2)
        return testloader
    elif args.dataset == 'imagenet':
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=ImageNet_TRAIN_MEAN, std=ImageNet_TRAIN_STD)
        ])

        testdata = torchvision.datasets.ImageFolder(root='../../../share/jianheng/data/imagenet/val',
                                                    transform=transform_test)
        testloader = DataLoader(testdata, batch_size=args.b, shuffle=True, num_workers=2)
        return testloader

