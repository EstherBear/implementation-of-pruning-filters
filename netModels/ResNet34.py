import torchvision


def MyResNet34():
    return torchvision.models.resnet34(pretrained=True)
