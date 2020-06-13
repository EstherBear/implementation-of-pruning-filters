import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", default='resnet34', help='net type')
    parser.add_argument("-dataset", default='imagenet', help='dataset')
    parser.add_argument("-b", default=256, type=int, help='batch size')
    parser.add_argument("-lr", default=0.1, help='initial learning rate', type=float)
    parser.add_argument("-e", default=90, help='EPOCH', type=int)
    parser.add_argument("-optim", default="SGD", help='optimizer')
    parser.add_argument("-gpu", default="0,1", help='select GPU', type=str)
    parser.add_argument("-retrainflag", action='store_true', help='retrain or not', default=False)
    parser.add_argument("-retrainepoch", default=20, help='retrain EPOCH', type=int)
    parser.add_argument("-retrainlr", default=0.001, help='retrain learning rate', type=float)
    parser.add_argument("-trainflag", action='store_true', help='train or not', default=False)
    parser.add_argument("-pruneflag", action='store_true', help='prune or not', default=False)
    parser.add_argument("-sortflag", action='store_true', help='sort filter by abs sum of weights or not', default=False)
    parser.add_argument("-independentflag", action='store_true', help='pruning strategy', default=False)
    parser.add_argument("-shortcutflag", action='store_true', help='prune the shortcut', default=True)
    parser.add_argument("-prune_channels",  nargs='+', type=int,
                        help='the number of channels to prune corresponding to the prune_layers')
    parser.add_argument("-prune_layers",  nargs='+', help='the layers to prune')
    args = parser.parse_args()
    return args
