import torch.nn as nn
import torch
import torch.cuda

import matplotlib.pyplot as plt
import numpy as np

import os
import json
import datetime
import sys

from netModels.VGG import MyVgg16
from netModels.ResNet34 import MyResNet34

from tools.get_data import get_test_loader
from tools.get_data import get_train_loader
from tools.get_parameters import get_args

from prune import prune_net
from train import eval_epoch
from train import training

CHECK_POINT_PATH = "./checkpoint"
colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
lines = ['-', '--', '-.']
vgg16_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10',
                'conv_11', 'conv_12', 'conv_13']
vgg16_total_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
first_layers = ['conv_2', 'conv_4', 'conv_6', 'conv_8', 'conv_10', 'conv_12', 'conv_14', 'conv_16', 'conv_18',
                'conv_20','conv_22', 'conv_24', 'conv_26', 'conv_28', 'conv_30', 'conv_32']
first_total_channels = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512]
shortcut_layers = ['downsample_1', 'downsample_2', 'downsample_3']
shortcut_total_channels = [128, 256, 512]
device_ids = [0, 1]


def sort_filter(args):
    # get the desired net
    load_path = os.path.join(CHECK_POINT_PATH, args.net, "train", "bestParam.pth")
    net = get_net(args.net)
    new_net = get_net(args.net)
    if os.path.exists(load_path):
        net.load_state_dict(torch.load(load_path))

    # make the figure
    shortcut = ""
    if args.shortcutflag:
        shortcut = "shortcut_"
    plt.figure()
    conv_count = 0
    figure_count = 1
    for layer in net.module.modules():
        if isinstance(layer, nn.Conv2d):
            # exclude shortcut conv or residual conv
            if args.shortcutflag:
                if layer.kernel_size != (1, 1):
                    continue
            else:
                if layer.kernel_size == (1, 1):
                    continue
            line_style = colors[conv_count % len(colors)] + lines[conv_count // len(colors) % len(lines)]
            weight = layer.weight.data.cpu().numpy()
            abs_sum_sorted = np.sort(np.sum(((np.abs(weight)).reshape(weight.shape[0], -1)), axis=1), axis=0)[::-1]
            norm_filter = abs_sum_sorted/abs_sum_sorted[0]
            conv_count += 1
            plt.plot(np.linspace(0, 100, norm_filter.shape[0]), norm_filter, line_style,
                     label=shortcut + 'conv %d' % conv_count)

            # if there are too many convs in a figure, make a new one
            if conv_count % 17 == 0:
                plt.title("Data: %s" % args.dataset + ", Model: %s" % args.net)
                plt.ylabel("normalized abs sum of filter weight")
                plt.xlabel("filter index / # filters (%)")
                plt.legend(loc='upper right')
                plt.xlim([0, 140])
                # plt.grid()
                plt.savefig(args.net + "_" + shortcut + str(figure_count) + "_" + "filters_ranked.png",
                            bbox_inches='tight')
                plt.show()
                plt.figure()
                figure_count += 1

    plt.title("Data: %s" % args.dataset + ", Model: %s" % args.net)
    plt.ylabel("normalized abs sum of filter weight")
    plt.xlabel("filter index / # filters (%)")
    plt.legend(loc='upper right')
    plt.xlim([0, 140])
    # plt.grid()
    plt.savefig(args.net + "_" + shortcut + str(figure_count) + "_" + "filters_ranked.png", bbox_inches='tight')
    plt.show()


def prune_analysis(args):
    # get the desired net layers, and channels
    layers, total_channels = get_list(args.net, args.shortcutflag)
    load_path = os.path.join(CHECK_POINT_PATH, args.net, "train", "bestParam.pth")

    # get the args for eval
    test_loader = get_test_loader(args)
    independentflag = False

    # the parameter for prune
    max_prune_ratio = 0.90
    accuracy1 = {}
    accuracy5 = {}

    # for all layers in the net
    for conv, channels in zip(layers[14:], total_channels[14:]):
        new_net = get_net(args.net)
        if os.path.exists(load_path):
            new_net.load_state_dict(torch.load(load_path))
        # ##
        # print(new_net)
        # return
        print("evaluating")
        top1, top5, loss, infer_time = eval_epoch(new_net, test_loader)
        print("Eval before pruning" + ": Loss:{:.3f}\t acc1:{:.3%}\t acc5:{:.3%}\t Inference time:{:.3}\n"
              .format(loss, top1, top5, infer_time / len(test_loader.dataset)))

        accuracy1[conv] = [top1]
        accuracy5[conv] = [top5]

        prune_layers = [conv]
        prune_channels = np.linspace(0, int(channels * max_prune_ratio), 10, dtype=int)
        prune_channels = (prune_channels[1:] - prune_channels[:-1]).tolist()

        # for each layer
        for index, prune_channel in enumerate(prune_channels):
            # prune
            new_net = prune_net(new_net, independentflag, prune_layers, [prune_channel], args.net, args.shortcutflag)
            # eval
            print("evaluating")
            top1, top5, loss, infer_time = eval_epoch(new_net, test_loader)
            print("Eval after pruning " + conv, index, ":\t Loss:{:.3f}\t acc1:{:.3%}\t acc5:{:.3%}\t " 
                                                       "Inference time:{:.3}\n".format(loss, top1, top5, infer_time /
                                                                                       len(test_loader.dataset)))

            accuracy1[conv].append(top1)
            accuracy5[conv].append(top5)

            with open('top1_2', 'w') as fout:
                json.dump(accuracy1, fout)

            with open('top5_2', 'w') as fout:
                json.dump(accuracy5, fout)

    plt.figure()
    for index, (conv, acc1) in enumerate(accuracy1.items()):
        line_style = colors[index % len(colors)] + lines[index // len(colors) % len(lines)] + 'o'
        xs = np.linspace(0, 90, len(acc1))
        plt.plot(xs, acc1, line_style,
                 label=conv+' '+str(total_channels[index]))
    plt.title("Data: %s" % args.dataset + ", Model: %s" % args.net + ", pruned smallest filters (greedy)")
    plt.ylabel("Accuracy(top1)")
    plt.xlabel("Filters Pruned Away(%)")
    plt.legend(loc='lower right')
    plt.xlim([0, 140])
    # plt.grid()
    plt.savefig(args.dataset + "_pruned_top1.png", bbox_inches='tight')
    plt.show()

    plt.figure()
    for index, (conv, acc5) in enumerate(accuracy5.items()):
        line_style = colors[index % len(colors)] + lines[index // len(colors) % len(lines)] + 'o'
        xs = np.linspace(0, 90, len(acc5))
        plt.plot(xs, acc5, line_style,
                 label=conv + ' ' + str(total_channels[index]))
    plt.title("Data: %s" % args.dataset + ", Model: %s" % args.net + ", pruned smallest filters (greedy)")
    plt.ylabel("Accuracy(top5)")
    plt.xlabel("Filters Pruned Away(%)")
    plt.legend(loc='lower right')
    plt.xlim([0, 140])
    # plt.grid()
    plt.savefig(args.dataset + "_pruned_top5.png", bbox_inches='tight')
    plt.show()


def get_net(net_name):
    if net_name == 'vgg16':
        net = MyVgg16(10)
    elif net_name == "resnet34":
        net = MyResNet34()
    else:
        print("The net is not provided.")
        exit(0)
    net = nn.DataParallel(net, device_ids=device_ids)
    net = net.cuda()
    return net


def get_list(net_name, shortcutflag):
    if net_name == 'vgg16':
        return vgg16_layers, vgg16_total_channels
    elif net_name == "resnet34":
        if shortcutflag:
            return shortcut_layers, shortcut_total_channels
        else:
            return first_layers, first_total_channels
    else:
        print("The net is not provided.")
        exit(0)


def prune_retrain_analysis():
    load_path = "./checkpoint/vgg16/train/bestParam.pth"
    test_loader = get_test_loader(args)
    loss_function = nn.CrossEntropyLoss()

    new_net = get_net()
    new_net.load_state_dict(torch.load(load_path))
    top1_org, top5_org, loss, infer_time = eval_epoch(new_net, test_loader, loss_function)

    independentflag = False
    max_prune_ratio = 0.90
    min_prune_ratio = 0.20
    accuracy1 = {}
    accuracy5 = {}

    # for all layers
    for conv, channels in zip(layers, total_channels):
        accuracy1[conv] = [top1_org]
        accuracy5[conv] = [top5_org]

        prune_layers = [conv]
        prune_channels = np.linspace(int(channels * min_prune_ratio), int(channels * max_prune_ratio), 8, dtype=int).tolist()

        # for each layer
        for index, prune_channel in enumerate(prune_channels):
            # get net and prune
            new_net = get_net()
            new_net.load_state_dict(torch.load(load_path))
            new_net = prune_net(new_net, independentflag, prune_layers, [prune_channel], device_ids)

            # retrain
            new_net = filter_retrain(new_net, conv + ':pruned %d' % prune_channel)

            # eval
            top1, top5, loss, infer_time = eval_epoch(new_net, test_loader, loss_function)
            print("Eval after pruning" + conv, index, ":\t Loss:{:.3f}\t acc1:{:.3%}\t acc5:{:.3%}\t "
                                                      "Inference time:{:.3}\n".format(loss, top1, top5, infer_time /
                                                                                      len(test_loader.dataset)))
            accuracy1[conv].append(top1)
            accuracy5[conv].append(top5)

            with open('top1', 'w') as fout:
                json.dump(accuracy1, fout)

            with open('top5', 'w') as fout:
                json.dump(accuracy5, fout)

    with open('top1', "r") as jsonfile:
        accuracy1 = json.load(jsonfile)
    with open('top5', "r") as jsonfile:
        accuracy5 = json.load(jsonfile)

    plt.figure()
    for index, (conv, acc1) in enumerate(accuracy1.items()):
        line_style = colors[index % len(colors)] + lines[index // len(colors)] + 'o'
        xs = [0] + list(np.linspace(20, 90, len(acc1)-1))
        xs = np.array(xs)
        plt.plot(xs, acc1, line_style,
                 label=conv + ' ' + str(total_channels[index]))
    plt.title("Data: CIFAR-10, Model: %s, pruned smallest filters (greedy), retrain 20 epochs" % args.net)
    plt.ylabel("Accuracy(top1)")
    plt.xlabel("Filters Pruned Away(%)")
    plt.legend(loc='lower left')
    plt.xlim([0, 100])
    # plt.grid()
    plt.savefig("retrained_top1.png", bbox_inches='tight')
    plt.show()

    plt.figure()
    for index, (conv, acc5) in enumerate(accuracy5.items()):
        line_style = colors[index % len(colors)] + lines[index // len(colors)] + 'o'
        xs = [0] + list(np.linspace(20, 90, len(acc5)-1))
        xs = np.array(xs)
        plt.plot(xs, acc5, line_style,
                 label=conv + ' ' + str(total_channels[index]))
    plt.title("Data: CIFAR-10, Model: %s, pruned smallest filters (greedy), retrain 20 epochs" % args.net)
    plt.ylabel("Accuracy(top5)")
    plt.xlabel("Filters Pruned Away(%)")
    plt.legend(loc='lower left')
    plt.xlim([0, 100])
    # plt.grid()
    plt.savefig("retrained_top5.png", bbox_inches='tight')
    plt.show()


def filter_retrain(net, dirname):
    checkpoint_path = os.path.join(CHECK_POINT_PATH, args.net)
    time = str(datetime.date.today())
    most_recent_path = os.path.join(checkpoint_path, 'retrain', "most_recnet", dirname)
    if not os.path.exists(most_recent_path):
        os.makedirs(most_recent_path)
    retrain_checkpoint_path = os.path.join(checkpoint_path, 'retrain', time, dirname)
    if not os.path.exists(retrain_checkpoint_path):
        os.makedirs(retrain_checkpoint_path)

    train_loader = get_train_loader(args)
    test_loader = get_test_loader(args)
    loss_function = nn.CrossEntropyLoss()

    with open(os.path.join(retrain_checkpoint_path, 'EpochLog.txt'), 'w') as fepoch:
        with open(os.path.join(retrain_checkpoint_path, 'StepLog.txt'), 'w') as fstep:
            with open(os.path.join(retrain_checkpoint_path, 'EvalLog.txt'), 'w') as feval:
                with open(os.path.join(retrain_checkpoint_path, 'Best.txt'), 'w') as fbest:
                    net = training(net, 20, train_loader, test_loader, loss_function, True, 0.001, 'SGD', fepoch, fstep,
                                   feval, fbest, retrain_checkpoint_path, most_recent_path)
    return net


if __name__ == '__main__':
    # arguments from command line
    args = get_args()
    # Analysis
    # sort_filter(args)
    prune_analysis(args)
    # prune_retrain_analysis()

