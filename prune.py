import torch
import torch.nn as nn


def prune_net(net, independentflag, prune_layers, prune_channels, net_name, shortcutflag):
    print("pruning:")
    if net_name == 'vgg16':
        return prune_vgg(net, independentflag, prune_layers, prune_channels)
    elif net_name == "resnet34":
        return prune_resnet(net, independentflag, prune_layers, prune_channels, shortcutflag)
    else:
        print("The net is not provided.")
        exit(0)


def prune_vgg(net, independentflag, prune_layers, prune_channels):

    last_prune_flag = 0
    arg_index = 0
    conv_index = 1
    residue = None

    for i in range(len(net.module.features)):
        if isinstance(net.module.features[i], nn.Conv2d):
            # prune next layer's filter in dim=1
            if last_prune_flag:
                net.module.features[i], residue = get_new_conv(net.module.features[i], remove_channels, 1)
                last_prune_flag = 0
            # prune this layer's filter in dim=0
            if "conv_%d" % conv_index in prune_layers:
                remove_channels = channels_index(net.module.features[i].weight.data, prune_channels[arg_index], residue,
                                                 independentflag)
                print(prune_layers[arg_index], remove_channels)
                net.module.features[i] = get_new_conv(net.module.features[i], remove_channels, 0)
                last_prune_flag = 1
                arg_index += 1
            else:
                residue = None
            conv_index += 1
        elif isinstance(net.module.features[i], nn.BatchNorm2d) and last_prune_flag:
            # prune bn
            net.module.features[i] = get_new_norm(net.module.features[i], remove_channels)

    # prune linear
    if "conv_13" in prune_layers:
        net.module.classifier[0] = get_new_linear(net.module.classifier[0], remove_channels)
    net = net.cuda()
    print(net)
    return net


def prune_resnet(net, independentflag, prune_layers, prune_channels, shortcutflag):
    # init
    last_prune_flag = 0
    arg_index = 0
    residue = None
    layers = [net.module.layer1, net.module.layer2, net.module.layer3, net.module.layer4]

    # prune shortcut
    if shortcutflag:
        downsample_index = 1
    # identify channels to remove
    # prune this layer's filter in dim=0
    # prune next layer's filter in dim=1
    # prune bn
    # prune linear
    # prune non-shortcut
    else:
        conv_index = 2
        for layer_index in range(len(layers)):
            for block_index in range(len(layers[layer_index])):
                if "conv_%d" % conv_index in prune_layers:
                    # identify channels to remove
                    remove_channels = channels_index(layers[layer_index][block_index].conv1.weight.data,
                                                     prune_channels[arg_index], residue, independentflag)
                    print(prune_layers[arg_index], remove_channels)
                    # prune this layer's filter in dim=0
                    layers[layer_index][block_index].conv1 = get_new_conv(layers[layer_index][block_index].conv1,
                                                                          remove_channels, 0)
                    # prune next layer's filter in dim=1
                    layers[layer_index][block_index].conv2, residue = get_new_conv(
                        layers[layer_index][block_index].conv2, remove_channels, 1)
                    residue = 0
                    # prune bn
                    layers[layer_index][block_index].bn1 = get_new_norm(layers[layer_index][block_index].bn1,
                                                                        remove_channels)
                    arg_index += 1
                conv_index += 2
    net = net.cuda()
    print(net)
    return net


def channels_index(weight_matrix, prune_num, residue, independentflag):
    abs_sum = torch.sum(torch.abs(weight_matrix.view(weight_matrix.size(0), -1)), dim=1)
    if independentflag:
        abs_sum = abs_sum + torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)
    _, indices = torch.sort(abs_sum)
    return indices[:prune_num].tolist()


def select_channels(weight_matrix, remove_channels, dim):
    indices = torch.tensor(list(set(range(weight_matrix.shape[dim])) - set(remove_channels)))
    new = torch.index_select(weight_matrix, dim, indices.cuda())
    if dim == 1:
        residue = torch.index_select(weight_matrix, dim, torch.tensor(remove_channels).cuda())
        return new, residue
    return new


def get_new_conv(old_conv, remove_channels, dim):
    if dim == 0:
        new_conv = nn.Conv2d(in_channels=old_conv.in_channels,
                             out_channels=old_conv.out_channels - len(remove_channels),
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding,
                             dilation=old_conv.dilation, bias=old_conv.bias is not None)
        new_conv.weight.data = select_channels(old_conv.weight.data, remove_channels, dim)
        if old_conv.bias is not None:
            new_conv.bias.data = select_channels(old_conv.bias.data, remove_channels, dim)
        return new_conv
    else:
        new_conv = nn.Conv2d(in_channels=old_conv.in_channels - len(remove_channels), out_channels=old_conv.out_channels,
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding,
                             dilation=old_conv.dilation, bias=old_conv.bias is not None)
        new_conv.weight.data, residue = select_channels(old_conv.weight.data, remove_channels, dim)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data
        return new_conv, residue


def get_new_norm(old_norm, remove_channels):
    new = torch.nn.BatchNorm2d(num_features=old_norm.num_features - len(remove_channels), eps=old_norm.eps,
                               momentum=old_norm.momentum, affine=old_norm.affine,
                               track_running_stats=old_norm.track_running_stats)
    new.weight.data = select_channels(old_norm.weight.data, remove_channels, 0)
    new.bias.data = select_channels(old_norm.bias.data, remove_channels, 0)

    if old_norm.track_running_stats:
        new.running_mean.data = select_channels(old_norm.running_mean.data, remove_channels, 0)
        new.running_var.data = select_channels(old_norm.running_var.data, remove_channels, 0)

    return new


def get_new_linear(old_linear, remove_channels):
    new = torch.nn.Linear(in_features=old_linear.in_features - len(remove_channels),
                          out_features=old_linear.out_features, bias=old_linear.bias is not None)
    new.weight.data, residue = select_channels(old_linear.weight.data, remove_channels, 1)
    if old_linear.bias is not None:
        new.bias.data = old_linear.bias.data
    return new
