import torch.nn as nn
import torch
import os
import sys

import datetime
import torch.cuda

from tools.optim_sche import get_optim_sche
from tools.get_data import get_train_loader, get_test_loader
from tools.get_parameters import get_args
from tools.flops_params import get_flops_params

from prune import prune_net

CHECK_POINT_PATH = "./checkpoint"


def train_epoch(net, epoch, trainloader, loss_function, optimizer, fepoch, fstep):
    net.train()

    length = len(trainloader)
    total_sample = len(trainloader.dataset)
    total_loss = 0
    correct_1 = 0
    correct_5 = 0
    batch_size = 0

    for step, (x, y) in enumerate(trainloader):
        x = x.cuda()
        y = y.cuda()
        if step == 0:
            batch_size = len(y)
        optimizer.zero_grad()

        output = net(x)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predict = output.topk(5, 1, True, True)
        # _, predict = torch.max(output, 1)
        predict = predict.t()
        correct = predict.eq(y.view(1, -1).expand_as(predict))
        correct_1 += correct[:1].view(-1).sum()
        correct_5 += correct[:5].view(-1).sum()
        # correct += (predict == y).sum()

        if step % 30 == 0:
            print("Epoch:{}\t Step:{}\t TrainedSample:{}\t TotalSample:{}\t Loss:{:.3f}".format(
                epoch + 1, step + 1, step * batch_size + len(y), total_sample, loss.item()
            ))
            fstep.write("Epoch:{}\t Step:{}\t TrainedSample:{}\t TotalSample:{}\t Loss:{:.3f}\n".format(
                epoch + 1, step + 1, step * batch_size + len(y), total_sample, loss.item()
            ))
            fstep.flush()

    fepoch.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc1:{:.3%}\t acc5:{:.3%}\n".format(
        epoch + 1, total_loss/length, optimizer.param_groups[0]['lr'], float(correct_1) / total_sample,
        float(correct_5)/total_sample
    ))
    fepoch.flush()
    return net


def eval_epoch(net, testloader):
    loss_function = nn.CrossEntropyLoss()
    net.eval()

    length = len(testloader)
    total_sample = len(testloader.dataset)
    total_loss = 0
    correct_1 = 0
    correct_5 = 0
    inference_time = 0

    for step, (x, y) in enumerate(testloader):
        x = x.cuda()
        y = y.cuda()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = net(x)
        # _, predict = torch.max(output, 1)
        _, predict = output.topk(5, 1, True, True)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        inference_time += start.elapsed_time(end)  # milliseconds

        loss = loss_function(output, y)

        total_loss += loss.item()

        predict = predict.t()
        correct = predict.eq(y.view(1, -1).expand_as(predict))
        correct_1 += correct[:1].view(-1).sum()
        correct_5 += correct[:5].view(-1).sum()
        # correct += (predict == y).sum()

    acc1 = float(correct_1) / total_sample
    acc5 = float(correct_5) / total_sample
    return acc1, acc5, total_loss/length, inference_time


def training(net, total_epoch, trainloader, testloader, retrain, lr, optim, most_recent_path, train_checkpoint_path):
    # define optimizer, scheduler and loss function
    optimizer, scheduler = get_optim_sche(lr, optim, net, args.dataset, retrain=retrain)
    loss_function = nn.CrossEntropyLoss()

    # initial best_acc(for early stop) and total_time(for inference time)
    best_acc = 0
    total_time = 0

    if not os.path.exists(train_checkpoint_path):
        os.makedirs(train_checkpoint_path)

    with open(os.path.join(train_checkpoint_path, 'EpochLog.txt'), 'w') as fepoch:
        with open(os.path.join(train_checkpoint_path, 'StepLog.txt'), 'w') as fstep:
            with open(os.path.join(train_checkpoint_path, 'EvalLog.txt'), 'w') as feval:
                with open(os.path.join(train_checkpoint_path, 'Best.txt'), 'w') as fbest:

                    print("start training")
                    for epoch in range(total_epoch):
                        train_epoch(net, epoch, trainloader, loss_function, optimizer, fepoch, fstep)

                        print("evaluating")
                        accuracy1, accuracy5, averageloss, inference_time = eval_epoch(net, testloader)
                        feval.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc1:{:.3%}\t acc5:{:.3%}\n".format(
                            epoch + 1, averageloss, optimizer.param_groups[0]['lr'], accuracy1, accuracy5
                        ))
                        feval.flush()

                        if scheduler is not None:
                            scheduler.step()

                        print("saving regular")
                        torch.save(net.state_dict(), os.path.join(train_checkpoint_path, 'regularParam.pth'))

                        if accuracy1 > best_acc:
                            print("saving best")
                            torch.save(net.state_dict(), os.path.join(train_checkpoint_path, 'bestParam.pth'))
                            torch.save(net.state_dict(), os.path.join(most_recent_path, 'bestParam.pth'))
                            fbest.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc1:{:.3%}\t acc5:{:.3%}\n".format(
                                epoch + 1, averageloss, optimizer.param_groups[0]['lr'], accuracy1, accuracy5
                            ))
                            fbest.flush()
                            best_acc = accuracy1
                        # print(inference_time)
                        total_time += (inference_time / len(testloader.dataset))

                    print(total_time)
                    print(total_time / total_epoch)
    return net


if __name__ == '__main__':
    # arguments from command line
    args = get_args()

    # data processing
    train_loader = get_train_loader(args)
    test_loader = get_test_loader(args)

    # define gpus and get net
    device_ids = [int(i) for i in list(args.gpu.split(','))]
    net = None
    if args.net == 'vgg16':
        from netModels.VGG import MyVgg16
        net = MyVgg16(10)
        print(net)
    else:
        print('We don\'t support this net.')
        sys.exit()
    net = nn.DataParallel(net, device_ids=device_ids)
    net = net.cuda()

    # define checkpoint path
    time = str(datetime.date.today() + datetime.timedelta(days=2))
    checkpoint_path = os.path.join(CHECK_POINT_PATH, args.net)
    train_checkpoint_path = os.path.join(checkpoint_path, 'train', time)
    train_most_recent_path = os.path.join(checkpoint_path, 'train')
    prune_checkpoint_path = os.path.join(checkpoint_path, 'prune', time)
    prune_most_recent_path = os.path.join(checkpoint_path, 'prune')
    retrain_checkpoint_path = os.path.join(checkpoint_path, 'retrain', time)
    retrain_most_recent_path = os.path.join(checkpoint_path, 'retrain')

    # train
    if args.trainflag:
        training(net, args.e, train_loader, test_loader, False, args.lr, args.optim,
                 train_most_recent_path, train_checkpoint_path)

    if args.pruneflag:
        if not os.path.exists(prune_checkpoint_path):
            os.makedirs(prune_checkpoint_path)
        net.load_state_dict(torch.load(os.path.join(train_most_recent_path, 'bestParam.pth')))
        new_net = prune_net(net, args.independentflag, args.prune_layers, args.prune_channels)
        top1, top5, loss, infer_time = eval_epoch(new_net, test_loader)
        print("Eval after pruning:\t Loss:{:.3f}\t acc1:{:.3%}\t acc5:{:.3%}\t Inference time:{:.3%}\n"
              .format(loss, top1, top5, infer_time/len(test_loader.dataset)))

        f, p = get_flops_params(new_net.module.cpu())

        new_net = new_net.cuda()
        with open(os.path.join(prune_checkpoint_path, 'flops_and_params'), 'w') as fp:
            fp.write("flops:{}\t params:{}\n".format(f, p))
            fp.flush()

        torch.save(new_net.state_dict(), os.path.join(prune_checkpoint_path, 'prunedParam.pth'))
        torch.save(new_net.state_dict(), os.path.join(prune_most_recent_path, 'prunedParam.pth'))

    if args.retrainflag:
        training(net, args.retrainepoch, train_loader, test_loader, True, args.retrainlr, args.optim,
                 retrain_most_recent_path, retrain_checkpoint_path)











