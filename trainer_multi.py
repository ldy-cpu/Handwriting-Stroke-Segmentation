import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss,BCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss,DiceLoss_multi, DiceLoss_multi_hard
from torchvision import transforms


def worker_init_fn(worker_id):
    # random.seed(args.seed + worker_id)
    random.seed(1234 + worker_id)

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator,RandomGenerator_val
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))



    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",
                             transform=transforms.Compose(
                                 [RandomGenerator_val(output_size=[args.img_size, args.img_size])]))
    print("The length of val set is: {}".format(len(db_val)))

    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)




    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ######################################################################################################################################
    #单标签分类
    # ce_loss = CrossEntropyLoss()
    # dice_loss = DiceLoss(num_classes)
    ##########################################################################################################################################
    ######################################################################################################################################
    # 多标签分类
    ce_loss = BCELoss()
    dice_loss = DiceLoss_multi(num_classes)
    dice_loss_hard = DiceLoss_multi_hard(num_classes)
    ##########################################################################################################################################
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    sigmoid = nn.Sigmoid()
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            soft = True
##########################################################################################################################################
            #多标签分类
            outputs = sigmoid(outputs)
            soft = False
            loss_ce = ce_loss(outputs, label_batch[:].float())
##########################################################################################################################################
            # 单标签
            # loss_ce = ce_loss(outputs, label_batch[:])
#################################################################################
            # loss_dice = dice_loss(outputs, label_batch, weight=[1.8, 0.2], softmax=soft)   #单标签分类时用的权重比例
            loss_dice = dice_loss(outputs, label_batch, softmax=soft)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            # if iter_num % 20 == 0:
            #     image = image_batch[1, 0:1, :, :]
            #     image = (image - image.min()) / (image.max() - image.min())
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 30  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     logging.info('epoch:%d' % (epoch_num))



        if epoch_num > int(max_epoch / 2):
        # if epoch_num > 1:
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            model.eval()
            i = 0
            with torch.no_grad():
                total_loss_dice = 0
                total_acc = 0
                # total_loss = 0
                for i_batch, sampled_batch in enumerate(valloader):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    outputs = sigmoid(outputs)
                    # loss_ce = ce_loss(outputs, label_batch[:].float())

                    loss_dice_hard, acc = dice_loss_hard(outputs[:], label_batch[:], softmax=soft)

                    loss_dice = dice_loss(outputs[:], label_batch[:], softmax=soft)

                    print("acc:{}".format(acc))
                    print("loss_dice_hard:{}".format(loss_dice_hard))
                    print("loss_dice:{}".format(loss_dice))




                    # loss = 0.5 * loss_ce + 0.5 * loss_dice
                    total_loss_dice = (loss_dice_hard + total_loss_dice)
                    total_acc = (acc + total_acc)
                    # total_loss = (loss + total_loss)
                    i += 1

            total_loss_dice = total_loss_dice / i
            total_acc = total_acc / i
            dice = 1 - total_loss_dice
                # total_loss = total_loss / i

            print("average_acc:{}".format(total_acc))
            print("average_dice_hard:{}".format(dice))
            model.train()
            if dice > best_performance:
                best_performance = dice
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                logging.info('epoch:%d' % (epoch_num))






        if epoch_num >= max_epoch - 1:
            # save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            # torch.save(model.state_dict(), save_mode_path)
            # logging.info("save model to {}".format(save_mode_path))
            # iterator.close()
            break

    writer.close()
    return "Training Finished!"