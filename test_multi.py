import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator_test, RandomGenerator_val
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from scipy.ndimage import zoom
from torchvision import transforms
import configg


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./data/{}/all'.format(configg.name), help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=configg.num_classes, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=configg.epoch, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.004, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=configg.patch_size, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


def partial_regression(ori, stroke):

    circle_cpy = []
    for p in range(0, r+1):
        circle_cpy.append(0)


    reoutput = np.zeros((stroke.shape[0],stroke.shape[1]))
    for i in range(0,reoutput.shape[0]):
        for j in range(0,reoutput.shape[1]):
            if ori[i][j] == 255: #原图中背景不进行计算,且把背景中的灰色去掉
                reoutput[i][j] = 255
                continue

            circle_num = circle_cpy[:]  #记录每一圈有效像素个数
            circle_sum = circle_cpy[:]  #记录每一圈有效像素灰度值之和

            for ii in range(i - r,i + r + 1):
                for jj in range(j - r, j + r + 1):
                    if ii < 0 or ii >= reoutput.shape[0]:
                        continue
                    if jj < 0 or jj >= reoutput.shape[1]:
                        continue

                    if ori[ii][jj] == 255: #原图中背景像素不对目标像素产生影响
                        continue

                    circle_no = abs(max(ii-i,jj-j))
                    circle_num[circle_no] = circle_num[circle_no] + 1
                    circle_sum[circle_no] = circle_sum[circle_no] + stroke[ii][jj]
                    # print(stroke[ii][jj])

            pix_value = 0
            for p in range(0,r + 1):#每一圈的灰度平均值乘以这一圈的权重，所有圈加权得到目标像素灰度值
                if circle_num[p] == 0: #如果某一圈全是背景，则把这一圈的权重分配给其余圈
                    pix_value = pix_value + circle_weight[p] * pix_value
                    continue
                pix_value = pix_value + circle_sum[p]/circle_num[p] * circle_weight[p]
                # print(circle_sum[p],circle_num[p])
                # if stroke[i][j] > 128:
                    # print(pix_value)

            reoutput[i][j] = pix_value

    return reoutput

class DiceLoss_multi_hard(nn.Module):

    def __init__(self, n_classes):
        super(DiceLoss_multi_hard, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()


    def acc(self, score, target):
        correct = (score == target)
        correct = correct.float().sum()
        ac = correct/(224*224)
        assert ac<=1,'acc > 1'
        return ac

    def _dice_loss(self, score, target):
        # 逆转颜色
        target = 1 - target
        score = 1 - score
        target = target.float()

        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)

        score_hard = (inputs > 0.5).float()
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0

        dice = self._dice_loss(score_hard[:], target[:])


        accuracy = self.acc(score_hard[:], target[:])

        return dice, accuracy


def test_with_no_label(args, model, test_save_path=None):
    # db_test = Synapse_dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir,
    #                            transform=transforms.Compose(
    #                                [RandomGenerator_test(output_size=[args.img_size, args.img_size])]))
    # testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    db_val = Synapse_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="val",
                             transform=transforms.Compose(
                                 [RandomGenerator_val(output_size=[args.img_size, args.img_size])]))

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    loss_dice_hard = DiceLoss_multi_hard(configg.num_classes)
    logging.info("{} test iterations per epoch".format(len(valloader)))
    model.eval()
    sigmoid = nn.Sigmoid()
    total_dice = 0
    total_acc = 0
    j = 0
    for i_batch, sampled_batch in tqdm(enumerate(valloader)):
        j += 1
        image, case_name, label = sampled_batch["image"], sampled_batch['case_name'][0], sampled_batch["label"]
        label = label.squeeze(0)
        # image = image.squeeze(0)
        #
        # x, y = image.shape[0], image.shape[1]
        # if x != 256 or y != 256:
        #     image = zoom(image, (256 / x, 256 / y), order=3)
        # image = torch.from_numpy(image)
        # image = image.unsqueeze(0).unsqueeze(0)
        this_path = test_save_path + "/" + case_name
        if not os.path.exists(this_path):
            os.mkdir(this_path)




        with torch.no_grad():
            image = image.cuda()
            image = image.float()
            outputs = model(image)
            # outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)  #单标签分类
            outputs = sigmoid(outputs)#多标签分类
            image = image.squeeze(0).squeeze(0)
            image = image.cpu().numpy()
            image = image * 255
            outputs = outputs.squeeze(0)
            for i in range(0, configg.num_classes):
                output = outputs[i, :]
                output = output.squeeze(0)

                output = output.cpu().numpy()

                # output = output > 0.2
                output = output * 255




                # output = [output,output,output]
                output = np.array(output)


                # image = np.array(image)
                # output = output.transpose((1,2,0))

                reoutput = partial_regression(image, output[:])

                reoutput_tensor = torch.from_numpy(reoutput/255)
                dice_hard, acc = loss_dice_hard(reoutput_tensor[:], label[i,:], softmax = False)
                total_dice = total_dice + dice_hard
                total_acc = total_acc + acc

                # output = output > 128
                # output = output * 255

                save_pic = Image.fromarray(np.uint8(output))
                save_re = Image.fromarray(np.uint8(reoutput))

                save_pic.save(this_path + "/{}.png".format(i+1))
                save_re.save(this_path + "/{}_regression.png".format(i + 1))
    accuracy = total_acc/j/configg.num_classes
    f1 = total_dice/j/configg.num_classes
    print("accuracy = ", accuracy)
    print("f1 = ", f1)

if __name__ == "__main__":

    r = 8   #局部回归半径
    decay = 0.9 #向外圈影响因子递减系数

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataset_config = {
    #     'Synapse': {
    #         'Dataset': Synapse_dataset,
    #         'volume_path': '../data/Synapse/test_vol_h5',
    #         'list_dir': './lists/lists_Synapse',
    #         'num_classes': 9,
    #         'z_spacing': 1,
    #     },
    # }
    dataset_name = args.dataset
    # args.num_classes = dataset_config[dataset_name]['num_classes']
    # args.volume_path = dataset_config[dataset_name]['volume_path']
    # args.Dataset = dataset_config[dataset_name]['Dataset']
    # args.list_dir = dataset_config[dataset_name]['list_dir']
    # args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = False

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    # snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = args.volume_path + '/snapshot'
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot_path = os.path.join(args.volume_path, "snapshot_R50-ViT-B_16_skip3_epo{}_bs8_lr0.004_224".format(configg.epoch))
    # snapshot_path = os.path.join(args.volume_path,
    #                              "4class_snapshot_R50-ViT-B_16_skip3_epo170_bs8_lr0.004_224")


    snapshot = os.path.join(snapshot_path, 'best_model.pth')

    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    # logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    # logging.info(snapshot_name)

    # if args.is_savenii:
    # args.test_save_dir = '../predictions'

    test_save_path = os.path.join(args.volume_path, "test")
    test_save_path = os.path.join(test_save_path, "label_result")
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path, exist_ok=True)

    # else:
    #     test_save_path = None
    # inference(args, net, test_save_path)
########################################################
    circle_weight_sum = 1
    circle_weight = []

    last = 1
    for q in range(1, r + 1):  # 求所有圈权重的和，作为分母
        last = last * decay
        circle_weight_sum = circle_weight_sum + last

    son = 1
    for p in range(0, r + 1):
        circle_weight.append(son / circle_weight_sum)
        son = son * decay
    print(circle_weight)
###########################################################
    test_with_no_label(args,net,test_save_path)

