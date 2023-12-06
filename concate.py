import os
import configg
from PIL import Image
import numpy as np
import random
import shutil
from configg import concat_background

if not os.path.exists("./data/{}/all".format(configg.name)):
    os.mkdir("./data/{}/all".format(configg.name))

if not os.path.exists("./data/{}/all/train".format(configg.name)):
    os.mkdir("./data/{}/all/train".format(configg.name))
if not os.path.exists("./data/{}/all/val".format(configg.name)):
    os.mkdir("./data/{}/all/val".format(configg.name))

if not os.path.exists("./data/{}/all/train/image".format(configg.name)):
    os.mkdir("./data/{}/all/train/image".format(configg.name))
if not os.path.exists("./data/{}/all/val/image".format(configg.name)):
    os.mkdir("./data/{}/all/val/image".format(configg.name))

if not os.path.exists("./data/{}/all/train/label".format(configg.name)):
    os.mkdir("./data/{}/all/train/label".format(configg.name))
if not os.path.exists("./data/{}/all/val/label".format(configg.name)):
    os.mkdir("./data/{}/all/val/label".format(configg.name))
#
#
#################################################################################################
ori_name_path = "../data/ori_data/{}".format(configg.name)

whole_name = os.listdir("../data/ori_data/{}/condition".format(configg.name))
if not configg.num_classes == 2:
    if concat_background == False:

        for whole in whole_name:
            single_path = os.path.join(os.path.join(ori_name_path,"tar1"),whole)
            if not os.path.exists(single_path):
                continue
            concate = np.expand_dims(np.array(Image.open(single_path)),axis=0)

            tag = 0

            for i in range(2,configg.num_classes + 1):
                single_path = os.path.join(os.path.join(ori_name_path,"tar{}".format(i)),whole)
                if not os.path.exists(single_path):
                    tag = 1
                    break
                this = Image.open(single_path)
                this = np.expand_dims(np.array(this),axis=0)
                concate = np.concatenate((concate,this),axis=0)

            if tag == 1:
                continue




            #拼接完成，保存为.npy，以一定几率放在"./data/{}/all/train".format(config.name)   "./data/{}/all/val".format(config.name)
            rdm = random.uniform(0,10)
            if rdm < 2:
                np.save("./data/{}/all/val/label/{}.npy".format(configg.name, whole), concate)
                shutil.copy("../data/ori_data/{}/condition/".format(configg.name) + whole, "./data/{}/all/val/image/".format(configg.name) + whole)

            else:
                np.save("./data/{}/all/train/label/{}.npy".format(configg.name, whole),concate)
                shutil.copy("../data/ori_data/{}/condition/".format(configg.name) + whole,"./data/{}/all/train/image/".format(configg.name) + whole)

            #npy和condition分别放在label   image

            #还要写一个config，改train和val和test，test还要改保存图片的方式


            #改读取数据的方式

            #改bce的label输入，因为之前的代码label少一个维度(本来是怎么image和label不同维度计算的??)

    else:
        for whole in whole_name:
            single_path = os.path.join(os.path.join(ori_name_path, "tar1"), whole)
            if not os.path.exists(single_path):
                continue
            concate = np.expand_dims(np.array(Image.open(single_path)), axis=0)
            bck = 255 - concate

            tag = 0

            for i in range(2, configg.num_classes + 1):
                single_path = os.path.join(os.path.join(ori_name_path, "tar{}".format(i)), whole)
                if not os.path.exists(single_path):
                    tag = 1
                    break
                this = Image.open(single_path)
                this = np.expand_dims(np.array(this), axis=0)
                concate = np.concatenate((concate, this), axis=0)

            if tag == 1:
                continue

            # 拼接背景
            concate = np.concatenate((concate, bck), axis=0)

            # 拼接完成，保存为.npy，以一定几率放在"./data/{}/all/train".format(config.name)   "./data/{}/all/val".format(config.name)
            rdm = random.uniform(0, 10)
            if rdm < 2:
                np.save("./data/{}/all/val/label/{}.npy".format(configg.name, whole), concate)
                shutil.copy("../data/ori_data/{}/condition/".format(configg.name) + whole,
                            "./data/{}/all/val/image/".format(configg.name) + whole)

            else:
                np.save("./data/{}/all/train/label/{}.npy".format(configg.name, whole), concate)
                shutil.copy("../data/ori_data/{}/condition/".format(configg.name) + whole,
                            "./data/{}/all/train/image/".format(configg.name) + whole)
else:
    for whole in whole_name:
        single_path = os.path.join(os.path.join(ori_name_path, "tar{}".format(configg.target)), whole)
        if not os.path.exists(single_path):
            continue
        concate = Image.open(single_path)



        # 拼接完成，保存为.npy，以一定几率放在"./data/{}/all/train".format(config.name)   "./data/{}/all/val".format(config.name)
        rdm = random.uniform(0, 10)
        if rdm < 2:
            concate.save("./data/{}/all/val/label/{}".format(configg.name, whole))
            # np.save("./data/{}/all/val/label/{}.npy".format(configg.name, whole), concate)
            shutil.copy("../data/ori_data/{}/condition/".format(configg.name) + whole,
                        "./data/{}/all/val/image/".format(configg.name) + whole)

        else:
            concate.save("./data/{}/all/train/label/{}".format(configg.name, whole))
            # np.save("./data/{}/all/train/label/{}.npy".format(configg.name, whole), concate)
            shutil.copy("../data/ori_data/{}/condition/".format(configg.name) + whole,
                        "./data/{}/all/train/image/".format(configg.name) + whole)

##################################################################################################### single时保证训练集和验证集和之前的是一个字

# ori_name_path = "../data/ori_data/{}/tar{}".format(configg.name,configg.target)
# image_path = "./data/{}/all/train/image".format(configg.name)
# label_path = "./data/{}/all/train/label".format(configg.name)
#
# f = os.listdir(image_path)
# for k in f:
#     # if not os.path.exists(ori_name_path + '/' + k):
#     #     continue
#     shutil.copy(ori_name_path + '/' + k,
#                                 label_path + '/' + k)
#
# ori_name_path = "../data/ori_data/{}/tar{}".format(configg.name,configg.target)
# image_path = "./data/{}/all/val/image".format(configg.name)
# label_path = "./data/{}/all/val/label".format(configg.name)
#
# f = os.listdir(image_path)
# for k in f:
#     shutil.copy(ori_name_path + '/' + k,
#                                 label_path + '/' + k)


##################################################################################### multi时保证和之前的是一个字
# ii = os.listdir("./data/{}/all/train/image".format(configg.name))
# jj = os.listdir("./data/{}/all/val/image".format(configg.name))
# ori_name_path = "../data/ori_data/{}".format(configg.name)
#
# whole_name = os.listdir("../data/ori_data/{}/condition".format(configg.name))
# if not configg.num_classes == 2:
#     if concat_background == False:
#
#         for whole in ii:
#             single_path = os.path.join(os.path.join(ori_name_path,"tar1"),whole)
#             if not os.path.exists(single_path):
#                 continue
#             concate = np.expand_dims(np.array(Image.open(single_path)),axis=0)
#
#             tag = 0
#
#             for i in range(2,configg.num_classes + 1):
#                 single_path = os.path.join(os.path.join(ori_name_path,"tar{}".format(i)),whole)
#                 if not os.path.exists(single_path):
#                     tag = 1
#                     break
#                 this = Image.open(single_path)
#                 this = np.expand_dims(np.array(this),axis=0)
#                 concate = np.concatenate((concate,this),axis=0)
#
#             if tag == 1:
#                 continue
#
#
#
#
#             #拼接完成，保存为.npy，以一定几率放在"./data/{}/all/train".format(config.name)   "./data/{}/all/val".format(config.name)
#             # rdm = random.uniform(0,10)
#             # if rdm < 2:
#             #     np.save("./data/{}/all/val/label/{}.npy".format(configg.name, whole), concate)
#             #     shutil.copy("../data/ori_data/{}/condition/".format(configg.name) + whole, "./data/{}/all/val/image/".format(configg.name) + whole)
#             #
#             # else:
#             np.save("./data/{}/all/train/label/{}.npy".format(configg.name, whole),concate)
#             # shutil.copy("../data/ori_data/{}/condition/".format(configg.name) + whole,"./data/{}/all/train/image/".format(configg.name) + whole)
#
#         for whole in jj:
#             single_path = os.path.join(os.path.join(ori_name_path,"tar1"),whole)
#             if not os.path.exists(single_path):
#                 continue
#             concate = np.expand_dims(np.array(Image.open(single_path)),axis=0)
#
#             tag = 0
#
#             for i in range(2,configg.num_classes + 1):
#                 single_path = os.path.join(os.path.join(ori_name_path,"tar{}".format(i)),whole)
#                 if not os.path.exists(single_path):
#                     tag = 1
#                     break
#                 this = Image.open(single_path)
#                 this = np.expand_dims(np.array(this),axis=0)
#                 concate = np.concatenate((concate,this),axis=0)
#
#             if tag == 1:
#                 continue
#
#
#
#
#             #拼接完成，保存为.npy，以一定几率放在"./data/{}/all/train".format(config.name)   "./data/{}/all/val".format(config.name)
#             # rdm = random.uniform(0,10)
#             # if rdm < 2:
#             #     np.save("./data/{}/all/val/label/{}.npy".format(configg.name, whole), concate)
#             #     shutil.copy("../data/ori_data/{}/condition/".format(configg.name) + whole, "./data/{}/all/val/image/".format(configg.name) + whole)
#             #
#             # else:
#             np.save("./data/{}/all/val/label/{}.npy".format(configg.name, whole),concate)