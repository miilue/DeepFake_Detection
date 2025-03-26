import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import random

SIZE = 256

preprocess = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor()
])


# 读取单个图片并进行preprocess操作
def val_loader(path):
    img_pil = Image.open(path)  # PIL.Image.open()专接图片路径，用来直接读取该路径指向的图片。要求路径必须指明到哪张图，不能只是所有图所在的文件夹
    img_tensor = preprocess(img_pil)
    return img_tensor


# 数据增广，与RandomResizedCrop()有细微差别
def default_loader(path, new_startx, new_starty, HW):
    img_pil = Image.open(path)
    # if np.random.randint(1, 4) % 3 == 0:
    #     if np.random.randint(1, 7) == 1:
    #         least = np.random.randint(48, 160)
    #         img_pil = img_pil.resize((least, least), Image.ANTIALIAS)
    #     if np.random.randint(1, 7) == 1:
    #         least = np.random.randint(48, 160)
    #         img_pil = img_pil.resize((least, least), Image.NEAREST)
    #     if np.random.randint(1, 7) == 1:
    #         least = np.random.randint(48, 160)
    #         img_pil = img_pil.resize((least, least), Image.BILINEAR)
    #     if np.random.randint(1, 7) == 1:
    #         least = np.random.randint(48, 160)
    #         img_pil = img_pil.resize((least, least), Image.BICUBIC)

    img_pil = img_pil.crop([new_startx, new_starty, new_startx + HW, new_starty + HW])
    img_tensor = preprocess(img_pil)
    return img_tensor


class DealDataset(Dataset):
    '''
    一般形式
    class trainset(Dataset):
        def __init__(self, loader=default_loader):
            self.images = file_train
            self.target = number_train
            self.loader = loader

        def __getitem__(self, index):
            fn = self.images[index]
            img = self.loader(fn)
            target = self.target[index]
            return img,target

        def __len__(self):
            return len(self.images)
    '''
    def __init__(self, LENGTH, TYPE, loader=default_loader):
        self.len = LENGTH
        self.loader = loader

        self.train_fake_imgs = []  # 所有伪造图片路径
        df_fake_root = r'E:\data\deepfakeface\00-Datasets\00-FF++\Deepfakes\raw\train'
        df_train_fake_video_paths = os.listdir(df_fake_root)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表
        for i in df_train_fake_video_paths:
            video_path = df_fake_root + '/' + i  # 文件路径
            img = os.listdir(video_path)  # 图片路径
            self.train_fake_imgs.append([video_path + '/' + j for j in img])

        # f2f_fake_root = r'E:\data\deepfakeface\00-Datasets\00-FF++\Face2Face\raw\train'
        # f2f_train_fake_video_paths = os.listdir(f2f_fake_root)
        # for i in f2f_train_fake_video_paths:
        #     video_path = f2f_fake_root + '/' + i
        #     img = os.listdir(video_path)
        #     self.train_fake_imgs.append([video_path + '/' + j for j in img])

        # fs_fake_root = r'E:\data\deepfakeface\00-Datasets\00-FF++\FaceSwap\raw\train'
        # fs_train_fake_video_paths = os.listdir(fs_fake_root)
        # for i in fs_train_fake_video_paths:
        #     video_path = fs_fake_root + '/' + i
        #     img = os.listdir(video_path)
        #     self.train_fake_imgs.append([video_path + '/' + j for j in img])

        # fsf_fake_root = r'E:\data\deepfakeface\00-Datasets\00-FF++\FaceShifter\raw\train'
        # fsf_train_fake_video_paths = os.listdir(fsf_fake_root)
        # for i in fsf_train_fake_video_paths:
        #     video_path = fsf_fake_root + '/' + i
        #     img = os.listdir(video_path)
        #     self.train_fake_imgs.append([video_path + '/' + j for j in img])

        # nt_fake_root = r'/data2/Jianwei-Fei/00-Dataset/01-Images/00-FF++/NeuralTextures/raw/train/'
        # nt_train_fake_video_paths = os.listdir(nt_fake_root)
        # for i in nt_train_fake_video_paths:
        #     video_path = nt_fake_root + '/' + i
        #     img = os.listdir(video_path)
        #     self.train_fake_imgs.append([video_path + '/' + j for j in img])

        real_root = r'E:\data\deepfakeface\00-Datasets\00-FF++\Real\raw\train'
        train_real_video_paths = os.listdir(real_root)
        self.train_real_imgs = []
        for i in train_real_video_paths:
            video_path = real_root + '/' + i
            img = os.listdir(video_path)
            self.train_real_imgs.append([video_path + '/' + j for j in img])

        self.NUM_fake = len(self.train_fake_imgs)
        self.NUM_real = len(self.train_real_imgs)

    def __getitem__(self, index):  # DealDataset[index]
        if np.random.randint(0, 2):
            video_index = np.random.randint(0, self.NUM_fake)
            img_index = np.random.randint(0, len(self.train_fake_imgs[video_index]))
            img_path = self.train_fake_imgs[video_index][img_index]

            # str.replace(old, new[, max])
            # 把字符串中的 old（旧字符串） 替换成 new(新字符串)，如果指定第三个参数max，则替换不超过 max 次
            # mask_path = img_path.replace('raw', 'mask')  # 把图片路径中的raw替换成mask

            # fake_mask = cv2.imread(mask_path, 0)  # flag = 0，8位深度，1通道
            _ = cv2.imread(img_path, 0).shape[0]

            if np.random.randint(1, 4) % 3 == 0:
                new_startx = random.randint(0, int(0.2 * _))
                new_starty = random.randint(0, int(0.2 * _))
                HW = random.randint(int(0.7 * _), _ - max(new_startx, new_starty) - 1)
            else:
                new_startx, new_starty, HW = 0, 0, _

            img = self.loader(img_path, new_startx, new_starty, HW)

            label = torch.tensor([0])

        else:
            video_index = np.random.randint(0, self.NUM_real)
            img_index = np.random.randint(0, len(self.train_real_imgs[video_index]))
            img_path = self.train_real_imgs[video_index][img_index]
            _ = cv2.imread(img_path, 0).shape[0]

            if np.random.randint(1, 4) % 3 == 0:
                new_startx = random.randint(0, int(0.2 * _))
                new_starty = random.randint(0, int(0.2 * _))
                HW = random.randint(int(0.7 * _), _ - max(new_startx, new_starty) - 1)
            else:
                new_startx, new_starty, HW = 0, 0, _

            img = self.loader(img_path, new_startx, new_starty, HW)

            label = torch.tensor([1])

        return img, label
        # ,mask_up2, mask_up_bank2, mask_left2, mask_left_bank2)

    def __len__(self):
        return self.len


def getDataset(VAL_REAL_ROOT, VAL_FAKE_ROOT):
    real_root = VAL_REAL_ROOT
    test_real_video_paths = os.listdir(real_root)
    test_real_imgs = []
    for i in test_real_video_paths:
        video_path = real_root + '/' + i
        img = os.listdir(video_path)
        test_real_imgs.append([video_path + '/' + j for j in img])

    fake_root = VAL_FAKE_ROOT
    test_fake_video_paths = os.listdir(fake_root)
    test_fake_imgs = []
    for i in test_fake_video_paths:
        video_path = fake_root + '/' + i
        img = os.listdir(video_path)
        test_fake_imgs.append([video_path + '/' + j for j in img])

    NUM_fake = len(test_fake_imgs)
    NUM_real = len(test_real_imgs)
    return NUM_fake, NUM_real, test_fake_imgs, test_real_imgs


# def getValdata(size, NUM_fake, NUM_real, test_fake_imgs, test_real_imgs):
#     imgs = []
#     labels = []
#     for i in range(size):
#         if np.random.randint(0, 2):
#             video_index = np.random.randint(0, NUM_fake)
#             img_index = np.random.randint(0, len(test_fake_imgs[video_index]))
#             img_path = test_fake_imgs[video_index][img_index]
#             img = val_loader(img_path)
#             imgs.append(img)
#             labels.append(0)
#         else:
#             video_index = np.random.randint(0, NUM_real)
#             img_index = np.random.randint(0, len(test_real_imgs[video_index]))
#             img_path = test_real_imgs[video_index][img_index]
#             img = val_loader(img_path)
#             imgs.append(img)
#             labels.append(1)
#
#     return torch.stack(imgs, dim=0), labels


def getValdata(video_index, NUM_fake, NUM_real, test_fake_imgs, test_real_imgs):
    imgs = []
    labels = []
    for img_index in range(len(test_fake_imgs[video_index])):
        img_path = test_fake_imgs[video_index][img_index]
        img = val_loader(img_path)
        imgs.append(img)
        labels.append(0)

    for img_index in range(len(test_real_imgs[video_index])):
        img_path = test_real_imgs[video_index][img_index]
        img = val_loader(img_path)
        imgs.append(img)
        labels.append(1)

    return torch.stack(imgs, dim=0), labels


def calcAUC_byProb(labels, probs):
    N = 0
    P = 0
    neg_prob = []
    pos_prob = []

    for _, i in enumerate(labels):
        if (i == 1):
            P += 1
            pos_prob.append(probs[_])
        else:
            N += 1
            neg_prob.append(probs[_])
    number = 0
    for pos in pos_prob:
        for neg in neg_prob:
            if (pos > neg):
                number += 1
            elif (pos == neg):
                number += 0.5
    return number / (N * P)


def calcACC_byProb(ret_labels, lb_hist):
    P = 0
    r_lb_hist = []
    for i in range(len(lb_hist)):
        if lb_hist[i] >= 0.5:
            r_lb_hist.append(1)
        else:
            r_lb_hist.append(0)

    for i in range(len(ret_labels)):
        if ret_labels[i] == r_lb_hist[i]:
            P += 1

    acc = P / len(ret_labels)
    return acc


def Val(model, VAL_FAKE_ROOT, VAL_REAL_ROOT, ACC=False):
    model.eval()
    with torch.no_grad():
        NUM_fake, NUM_real, test_fake_imgs, test_real_imgs = getDataset(VAL_REAL_ROOT, VAL_FAKE_ROOT)

        lb_hist = []
        ret_labels = []
        NUM = min(NUM_fake, NUM_real)
        for video_index in range(NUM):
            inputs, label = getValdata(video_index, NUM_fake, NUM_real, test_fake_imgs, test_real_imgs)
            input = inputs.cuda()
            pred = model(input)

            # up = torch.sigmoid(output1).detach().cpu().numpy()[:, :, :-1, 1:-1]
            # down = torch.sigmoid(output1).detach().cpu().numpy()[:, :, 1:, 1:-1]
            # left = torch.sigmoid(output3).detach().cpu().numpy()[:, :, 1:-1, :-1]
            # right = torch.sigmoid(output3).detach().cpu().numpy()[:, :, 1:-1, 1:]
            #
            # up_bank = torch.sigmoid(output2).detach().cpu().numpy()[:, :, 1:-1, 2:-2]
            # down_bank = torch.sigmoid(output2).detach().cpu().numpy()[:, :, 1:-1, 2:-2]
            # left_bank = torch.sigmoid(output4).detach().cpu().numpy()[:, :, 2:-2, 1:-1]
            # right_bank = torch.sigmoid(output4).detach().cpu().numpy()[:, :, 2:-2, 1:-1]

            # sim_map = np.mean(np.concatenate((up, down, left, right, up_bank, down_bank, left_bank, right_bank), axis=1),
            #                   axis=(1, 2, 3))
            # batch_sim_map_avg = list(sim_map)

            # ret_hist += batch_sim_map_avg
            # prd = np.squeeze(np.array(_) > 0.5)
            lb_hist += list(pred.detach().cpu().numpy())
            ret_labels += label
        auc = calcAUC_byProb(ret_labels, lb_hist)

        # acc, th = findthrehold(ret_hist, ret_labels)
        # print('Threshold:', np.round(th, 3), 'Accuracy:', np.round(acc * 100, 2), 'AUC:', np.round(auc, 4))
        print('AUC:', np.round(auc, 4))

        acc = 0
        if ACC:
            acc = calcACC_byProb(ret_labels, lb_hist)

    return auc, acc


# def Val(model, VAL_FAKE_ROOT, VAL_REAL_ROOT, ACC=False):
#     model.eval()
#     with torch.no_grad():
#         NUM_fake, NUM_real, test_fake_imgs, test_real_imgs = getDataset(VAL_REAL_ROOT, VAL_FAKE_ROOT)
#
#         lb_hist = []
#         ret_labels = []
#         for i in range(40):
#             inputs, label = getValdata(32, NUM_fake, NUM_real, test_fake_imgs, test_real_imgs)
#             input = inputs.cuda()
#             pred = model(input)
#
#             # up = torch.sigmoid(output1).detach().cpu().numpy()[:, :, :-1, 1:-1]
#             # down = torch.sigmoid(output1).detach().cpu().numpy()[:, :, 1:, 1:-1]
#             # left = torch.sigmoid(output3).detach().cpu().numpy()[:, :, 1:-1, :-1]
#             # right = torch.sigmoid(output3).detach().cpu().numpy()[:, :, 1:-1, 1:]
#             #
#             # up_bank = torch.sigmoid(output2).detach().cpu().numpy()[:, :, 1:-1, 2:-2]
#             # down_bank = torch.sigmoid(output2).detach().cpu().numpy()[:, :, 1:-1, 2:-2]
#             # left_bank = torch.sigmoid(output4).detach().cpu().numpy()[:, :, 2:-2, 1:-1]
#             # right_bank = torch.sigmoid(output4).detach().cpu().numpy()[:, :, 2:-2, 1:-1]
#
#             # sim_map = np.mean(np.concatenate((up, down, left, right, up_bank, down_bank, left_bank, right_bank), axis=1),
#             #                   axis=(1, 2, 3))
#             # batch_sim_map_avg = list(sim_map)
#
#             # ret_hist += batch_sim_map_avg
#             # prd = np.squeeze(np.array(_) > 0.5)
#             lb_hist += list(pred.detach().cpu().numpy())
#             ret_labels += label
#         auc = calcAUC_byProb(ret_labels, lb_hist)
#
#         # acc, th = findthrehold(ret_hist, ret_labels)
#         # print('Threshold:', np.round(th, 3), 'Accuracy:', np.round(acc * 100, 2), 'AUC:', np.round(auc, 4))
#         print('AUC:', np.round(auc, 4))
#
#         acc = 0
#         if ACC:
#             acc = calcACC_byProb(ret_labels, lb_hist)
#
#     return auc, acc
