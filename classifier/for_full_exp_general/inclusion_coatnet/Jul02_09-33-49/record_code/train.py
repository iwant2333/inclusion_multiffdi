# -*- coding: utf-8 -*-

import os
import shutil
import sys
import argparse
import time
import itertools, random, pandas as pd
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
# from sklearn.metrics import confusion_matrix
# import scikitplot as skplt
import sklearn.metrics
import numpy as np
import warnings,copy
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn import parallel
from torch import distributed as dist

sys.path.append('./')
from utils.util import set_prefix, write, add_prefix, accuracy, AverageMeter
from utils.FocalLoss import *
from utils.label_smooth import *
from utils.large_margin_softmax import *
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss, L1Loss

import dataset_processing
import dataset_transforms
import rfm_utils
from models import *


model_register = {
    'v0_backbone_mae_vit':          Deepfas_v0_backbone_mae_vit,
    'v1_auxinfo_f3netlfs':          Deepfas_v1_auxinfo_f3netlfs,
    'v2_multiclass_f3netlfs':       Deepfas_v2_multiclass_f3netlfs,
    'v2tiny_multiclass_f3netlfs':   Deepfas_v2tiny_multiclass_f3netlfs,
    'inclusion_coatnet':            Deepfas_Inclusion_CoAtNet,
}


def load_model_weights(model, weights_path=None, strict=True):

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['state_dict']
    if "model" in state_dict: 
        state_dict = state_dict["model"]

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    own_state = model.state_dict()
    for name, param in state_dict.items():
        # try to get correct name
        if name in own_state: pass
        elif name.replace('module.','',1) in own_state:
            name = name.replace('module.','',1)  # rename, delete 'module.'
        elif 'module.'+name in own_state:
            name = 'module.'+name  # rename, add 'module.'

        if name in own_state:
            if param.shape == own_state[name].shape:
                new_state_dict[name] = param
            else:
                mycfg.debug_print("Skip for mistaken shape: {}, {}".format(name, param.shape))
        else:
            mycfg.debug_print("Skip for mistaken name: {}, {}".format(name, param.shape))
    model.load_state_dict(new_state_dict, strict=strict)
    return model

def pretrain(model, weights_path):

    state_dict = torch.load(weights_path)
    own_state = model.state_dict()

    for name, param in state_dict.items():
        realname = name.replace('module.','')
        if realname in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[realname].copy_(param)
                # mycfg.debug_print(realname)
            except:
                mycfg.debug_print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(realname, own_state[name].size(), param.size()))
                mycfg.debug_print("But don't worry about it. Continue pretraining.")

    # import ipdb; ipdb.set_trace()
    return model

def compute_validate_meter(model, best_model_path, val_loader):
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    best_acc = checkpoint['best_accuracy']
    mycfg.debug_print('best accuracy={:.4f}'.format(best_acc))
    pred_y = list()
    test_y = list()
    probas_y = list()
    for data, target in val_loader:
        if mycfg.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        probas_y.extend(output.data.cpu().numpy().tolist())
        pred_y.extend(output.data.cpu().max(1, keepdim=True)[1].numpy().flatten().tolist())
        test_y.extend(target.data.cpu().numpy().flatten().tolist())

    # confusion = confusion_matrix(pred_y, test_y)
    # plot_confusion_matrix(confusion, classes=['spoof', 'live'])
    # plt_roc(test_y, probas_y)

def plt_roc(test_y, probas_y, plot_micro=False, plot_macro=False):
    assert isinstance(test_y, list) and isinstance(probas_y, list), 'the type of input must be list'
    skplt.metrics.plot_roc(test_y, probas_y, plot_micro=plot_micro, plot_macro=plot_macro)
    plt.savefig(add_prefix(mycfg.prefix, 'roc_auc_curve.png'))
    plt.close()

def plot_confusion_matrix(cm, classes=[0,1], normalize=False, save_path='cm.jpg', print_flag=True):
    if normalize:
        mycfg.debug_print("Confusion matrix, with normalization:")
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(4)
    else:
        mycfg.debug_print('Confusion matrix, without normalization:')

    if print_flag:
        tmp_cm = cm
        # np.set_printoptions(precision=2, suppress=True, linewidth=20000)
        # np.set_printoptions(precision=4, threshold=np.inf, linewidth=20000)
        np.set_printoptions(threshold=np.inf, linewidth=20000)
        mycfg.debug_print(tmp_cm)
        if not normalize:
            tmp_cm = (tmp_cm.astype('float') / tmp_cm.sum(axis=1)[:, np.newaxis]).round(4)
        for i in range(tmp_cm.shape[0]):
            mycfg.debug_print(f"acc of class_idx={i}: {tmp_cm[i,i]}")
        mycfg.debug_print("\n\n")

    plt.figure(99, figsize=(40,40))  # figsize单位为英寸
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Pred label')
    plt.savefig(save_path)
    plt.close()

def save_checkpoint(state, epoch, acc, tpr, filename='checkpoint.pth.tar'):
    # save training state after each epoch
    save_model_name = '{:d}_acc@1_{:.6f}_tpr@5e-3_{:.6f}_thresh@5e-3_{:.6f}.pth.tar'.format(epoch, acc, tpr['tpr'], tpr['thresh'])
    torch.save(state, add_prefix(mycfg.prefix, save_model_name))
    
    # # import ipdb; ipdb.set_trace()
    # shutil.copyfile(add_prefix(mycfg.prefix, save_model_name), add_prefix(mycfg.prefix, filename))


def stats_tpr_fpr(preds, labels, print_flag=True, draw_flag=True):
    # mycfg.debug_print(preds[:20])
    # mycfg.debug_print(labels[:20])
    # fpr is 好人的误拦截
    # tpr is 坏人的拦截率
    list_fpr, list_tpr, list_thresh = sklearn.metrics.roc_curve(labels, preds, pos_label=None,sample_weight=None,drop_intermediate=True)
    roc_auc = sklearn.metrics.auc(list_fpr, list_tpr)
    # mycfg.debug_print(list_fpr)
    # mycfg.debug_print(list_tpr)
    # mycfg.debug_print(list_thresh)
    # mycfg.debug_print('roc_auc: {}'.format(roc_auc))
    fpr_target = [1e-2, 5e-3, 3e-3, 2e-3, 1e-3, ]
    stat_result = [get_tpr_given_fpr_target(list_fpr, list_tpr, list_thresh, t) for t in fpr_target]
    stat_info = 'Testing/tpr_fpr_1e-2: {}, thresh: {}, fpr: {}'.format(stat_result[0]['tpr'], stat_result[0]['thresh'], stat_result[0]['fpr']) + '\n' + \
                'Testing/tpr_fpr_5e-3: {}, thresh: {}, fpr: {}'.format(stat_result[1]['tpr'], stat_result[1]['thresh'], stat_result[1]['fpr']) + '\n' + \
                'Testing/tpr_fpr_3e-3: {}, thresh: {}, fpr: {}'.format(stat_result[2]['tpr'], stat_result[2]['thresh'], stat_result[2]['fpr']) + '\n' + \
                'Testing/tpr_fpr_2e-3: {}, thresh: {}, fpr: {}'.format(stat_result[3]['tpr'], stat_result[3]['thresh'], stat_result[3]['fpr']) + '\n' + \
                'Testing/tpr_fpr_1e-3: {}, thresh: {}, fpr: {}'.format(stat_result[4]['tpr'], stat_result[4]['thresh'], stat_result[4]['fpr']) + '\n' + \
                'Testing/roc_auc: {}'.format(roc_auc)     + '\n\n\n' 
    if print_flag:
        mycfg.debug_print(stat_info)
    if draw_flag:
        draw_roc(list_fpr, list_tpr, model_name=mycfg.model_name, log_dir=mycfg.prefix)
    return stat_result, stat_info

def get_tpr_given_fpr_target(fpr, tpr, thresh, fpr_target=1e-3):
    index = 0
    for i in range(len(fpr)):
        if fpr[i] > fpr_target:
            index = i
            # print (index, fpr[index])
            if abs(fpr[i-1] - fpr_target) < abs(fpr[i] - fpr_target):
                index = i-1
            # print (index, fpr[index])
            break
    result = {'fpr':fpr[index], 'tpr':tpr[index], 'thresh':thresh[index]}
    # mycfg.debug_print('fpr: {}\t tpr: {}\t thresh: {}'.format(*result.values()))
    return result

def draw_roc(fpr, tpr, fig_num=66, model_name='', log_dir=''):
    plt.figure(fig_num)
    # plt.semilogx(fpr, tpr, 'r', label = model_name)
    # plt.plot([0,1], [0,1], ls='--')  # 正对角线, Diagonal
    # plt.plot([0,1], [1,0], ls='--')  # 副对角线, Subdiagonal
    plt.semilogx(fpr, tpr, label = model_name)
    plt.grid(True)
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.savefig(os.path.join(log_dir, model_name+"__roc.png"))
    return

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt

def calc_metrics(preds, labels, log_dir='', prefix=''):
    num_classes = preds.shape[1]  # N*C, C=2 or 1+4

    from sklearn.preprocessing import label_binarize
    # 形式二：类别原始值（0,1,2,3,4...）
    # 形式三：one-hot值
    y_labels = labels
    # print(num_classes)
    # print(y_labels)
    # import ipdb;ipdb.set_trace()
    y_labels_onehot = label_binarize(y_labels, classes=np.arange(num_classes))
    # 形式一：预测概率值
    # 形式二：各类概率值
    # 形式三：one-hot值
    y_score_preds = preds
    y_score_labels = y_score_preds.argmax(axis=1)
    y_score_onehot = label_binarize(y_score_labels, classes=np.arange(num_classes))
    
    # save test confuse matrix
    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(y_labels, y_score_labels)
    save_path = os.path.join(log_dir, prefix+'__confusion_matrix__{}.png'.format(datetime.now().strftime('%b%d_%H-%M-%S')))
    plot_confusion_matrix(confusion, classes=np.arange(num_classes).tolist(), save_path=save_path, print_flag=True)

    from sklearn.metrics import accuracy_score, precision_score
    from sklearn.metrics import recall_score, f1_score, roc_auc_score
    accuracy  = accuracy_score(y_labels, y_score_labels)  # OA, Overall Accuracy
    # precision = precision_score(y_labels, y_score_labels, average='micro')  # 'micro': Calculate metrics globally by considering each element of the label indicator matrix as a label.
    # recall    = recall_score(y_labels, y_score_labels, average='micro')     # 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    # f1score   = f1_score(y_labels, y_score_labels, average='micro')
    # roc_auc   = roc_auc_score(y_labels_onehot, y_score_onehot, average='micro')
    mycfg.debug_print(prefix+' accuracy: {}'.format(accuracy))
    # mycfg.debug_print(prefix+' precision: {}'.format(precision))
    # mycfg.debug_print(prefix+' recall: {}'.format(recall))
    # mycfg.debug_print(prefix+' f1-score: {}'.format(f1score))
    # mycfg.debug_print(prefix+' roc_auc (onehot): {}\n'.format(roc_auc))

    metric_result = {}  # All evaluation metrics
    metric_result['accuracy']  = accuracy
    # metric_result['precision'] = precision
    # metric_result['recall']    = recall
    # metric_result['f1score']   = f1score
    # metric_result['roc_auc']   = roc_auc
    
    if num_classes>2:  # for multi-classes: overall roc
        preds  = y_score_preds.ravel()  # tensor.ravel() Return a contiguous flattened tensor. 
        labels = y_labels_onehot.ravel()
    else:
        preds  = y_score_preds[:,1]
        labels = y_labels
    
    try:
        roc_result, roc_info = stats_tpr_fpr(preds, labels)
        metric_result['roc_result'] = roc_result
        metric_result['roc_info'] = roc_info
    except: 
        metric_result['roc_result'] = None
        metric_result['roc_info'] = ""

    return metric_result


def get_data_paths_trans():

    img_size = (mycfg.img_size, mycfg.img_size)  # (224, 224)
    train_transforms = dataset_transforms.get_transform(img_size, for_val=False)
    val_transforms = dataset_transforms.get_transform(img_size, for_val=True)
    mycfg.debug_print("train_transforms: \n", train_transforms)
    mycfg.debug_print("val_transforms: \n", val_transforms)
    load_func = None  # use default load_dataset_csv(...) in dataset_processing.py

    ################################################################################################################# my exp
    # data_root = "/home/t4/sankuai.luoman/disk/code/face_liveness/deepfas/dataset"  # 78 or 233
    data_root = "/home/zoloz/8T-2/zhanyi/data/openset_df_competition/phase1/"  # hz01
    train_dataset_path = [  # total: 280w+280w+160w+160w = 880w
        ########################################################## # swap: (75+45+20)w fake + (75+45+20)w real = 140w fake + 140w real = 280w
        #f'{data_root}/s1_20220310/for_deepfas_proj_swap_RFM_Freq_PCL__selected__imglist__select_trainset_fake_75w_real_75w.txt',
        #f'{data_root}/s1_20220310/for_deepfas_proj_swap_RFM_Freq_PCL__selected__imglist__part2_select_trainset_fake_45w_real_45w.txt',
        #f'{data_root}/s1_20220310/for_deepfas_proj_swap_RFM_Freq_PCL__selected__imglist__part3_select_trainset_fake_20w_real_20w.txt',
        # ########################################################## # animation: (70+150)w fake + 60w real = 220w fake + 60w real = 280w
        #f'{data_root}/s1_20220310_oneframe/for_deepfas_proj_animation_oneframe__selected__vidlist__select_trainset_fake_70w_real_0w.txt',
        #f'{data_root}/s1_20220310_oneframe/for_deepfas_proj_animation_oneframe__selected__vidlist__part2_kodf_partial_10%_trainset_synthesized_fake_150w_real_0w.txt',
        # f'{data_root}/s1_20220310_oneframe/for_deepfas_proj_animation_oneframe__selected__vidlist__kodf_partial_10%_trainset_fake_0w_real_60w.txt',
        # ########################################################## # attedit+genface: 80w fake + 80w real = 160w
        #f'{data_root}/s1_20220310/for_deepfas_proj_general_df__selected__imglist__attedit_genface_select_trainset_fake_80w_real_0w.txt',
        #f'{data_root}/s1_20220310_oneframe/for_deepfas_proj_animation_oneframe__selected__imglist__part2_zmng_connect_live_oneframe_130w__rndsel_80w.txt',
        # ########################################################## # zhike liveface: 160w real = 160w
        #f'{data_root}/s1_20220310/for_deepfas_proj_general_df__selected__imglist__real_zhike_liveface_select_trainset_fake_0w_real_160w.txt',
        f'{data_root}/trainset_label.txt',
    ]
    test_dataset_path = [  # total: 120w+32w+80w+28w = 260w
        ########################################################## # swap: (35+15+10)w fake + (35+15+10)w real = 60w fake + 60w real = 120w
        #f'{data_root}/s1_20220310/for_deepfas_proj_swap_RFM_Freq_PCL__selected__imglist__select_valset_fake_35w_real_35w.txt',
        #f'{data_root}/s1_20220310/for_deepfas_proj_swap_RFM_Freq_PCL__selected__imglist__part2_select_valset_fake_15w_real_15w.txt',
        #f'{data_root}/s1_20220310/for_deepfas_proj_swap_RFM_Freq_PCL__selected__imglist__part3_select_valset_fake_10w_real_10w.txt',
        # ########################################################## # animation: (12+18)w fake + 2w real = 30w fake + 2w real = 32w
        #f'{data_root}/s1_20220310_oneframe/for_deepfas_proj_animation_oneframe__selected__vidlist__select_valset_fake_12w_real_0w.txt',
        #f'{data_root}/s1_20220310_oneframe/for_deepfas_proj_animation_oneframe__selected__vidlist__part2_kodf_partial_10%_valset_synthesized_fake_18w_real_0w.txt',
        # f'{data_root}/s1_20220310_oneframe/for_deepfas_proj_animation_oneframe__selected__vidlist__kodf_partial_10%_valset_fake_0w_real_2w.txt',
        # ########################################################## # attedit+genface: 40w fake + 40w real = 80w
        #f'{data_root}/s1_20220310/for_deepfas_proj_general_df__selected__imglist__attedit_genface_select_valset_fake_40w_real_0w.txt',
        #f'{data_root}/s1_20220310_oneframe/for_deepfas_proj_animation_oneframe__selected__imglist__part2_zmng_connect_live_oneframe_130w__rndsel_left_52w.txt',
        # ########################################################## # zhike liveface: 28w real = 28w
        #f'{data_root}/s1_20220310/for_deepfas_proj_general_df__selected__imglist__real_zhike_liveface_select_valset_fake_0w_real_32w.txt',
        f'{data_root}/valset_label.txt',
    ]
    # train_dataset_path.extend([
    # ])
    # test_dataset_path.extend([
    # ])


    # # ################################################################################################################# cvpr2022 challenge exp
    # data_root = "/media/hdd4tb/sankuai/code/face_liveness/deepfas/dataset/s1_20220323__for_cvpr2022_workshop_deepfake_final_export/phase1"  # hz01
    # train_dataset_path = [f'{data_root}/trainset_label.txt',]
    # test_dataset_path = [f'{data_root}/valset_label.txt',]
    # def load_dataset_csv(data_path='', data_ratio=1.0, sep=','):
    #     df = pd.read_csv(data_path, sep=" ", header=None, names=['img_md5sum', 'target'])
    #     df['img_path'] = df['img_md5sum'].apply(lambda x: os.path.join(data_root,data_path.split('/')[-1].split('_')[0],f"{x}.jpg"))
    #     df = df.sample(frac=data_ratio).reset_index(drop=True)
    #     return df
    # load_func = deepcopy(load_dataset_csv)


    save_dir = os.path.join(mycfg.prefix, "record_data")
    os.makedirs(save_dir, exist_ok=True)
    if mycfg.gpu==0: 
        for f in train_dataset_path: os.system(f"cp -rf '{f}' '{save_dir}/' ")
        for f in test_dataset_path: os.system(f"cp -rf '{f}' '{save_dir}/' ")
 
    return train_transforms, val_transforms, train_dataset_path, test_dataset_path, load_func
    

def load_dataset(data_ratio=1.0, data_dir=''):
    mycfg.debug_print('='*40, 'Calling load_dataset()...')

    train_transforms, val_transforms, train_dataset_path, test_dataset_path, load_func = get_data_paths_trans()

    train_dataset = dataset_processing.DatasetProcessing(
        train_dataset_path, phase='Train', transform=train_transforms, data_ratio=data_ratio, data_dir=data_dir, load_func=load_func)
    val_dataset = dataset_processing.DatasetProcessing(
        test_dataset_path, phase='Val', transform=val_transforms, data_ratio=data_ratio, data_dir=data_dir, load_func=load_func)
    mycfg.debug_print('='*40, 'Loaded dataset successfully!!!')
    
    # DP模式下，使用dataset_processing.PrefetchLoader导数据的速度反而变慢???
    train_loader = dataset_processing.PrefetchLoader(
        dataset_processing.DataLoaderX(train_dataset,
                            batch_size=mycfg.batch_size,
                            shuffle=False,
                            sampler=torch.utils.data.distributed.DistributedSampler(train_dataset),
                            num_workers=mycfg.num_workers,
                            # num_workers=8*1,
                            # num_workers=0,
                            pin_memory=True, persistent_workers=True))
    val_loader = dataset_processing.PrefetchLoader(
        dataset_processing.DataLoaderX(val_dataset,
                            batch_size=int(mycfg.batch_size//2),
                            shuffle=False,
                            sampler=None,
                            num_workers=mycfg.num_workers,
                            # num_workers=8*1,
                            # num_workers=0,
                            pin_memory=True, persistent_workers=True))
    return train_loader, val_loader

def load_dataset_RealFakeHalf(data_ratio=1.0, data_dir=''):
    def get_dataloader_RealFakeHalf(dataset_path, batch_size, phase='Train', transform=None, data_ratio=1.0, data_dir='', load_func=None):
        df_real, df_fake = dataset_processing.split_neg_pos_datadf(dataset_path, data_ratio)
        datasetR = dataset_processing.DatasetProcessing(data_df=df_real, phase=phase, transform=transform, data_dir=data_dir, load_func=load_func)
        datasetF = dataset_processing.DatasetProcessing(data_df=df_fake, phase=phase, transform=transform, data_dir=data_dir, load_func=load_func)
        dataloaderR = dataset_processing.DataLoaderX(datasetR,
                                batch_size=batch_size//2,
                                # shuffle=True if phase=='Train' else False,
                                shuffle=False,
                                sampler=torch.utils.data.distributed.DistributedSampler(datasetR),
                                num_workers=8*1,
                                pin_memory=True, persistent_workers=True)
        dataloaderF = dataset_processing.DataLoaderX(datasetF,
                                batch_size=batch_size//2,
                                # shuffle=True if phase=='Train' else False,
                                shuffle=False,
                                sampler=torch.utils.data.distributed.DistributedSampler(datasetF),
                                num_workers=8*1,
                                pin_memory=True, persistent_workers=True)
        return dataset_processing.PrefetchLoader_Two(dataloaderR, dataloaderF)
    mycfg.debug_print('='*40, 'Calling load_dataset_RealFakeHalf()...')
    train_transforms, val_transforms, train_dataset_path, test_dataset_path, load_func = get_data_paths_trans()
    train_loader = get_dataloader_RealFakeHalf(
        train_dataset_path, mycfg.batch_size, 'Train', train_transforms, data_ratio=data_ratio, data_dir=data_dir, load_func=load_func)
    val_loader = dataset_processing.PrefetchLoader( dataset_processing.DataLoaderX(
        dataset_processing.DatasetProcessing(test_dataset_path, 'Val', 
            transform=val_transforms, data_ratio=data_ratio, shuffle=False, data_dir=data_dir, load_func=load_func),
                batch_size=int(mycfg.batch_size//2), shuffle=False, sampler=None, num_workers=8*1, pin_memory=True, persistent_workers=True))
    mycfg.debug_print('='*40, 'Loaded dataset successfully!!!')
    return train_loader, val_loader


def cal_fam(model, inputs, key_list):
    model.eval()
    model.zero_grad()
    for k,v in inputs.items():
        inputs[k] = v.detach().clone()
        inputs[k].requires_grad_()
    
    outputs = model.forward(inputs, phase="only_prediction")

    target = (outputs[:, 1]-outputs[:, 0])
    # target = (outputs['prob'][:, 1]-outputs['prob'][:, 0])
    # if 'depth_map' in outputs: 
    #     target += MSELoss()(outputs['depth_map'],outputs['depth_map'])*0
    target.backward(torch.ones(target.shape).cuda(mycfg.gpu))
    
    fam_list = {}
    for key_item in key_list:
        if key_item not in inputs: continue
        fam = torch.abs(inputs[key_item].grad)
        fam = torch.max(fam, dim=1, keepdim=True)[0]
        fam_list[key_item] = fam
    return fam_list

def run_rfm(model, inputs, key_list=['img_frame1',], pointcnt_max=3):
    if pointcnt_max<=0: return model,inputs
    
    model.eval()
    masks = cal_fam(model, inputs, key_list)

    for key_item,mask in masks.items():
        ''' ↓ the implementation of RFM ↓ '''
        imgmask = torch.ones_like(mask)
        imgh = imgw = mycfg.img_size

        for i in range(len(mask)):
            maxind = np.argsort(mask[i].cpu().numpy().flatten())[::-1]
            pointcnt = 0
            for pointind in maxind:
                pointx = pointind //imgw
                pointy = pointind % imgw

                if imgmask[i][0][pointx][pointy] == 1:
                    eH,eW = 120,120
                    maskh = random.randint(1, eH)
                    maskw = random.randint(1, eW)
                    sh = random.randint(1, maskh)
                    sw = random.randint(1, maskw)

                    top = max(pointx-sh, 0)
                    bot = min(pointx+(maskh-sh), imgh)
                    lef = max(pointy-sw, 0)
                    rig = min(pointy+(maskw-sw), imgw)
                    imgmask[i][:, top:bot, lef:rig] = torch.zeros_like(imgmask[i][:, top:bot, lef:rig])

                    pointcnt += 1
                    if pointcnt >= pointcnt_max: break

        inputs[key_item] = imgmask * inputs[key_item] + (1-imgmask) * (torch.rand_like(inputs[key_item])*2-1.)
        ''' ↑ the implementation of RFM ↑ '''
    
    model.train()
    return model, inputs

def train(model, train_loader, optimizer, scheduler, criterions, epoch, writer_count, writer):
    model.train(True)
    am_acc_top1 = AverageMeter()
    am_acc_top1_aux = AverageMeter()
    am_epoch_time = AverageMeter()
    batch_nums = len(train_loader)
    mycfg.debug_print('-' * 10)
    mycfg.debug_print(f'train batch_nums per process/gpu: {batch_nums}')

    for idx, (data_inputs,data_labels) in enumerate(train_loader):
        # mycfg.debug_print(f'idx={idx}');continue

        inputs = data_inputs['img_inputs']
        labels_class = data_labels['label']
        labels_depth = inputs['img_depth']
        labels_recon = inputs['img_orig1']
        labels_mask  = inputs['img_mask']
        labels_dict = {
            'class': labels_class, 
            'class_aux': data_labels.get('label_multiclass', None), 
            'depth': labels_depth,
            'recon': labels_recon,
            'mask': labels_mask,}

        batch_time = time.time()
        batch_size = len(labels_class)

        # # import ipdb; ipdb.set_trace()
        # mycfg.debug_print(f'idx:{idx}', labels_class[:4], labels_class[-4:])
        # mycfg.debug_print(f'idx:{idx}', data_inputs['img_path'][0][:4], data_inputs['img_path'][1][:4])
        # continue

        if "_rfm_" in mycfg.model_name:  # stage-wise rfm strategies
            pointcnt_max = 3
            # ################################# for 20 epochs
            # if epoch<3: pointcnt_max = 0
            # elif epoch<6: pointcnt_max = 1
            # elif epoch<12: pointcnt_max = 2
            # else: pointcnt_max = 3
            # ################################# for 40 epochs
            if epoch<6: pointcnt_max = 0
            elif epoch<12: pointcnt_max = 1
            elif epoch<24: pointcnt_max = 2
            else: pointcnt_max = 3
            model,inputs = run_rfm(model,inputs,key_list=['img_frame1',],pointcnt_max=pointcnt_max)

        optimizer.zero_grad()  # zero the parameter gradients
        model.zero_grad()      # safer way
        outputs, loss_dict = model.forward(inputs, criterions, labels_dict)  # forward

        # calc loss
        loss = loss_dict['loss_total']
        # # DDP will automatically average the gradients on different GPUs during backpropagation.
        # loss.backward()
        # we extra utilize the loss term in [14] with b = 0.04 to stabilize training.
        # paper: Do we need zero training loss after achieving zero training error?
        flood = (loss-0.04).abs() + 0.04
        # 反向梯度下降
        flood.backward()
        # # 做梯度裁剪，防止梯度爆炸
        # torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)  # 反而会导致acc下降
        # 优化器参数更新
        optimizer.step()

        acc_top1 = None     # binary classification
        acc_top1_aux = None # multi classification
        acc_top1 = accuracy(outputs['prob'], labels_dict['class'], topk=(1,))[0]
        am_acc_top1.update(acc_top1.data.item(), batch_size)
        if outputs.get('prob_aux') is not None:
            acc_top1_aux = accuracy(outputs['prob_aux'], labels_dict['class_aux'], topk=(1,))[0]
            am_acc_top1_aux.update(acc_top1_aux.data.item(), batch_size)

        batch_time = time.time() - batch_time
        epoch_time = batch_nums*batch_time/60/60  # hour
        am_epoch_time.update(epoch_time)

        if idx % mycfg.print_freq == 0 and mycfg.global_rank==0:
            # import ipdb; ipdb.set_trace()
            # mycfg.debug_print("=" * 60 + ' Faclon...')
            curr_lr = float(optimizer.param_groups[0]['lr'])
            print_info  = f'Epoch: {epoch}/{mycfg.epochs} [{idx}/{len(train_loader)} ({100.*idx/len(train_loader):.0f}%)], '
            print_info += f'Acc@1: {am_acc_top1.avg:.4f}, Acc_aux@1: {am_acc_top1_aux.avg:.4f}, '
            print_info += f'LR: {curr_lr:.12f}, Time: {am_epoch_time.avg:.4f}h, '
            
            for k,v in loss_dict.items():
                # v = reduce_tensor(v.data, mycfg.world_size)  # loss reduced
                am_name = 'am_{}'.format(k)         # AverageMeter, 
                if am_name not in locals().keys():  # if non-existed, define now
                    exec('{} = AverageMeter()'.format(am_name))
                am = eval(am_name)
                am.update(v.data.item())
                writer.add_scalar(f"Training/Training_{k}", am.avg, writer_count)
                print_info += f'{k}: {am.avg:.6f}, '
                # print_info += f'{k}: {am.val:.6f}, '
                # print_info += f'{k}: {am.val:.6f} ({am.avg:.6f}), '
            
            mycfg.debug_print(print_info[:-2])
            # import ipdb; ipdb.set_trace()
            writer.add_scalar("Training/Training_Accuracy", am_acc_top1.avg, writer_count)
            writer.add_scalar("Training/Training_Acc_aux", am_acc_top1_aux.avg, writer_count)
            writer.add_scalar("Training/Training_LR", float(optimizer.param_groups[0]['lr']), writer_count)
            writer_count += 1

        if mycfg.scheduler_type == 'cosine':  # lr adjust at each step 
            scheduler.step()
        else:  # lr decay at some epoch
            scheduler.step(epoch)
        
        # import ipdb; ipdb.set_trace()
        # if idx > 5: break
        torch.cuda.synchronize()

    return writer_count, scheduler


def validate(model, val_loader, criterions, epoch, writer_count_test, writer):
    model.eval()
    total_path = None
    total_prob = None
    total_labl = None
    total_prob_aux = None
    total_labl_aux = None

    mycfg.debug_print('-' * 10)
    batch_nums = len(val_loader)
    mycfg.debug_print(f'val batch_nums per process/gpu: {batch_nums}')

    for idx, (data_inputs,data_labels) in tqdm(enumerate(val_loader)):
        inputs = data_inputs['img_inputs']
        img_paths = data_inputs['img_path']
        labels_class = data_labels['label']
        labels_depth = inputs['img_depth']
        labels_recon = inputs['img_orig1']
        labels_mask  = inputs['img_mask']
        labels_dict = {
            'class': labels_class, 
            'class_aux': data_labels.get('label_multiclass', None),
            'depth': labels_depth,
            'recon': labels_recon,
            'mask': labels_mask,}

        outputs, loss_dict = model.forward(inputs, criterions, labels_dict)  # forward
        # continue
        
        prob = F.softmax(outputs['prob'], dim=1)  # classifier pred score
        prob = prob.detach().cpu().numpy()
        labl = labels_class.cpu().numpy()
        # print(prob.shape)  # torch.Size([256, 2]) == (N, classes)
        if outputs.get('prob_aux') is not None:
            prob_aux = F.softmax(outputs['prob_aux'], dim=1)  # aux classifier pred score
            prob_aux = prob_aux.detach().cpu().numpy()
            labl_aux  = labels_dict['class_aux'].cpu().numpy()
        else:
            prob_aux = prob  # if no aux, use main as aux
            labl_aux  = labl
        total_path = img_paths if total_path is None else total_path.extend(img_paths)
        total_prob = prob if total_prob is None else np.concatenate([total_prob, prob], axis=0)
        total_labl = labl if total_labl is None else np.concatenate([total_labl, labl], axis=0)
        total_prob_aux = prob_aux if total_prob_aux is None else np.concatenate([total_prob_aux, prob_aux], axis=0)
        total_labl_aux = labl_aux if total_labl_aux is None else np.concatenate([total_labl_aux, labl_aux], axis=0)

        # import ipdb; ipdb.set_trace()
        # if idx > 5: break
        writer_count_test += 1
        for k,v in loss_dict.items():
            am_name = 'test_{}'.format(k)       # AverageMeter, 
            if am_name not in locals().keys():  # if non-existed, define now
                exec('{} = 0'.format(am_name))
            am = eval(am_name)
            am += v.data.item()
            if mycfg.global_rank==0:
                writer.add_scalar(f"Testing/Testing_{k}", v, writer_count_test)
    

    # ########################################################## test metrics
    # mycfg.debug_print('total_prob: \n', total_prob[:3])
    # mycfg.debug_print('total_labl: \n', total_labl[:3])

    # save pred result
    save_path = os.path.join(mycfg.prefix, f'test_result_epoch_{epoch}.csv')
    save_result = {
        'img_path':    total_path,
        'preds':       total_prob.tolist(),
        'labels':      total_labl.tolist(),
        'preds_aux':   total_prob_aux.tolist(),
        'labels_aux':  total_labl_aux.tolist(),
    }
    save_df = pd.DataFrame.from_dict(save_result, orient='index')
    save_df = save_df.round({'preds': 6, 'preds_aux': 6, })
    save_df.to_csv(save_path, index=False)

    # calc test metrics
    test_result = {
        'preds':total_prob,  'preds_aux':total_prob_aux, 
        'labels':total_labl, 'labels_aux':total_labl_aux, 
    }
    metric_result = calc_metrics(test_result['preds'], test_result['labels'], log_dir=mycfg.prefix, prefix='Binary-class')  # Binary-Class
    metric_result_aux = calc_metrics(test_result['preds_aux'], test_result['labels_aux'], log_dir=mycfg.prefix, prefix='Multi-class')  # Multi-Class

    test_acc = metric_result['accuracy']
    test_acc_aux = metric_result_aux['accuracy']
    stat_info  = f'==> Epoch: {epoch:d}, Test Accuracy: {test_acc:.8f}%, Test Acc_multi: {test_acc_aux:.8f}%\n'
    stat_info += metric_result['roc_info']
    stat_info += metric_result_aux['roc_info']

    if mycfg.global_rank==0:
        stat_path = os.path.join(mycfg.prefix, 'test.log')
        with open(stat_path, 'a+') as f: f.write(stat_info)
        mycfg.debug_print(stat_info)

    return test_acc, metric_result['roc_result'], writer_count_test

def testsave(model, val_loader, criterions, epoch, writer_count_test, writer, suffix=''):
    model.eval()
    mycfg.debug_print('----------')
    batch_nums = len(val_loader)
    mycfg.debug_print(f"testsave batch_nums per process/gpu: {batch_nums}")
    testsave_num = 0
    testsave_imgs = {
        'origin':[],
        'recons':[],
    }

    save_dir = f"{mycfg.prefix}/testvis_{suffix}/recons_epoch{epoch}"
    os.makedirs(save_dir, exist_ok=True)
    for batchidx, (data_inputs, data_labels) in tqdm(enumerate(val_loader)):
        inputs = data_inputs['img_inputs']
        labels_class = data_labels['label']
        labels_depth = inputs['img_depth']
        labels_recon = inputs['img_orig1']
        labels_mask  = inputs['img_mask']
        labels_dict = {
            'class': labels_class, 
            'depth': labels_depth,
            'recon': labels_recon,
            'mask': labels_mask,}

        outputs,_ = model.forward(inputs, criterions, labels_dict)  # forward

        if outputs.get('recon_map') is None: break

        recon_x = labels_dict['recon']
        recon_y = outputs['recon_map']
        for stepidx, (recon1, recon2) in enumerate(zip(recon_x, recon_y)):
            save_x = f"{save_dir}/{batchidx}_{stepidx}_origin.jpg"
            save_y = f"{save_dir}/{batchidx}_{stepidx}_recons.jpg"
            recon1,_ = rfm_utils.tensor2origimg((recon1[None, ...]), save_path=save_x)
            recon2,_ = rfm_utils.tensor2origimg((recon2[None, ...]), save_path=save_y)
            testsave_num += 1
            testsave_imgs['origin'].append(recon1)
            testsave_imgs['recons'].append(recon2)
            
        if testsave_num >= 20:
            for k,v in testsave_imgs.items(): 
                rfm_utils.vis_grid_image(v,save_path=f"{save_dir}_{k}.jpg")
            break

    # exit(0)


def train_worker(cfg):  # trian & validate
    global mycfg
    mycfg = cfg
    
    best_acc, best_tpr = 0.0, 0.0
    writer_count, writer_count_test = 0,0
    writer = SummaryWriter(mycfg.prefix) # writer for buffering intermedium results

    model = model_register[mycfg.model_name](mycfg)  # Falcon, Falcon_FFT
    resume_path = mycfg.resume_path if mycfg.get('resume_path') is not None else ""
    if os.path.exists(resume_path):
        model = load_model_weights(model, resume_path, strict=False)
        # model = pretrain(model, resume_path)
        mycfg.debug_print('finetune from best model....')
        mycfg.debug_print('resume_path: ', resume_path)
    model = model.cuda(mycfg.gpu)  # to GPU before define the self.optimizer

    # diff lr for network parameters: 
    params = dict(model.named_parameters())
    params_new = []
    for k, v in params.items():
        if not v.requires_grad: continue  # 可能会固定部分参数只训练另一部分     
        # factor base_lr for some new layers
        if k.startswith('lr_factor__'): params_new += [{'params': [v], 'lr': mycfg.lr * mycfg.lr_factor}]
        else: params_new += [{'params': [v], 'lr': mycfg.lr}]

    criterions = {}
    if mycfg.is_focal_loss:  # focal loss
        mycfg.debug_print('using FocalLoss')
        # criterion_class = FocalLoss(alpha=0.25, gamma=2)
        criterion_class = FocalLossWithLabelSmooth(alpha=0.25, gamma=2, eps=0.1)
    else:
        # cross entropy loss
        # mycfg.debug_print('using CrossEntropyLoss')
        # criterion_class = CrossEntropyLoss()
        mycfg.debug_print('using LargeMarginSoftmaxV1') 
        # criterion_class = LabelSmoothSoftmaxCEV1()
        # criterion_class = LargeMarginSoftmaxV1(lam=0.3)
        criterion_class = LargeMarginSoftmaxV1(lam=0.2)
    # cls loss
    criterions['class'] = copy.deepcopy(criterion_class)
    criterions['class_aux'] = copy.deepcopy(criterion_class)
    # depth map mse loss
    criterions['depth'] = MSELoss()
    criterions['recon'] = MSELoss()
    # pcl loss
    criterions['pcl_ce'] = nn.CrossEntropyLoss()
    criterions['pcl_bce'] = nn.BCELoss()
    for k,v in criterions.items(): criterions[k] = v.cuda(mycfg.gpu)

    if "_rfm_" in mycfg.model_name: 
        train_loader, val_loader = load_dataset_RealFakeHalf(mycfg.data_ratio, mycfg.data_dir)
    else: 
        train_loader, val_loader = load_dataset(mycfg.data_ratio, mycfg.data_dir)

    # optimizer = optim.Adam(model.parameters(), lr=mycfg.lr, betas=(0.9, 0.999))
    # optimizer = optim.Adam(model.parameters(), lr=mycfg.lr, betas=(0.99, 0.9999))
    optimizer = optim.SGD(model.parameters(), lr=mycfg.lr, weight_decay=1e-4, momentum=0.9)
    
    if mycfg.scheduler_type == 'cosine':  # lr adjust at each step 
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*mycfg.epochs, eta_min=mycfg.cosine_lr_end)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=mycfg.step_size, gamma=mycfg.gamma)  # decay=0.2 per step_size

    # 分布式中，对于有BN的模型还可以采用同步BN获取更好的效果，在DDP之前调用
    mycfg.debug_print("Using torch.nn.SyncBatchNorm.convert_sync_batchnorm")
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  
    mycfg.debug_print("Using torch.nn.parallel.DistributedDataParallel")
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[mycfg.gpu])
    model = parallel.DistributedDataParallel(model, device_ids=[mycfg.gpu], find_unused_parameters=True)

    
    since = time.time()
    mycfg.debug_print('-' * 10)
    for epoch in range(mycfg.epochs):

        if mycfg.test_first:
            mycfg.debug_print("test first...")
            mycfg.test_first = False
            test_acc, test_tpr, writer_count_test = validate(model, val_loader, criterions, epoch, writer_count_test, writer)
            # testsave(model, val_loader, criterions, epoch, writer_count_test, writer, suffix='val_loader')
            # testsave(model, train_loader, criterions, epoch, writer_count_test, writer, suffix='train_loader')

        writer_count, scheduler = train(model, train_loader, optimizer, scheduler, criterions, epoch, writer_count, writer)
        # if (epoch+1)%2==0: continue  # validate per 2 epochs, save time
        if (epoch+1)%2!=0: continue  # validate per 5 epochs, save time

        test_acc, test_tpr, writer_count_test = validate(model, val_loader, criterions, epoch, writer_count_test, writer)
        # testsave(model, val_loader, criterions, epoch, writer_count_test, writer, suffix='val_loader')
        # # testsave(model, train_loader, criterions, epoch, writer_count_test, writer, suffix='train_loader')
        # continue 

        if mycfg.global_rank==0:
            curr_acc = test_acc
            curr_tpr = test_tpr[1]['tpr']
            state = {
                'epoch': epoch,
                'arch': mycfg.model_name,
                'state_dict': model.state_dict(),
                'best_accuracy': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            # save training state after each epoch
            torch.save(state, add_prefix(mycfg.prefix, 'latest_checkpoint.pth.tar'))

            is_best = (curr_acc > best_acc) or (curr_tpr > best_tpr)
            if is_best:
                best_acc = max(curr_acc, best_acc)
                best_tpr = max(curr_tpr, best_tpr)
                save_checkpoint(state, epoch, curr_acc, test_tpr[1])  # tpr,thresh @fpr=1e-5

            # upload pth & log to oss
            src_dir = mycfg.prefix
            dst_dir = os.path.join("oss://tipalgog/sankuai/code_backup_from_all_servers/code/face_liveness/deepfas/deepfas_proj/general/general_df/", mycfg.prefix)
            os.system("cp -rf '{}' '{}'".format(mycfg.log_name, src_dir))
            os.system("ossutil64 cp -rf '{}' '{}'".format(src_dir, dst_dir))

    time_elapsed = time.time() - since
    mycfg.debug_print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# if __name__ == '__main__':
#     train_worker()

