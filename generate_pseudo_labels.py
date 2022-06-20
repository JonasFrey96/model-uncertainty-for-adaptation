#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT

import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from datasets import CrossCityDataset, get_val_transforms
from utils import ScoreUpdater, colorize_mask
from datasets import get_val_transforms, ScanNet
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
osp = os.path


def validate_model(model, save_round_eval_path, round_idx, args):
    logger = logging.getLogger('crosscityadap')
    ## Doubles as a pseudo label generator


    output_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ds = CrossCityDataset(root=args.data_tgt_dir, list_path=args.data_tgt_test_list.format(args.city), transforms=transforms)
    ds = ScanNet(
            root="/home/jonfrey/Datasets/scannet",
            mode="train",
            scenes=["scene0000"],
            output_trafo=output_transform,
            output_size=(320, 640),
            degrees=10,
            data_augmentation=True,
            flip_p=0.5,
            jitter_bcsh=[0.3, 0.3, 0.3, 0.05]
        )
    loader = torch.utils.data.DataLoader(ds, batch_size=6, pin_memory=torch.cuda.is_available(), num_workers=6)
    
    
    # val_transforms = get_val_transforms(args)
    # dataset = CrossCityDataset(args.data_tgt_dir,
    #                            args.data_tgt_train_list.format(args.city), transforms=val_transforms)
    # loader = DataLoader(dataset, batch_size=12, num_workers=4, pin_memory=torch.cuda.is_available())
    
    
    

    scorer = ScoreUpdater(args.num_classes, len(loader))

    save_pred_vis_path = osp.join(save_round_eval_path, 'pred_vis')
    save_prob_path = osp.join(save_round_eval_path, 'prob')
    save_pred_path = osp.join(save_round_eval_path, 'pred')
    if not os.path.exists(save_pred_vis_path):
        os.makedirs(save_pred_vis_path)
    if not os.path.exists(save_prob_path):
        os.makedirs(save_prob_path)
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)

    conf_dict = {k: [] for k in range(args.num_classes)}
    pred_cls_num = np.zeros(args.num_classes)
    ## evaluation process
    logger.info('###### Start evaluating target domain train set in round {}! ######'.format(round_idx))
    start_eval = time.time()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image, label, name = batch

            image = image.to(device)
            output = model(image).cpu().softmax(1)

            flipped_out = model(image.flip(-1)).cpu().softmax(1)
            output = 0.5 * (output + flipped_out.flip(-1))

            # image = image.cpu()
            pred_prob, pred_labels = output.max(1)
            # scorer.update(pred_labels.view(-1), label.view(-1))

            for b_ind in range(image.size(0)):
                t = name[b_ind].split('/')
                image_name = t[-3] + "_" + (t[-1].split('.')[0])
                
                np.save('%s/%s.npy' % (save_prob_path, image_name), output[b_ind].numpy().transpose(1, 2, 0))
                if args.debug:
                    colorize_mask(pred_labels[b_ind].numpy().astype(np.uint8)).save(
                        '%s/%s_color.png' % (save_pred_vis_path, image_name))
                Image.fromarray(pred_labels[b_ind].numpy().astype(np.uint8)).save(
                    '%s/%s.png' % (save_pred_path, image_name))

            if args.kc_value == 'conf':
                for idx_cls in range(args.num_classes):
                    idx_temp = pred_labels == idx_cls
                    pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + idx_temp.sum()
                    if idx_temp.any():
                        conf_cls_temp = pred_prob[idx_temp].numpy().astype(np.float32)[::args.ds_rate]
                        conf_dict[idx_cls].extend(conf_cls_temp)
    model.train()
    logger.info('###### Finish evaluating target domain train set in round {}! Time cost: {:.2f} seconds. ######'.format(
        round_idx, time.time() - start_eval))
    return conf_dict, pred_cls_num, save_prob_path, save_pred_path
