import os
import cv2
import argparse
import prettytable as pt
import numpy as np
import torch

import evaluation.metrics as Measure


def evaluator_online(gt_ary, pred_ary):
    # define measures
    EM = Measure.Emeasure()
    SM = Measure.Smeasure()
    FM = Measure.Fmeasure()
    MAE = Measure.MAE()
    WFM = Measure.WeightedFmeasure()
    if torch.is_tensor(gt_ary):
        gt_ary = gt_ary.squeeze().detach().cpu().numpy().astype(np.uint8)
        if len(gt_ary.shape) == 2:
            gt_ary = [gt_ary]
    if torch.is_tensor(pred_ary):
        pred_ary = pred_ary.squeeze().detach().cpu().numpy().astype(np.uint8)
        if len(pred_ary.shape) == 2:
            pred_ary = [pred_ary]

    if isinstance(gt_ary, list) and isinstance(pred_ary, list):
        assert len(gt_ary) == len(pred_ary)

    for idx, (gt_ary, pred_ary) in enumerate(zip(gt_ary, pred_ary)):
        if pred_ary.shape != gt_ary.shape:
            pred_ary = cv2.resize(pred_ary, (gt_ary.shape[1], gt_ary.shape[0]))

        EM.step(pred=pred_ary, gt=gt_ary)
        SM.step(pred=pred_ary, gt=gt_ary)
        FM.step(pred=pred_ary, gt=gt_ary)
        MAE.step(pred=pred_ary, gt=gt_ary)
        WFM.step(pred=pred_ary, gt=gt_ary)

    em = EM.get_results()['em']
    sm = SM.get_results()['sm']
    fm = FM.get_results()['fm']
    mae = MAE.get_results()['mae']
    wfm = WFM.get_results()['wfm']

    return em, sm, fm, mae, wfm


def evaluator_online_S(gt_ary, pred_ary):
    # define measures
    # EM = Measure.Emeasure()
    SM = Measure.Smeasure()
    # FM = Measure.Fmeasure()
    MAE = Measure.MAE()
    # WFM = Measure.WeightedFmeasure()
    if torch.is_tensor(gt_ary):
        gt_ary = gt_ary.squeeze().detach().cpu().numpy().astype(np.uint8)
        if len(gt_ary.shape) == 2:
            gt_ary = [gt_ary]
    if torch.is_tensor(pred_ary):
        pred_ary = pred_ary.squeeze().detach().cpu().numpy().astype(np.uint8)
        if len(pred_ary.shape) == 2:
            pred_ary = [pred_ary]

    if isinstance(gt_ary, list) and isinstance(pred_ary, list):
        assert len(gt_ary) == len(pred_ary)

    for idx, (gt_ary, pred_ary) in enumerate(zip(gt_ary, pred_ary)):
        if pred_ary.shape != gt_ary.shape:
            pred_ary = cv2.resize(pred_ary, (gt_ary.shape[1], gt_ary.shape[0]))

        # EM.step(pred=pred_ary, gt=gt_ary)
        SM.step(pred=pred_ary, gt=gt_ary)
        # FM.step(pred=pred_ary, gt=gt_ary)
        MAE.step(pred=pred_ary, gt=gt_ary)
        # WFM.step(pred=pred_ary, gt=gt_ary)

    # em = EM.get_results()['em']
    sm = SM.get_results()['sm']
    # fm = FM.get_results()['fm']
    mae = MAE.get_results()['mae']
    # wfm = WFM.get_results()['wfm']

    return sm, mae
