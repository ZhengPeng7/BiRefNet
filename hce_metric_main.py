## hce_metric.py
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.morphology import skeletonize
from skimage.morphology import erosion, dilation, disk
from skimage.measure import label

import os
import sys
from tqdm import tqdm
from glob import glob
import pickle as pkl

def filter_bdy_cond(bdy_, mask, cond):

    cond = cv.dilate(cond.astype(np.uint8),disk(1))
    labels = label(mask) # find the connected regions
    lbls = np.unique(labels) # the indices of the connected regions
    indep = np.ones(lbls.shape[0]) # the label of each connected regions
    indep[0] = 0 # 0 indicate the background region

    boundaries = []
    h,w = cond.shape[0:2]
    ind_map = np.zeros((h,w))
    indep_cnt = 0

    for i in range(0,len(bdy_)):
        tmp_bdies = []
        tmp_bdy = []
        for j in range(0,bdy_[i].shape[0]):
            r, c = bdy_[i][j,0,1],bdy_[i][j,0,0]

            if(np.sum(cond[r,c])==0 or ind_map[r,c]!=0):
                if(len(tmp_bdy)>0):
                    tmp_bdies.append(tmp_bdy)
                    tmp_bdy = []
                continue
            tmp_bdy.append([c,r])
            ind_map[r,c] =  ind_map[r,c] + 1
            indep[labels[r,c]] = 0 # indicates part of the boundary of this region needs human correction
        if(len(tmp_bdy)>0):
            tmp_bdies.append(tmp_bdy)

        # check if the first and the last boundaries are connected
        # if yes, invert the first boundary and attach it after the last boundary
        if(len(tmp_bdies)>1):
            first_x, first_y = tmp_bdies[0][0]
            last_x, last_y = tmp_bdies[-1][-1]
            if((abs(first_x-last_x)==1 and first_y==last_y) or
               (first_x==last_x and abs(first_y-last_y)==1) or
               (abs(first_x-last_x)==1 and abs(first_y-last_y)==1)
              ):
                tmp_bdies[-1].extend(tmp_bdies[0][::-1])
                del tmp_bdies[0]

        for k in range(0,len(tmp_bdies)):
            tmp_bdies[k] =  np.array(tmp_bdies[k])[:,np.newaxis,:]
        if(len(tmp_bdies)>0):
            boundaries.extend(tmp_bdies)

    return boundaries, np.sum(indep)

# this function approximate each boundary by DP algorithm
# https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
def approximate_RDP(boundaries,epsilon=1.0):

    boundaries_ = []
    boundaries_len_ = []
    pixel_cnt_ = 0

    # polygon approximate of each boundary
    for i in range(0,len(boundaries)):
        boundaries_.append(cv.approxPolyDP(boundaries[i],epsilon,False))

    # count the control points number of each boundary and the total control points number of all the boundaries
    for i in range(0,len(boundaries_)):
        boundaries_len_.append(len(boundaries_[i]))
        pixel_cnt_ = pixel_cnt_ + len(boundaries_[i])

    return boundaries_, boundaries_len_, pixel_cnt_


def relax_HCE(gt, rs, gt_ske, relax=5, epsilon=2.0):
    # print("max(gt_ske): ", np.amax(gt_ske))
    # gt_ske = gt_ske>128
    # print("max(gt_ske): ", np.amax(gt_ske))

    # Binarize gt
    if(len(gt.shape)>2):
        gt = gt[:,:,0]

    epsilon_gt = 128#(np.amin(gt)+np.amax(gt))/2.0
    gt = (gt>epsilon_gt).astype(np.uint8)

    # Binarize rs
    if(len(rs.shape)>2):
        rs = rs[:,:,0]
    epsilon_rs = 128#(np.amin(rs)+np.amax(rs))/2.0
    rs = (rs>epsilon_rs).astype(np.uint8)

    Union = np.logical_or(gt,rs)
    TP = np.logical_and(gt,rs)
    FP = rs - TP
    FN = gt - TP

    # relax the Union of gt and rs
    Union_erode = Union.copy()
    Union_erode = cv.erode(Union_erode.astype(np.uint8),disk(1),iterations=relax)

    # --- get the relaxed False Positive regions for computing the human efforts in correcting them ---
    FP_ = np.logical_and(FP,Union_erode) # get the relaxed FP
    for i in range(0,relax):
        FP_ = cv.dilate(FP_.astype(np.uint8),disk(1))
        FP_ = np.logical_and(FP_, 1-np.logical_or(TP,FN))
    FP_ = np.logical_and(FP, FP_)

    # --- get the relaxed False Negative regions for computing the human efforts in correcting them ---
    FN_ = np.logical_and(FN,Union_erode) # preserve the structural components of FN
    ## recover the FN, where pixels are not close to the TP borders
    for i in range(0,relax):
        FN_ = cv.dilate(FN_.astype(np.uint8),disk(1))
        FN_ = np.logical_and(FN_,1-np.logical_or(TP,FP))
    FN_ = np.logical_and(FN,FN_)
    FN_ = np.logical_or(FN_, np.logical_xor(gt_ske,np.logical_and(TP,gt_ske))) # preserve the structural components of FN

    ## 2. =============Find exact polygon control points and independent regions==============
    ## find contours from FP_
    ctrs_FP, hier_FP = cv.findContours(FP_.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    ## find control points and independent regions for human correction
    bdies_FP, indep_cnt_FP = filter_bdy_cond(ctrs_FP, FP_, np.logical_or(TP,FN_))
    ## find contours from FN_
    ctrs_FN, hier_FN = cv.findContours(FN_.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    ## find control points and independent regions for human correction
    bdies_FN, indep_cnt_FN = filter_bdy_cond(ctrs_FN, FN_, 1-np.logical_or(np.logical_or(TP,FP_),FN_))

    poly_FP, poly_FP_len, poly_FP_point_cnt = approximate_RDP(bdies_FP,epsilon=epsilon)
    poly_FN, poly_FN_len, poly_FN_point_cnt = approximate_RDP(bdies_FN,epsilon=epsilon)

    return poly_FP_point_cnt, indep_cnt_FP, poly_FN_point_cnt, indep_cnt_FN

def compute_hce(pred_root,gt_root,gt_ske_root):

    gt_name_list = glob(pred_root+'/*.png')
    gt_name_list = sorted([x.split('/')[-1] for x in gt_name_list])

    hces = []
    for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
        gt_path = os.path.join(gt_root, gt_name)
        pred_path = os.path.join(pred_root, gt_name)

        gt = cv.imread(gt_path, cv.IMREAD_GRAYSCALE)
        pred = cv.imread(pred_path, cv.IMREAD_GRAYSCALE)

        ske_path = os.path.join(gt_ske_root,gt_name)
        if os.path.exists(ske_path):
            ske = cv.imread(ske_path,cv.IMREAD_GRAYSCALE)
            ske = ske>128
        else:
            ske = skeletonize(gt>128)

        FP_points, FP_indep, FN_points, FN_indep = relax_HCE(gt, pred,ske)
        print(gt_path.split('/')[-1],FP_points, FP_indep, FN_points, FN_indep)
        hces.append([FP_points, FP_indep, FN_points, FN_indep, FP_points+FP_indep+FN_points+FN_indep])

    hce_metric ={'names': gt_name_list,
                 'hces': hces}


    file_metric = open(pred_root+'/hce_metric.pkl','wb')
    pkl.dump(hce_metric,file_metric)
    # file_metrics.write(cmn_metrics)
    file_metric.close()

    return np.mean(np.array(hces)[:,-1])

def main():

    gt_root = "../DIS5K/DIS-VD/gt"
    gt_ske_root = ""
    pred_root = "../Results/isnet(ours)/DIS-VD"

    print("The average HCE metric: ", compute_hce(pred_root,gt_root,gt_ske_root))


if __name__ == '__main__':
    main()
