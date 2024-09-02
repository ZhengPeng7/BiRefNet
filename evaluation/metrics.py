import os
from tqdm import tqdm
import cv2
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt as bwdist
from skimage.morphology import skeletonize
from skimage.morphology import disk
from skimage.measure import label


_EPS = np.spacing(1)
_TYPE = np.float64


def evaluator(gt_paths, pred_paths, metrics=['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'HCE'], verbose=False):
    # define measures
    if 'E' in metrics:
        EM = EMeasure()
    if 'S' in metrics:
        SM = SMeasure()
    if 'F' in metrics:
        FM = FMeasure()
    if 'MAE' in metrics:
        MAE = MAEMeasure()
    if 'WF' in metrics:
        WFM = WeightedFMeasure()
    if 'HCE' in metrics:
        HCE = HCEMeasure()
    if 'MBA' in metrics:
        MBA = MBAMeasure()
    if 'BIoU' in metrics:
        BIoU = BIoUMeasure()

    if isinstance(gt_paths, list) and isinstance(pred_paths, list):
        # print(len(gt_paths), len(pred_paths))
        assert len(gt_paths) == len(pred_paths)

    for idx_sample in tqdm(range(len(gt_paths)), total=len(gt_paths)) if verbose else range(len(gt_paths)):
        gt = gt_paths[idx_sample]
        pred = pred_paths[idx_sample]

        pred = pred[:-4] + '.png'
        valid_extensions = ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']
        file_exists = False
        for ext in valid_extensions:
            if os.path.exists(pred[:-4] + ext):
                pred = pred[:-4] + ext
                file_exists = True
                break
        if file_exists:
            pred_ary = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)
        else:
            print('Not exists:', pred)

        gt_ary = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
        pred_ary = cv2.resize(pred_ary, (gt_ary.shape[1], gt_ary.shape[0]))

        if 'E' in metrics:
            EM.step(pred=pred_ary, gt=gt_ary)
        if 'S' in metrics:
            SM.step(pred=pred_ary, gt=gt_ary)
        if 'F' in metrics:
            FM.step(pred=pred_ary, gt=gt_ary)
        if 'MAE' in metrics:
            MAE.step(pred=pred_ary, gt=gt_ary)
        if 'WF' in metrics:
            WFM.step(pred=pred_ary, gt=gt_ary)
        if 'HCE' in metrics:
            ske_path = gt.replace('/gt/', '/ske/')
            if os.path.exists(ske_path):
                ske_ary = cv2.imread(ske_path, cv2.IMREAD_GRAYSCALE)
                ske_ary = ske_ary > 128
            else:
                ske_ary = skeletonize(gt_ary > 128)
                ske_save_dir = os.path.join(*ske_path.split(os.sep)[:-1])
                if ske_path[0] == os.sep:
                    ske_save_dir = os.sep + ske_save_dir
                os.makedirs(ske_save_dir, exist_ok=True)
                cv2.imwrite(ske_path, ske_ary.astype(np.uint8) * 255)
            HCE.step(pred=pred_ary, gt=gt_ary, gt_ske=ske_ary)
        if 'MBA' in metrics:
            MBA.step(pred=pred_ary, gt=gt_ary)
        if 'BIoU' in metrics:
            BIoU.step(pred=pred_ary, gt=gt_ary)

    if 'E' in metrics:
        em = EM.get_results()['em']
    else:
        em = {'curve': np.array([np.float64(-1)]), 'adp': np.float64(-1)}
    if 'S' in metrics:
        sm = SM.get_results()['sm']
    else:
        sm = np.float64(-1)
    if 'F' in metrics:
        fm = FM.get_results()['fm']
    else:
        fm = {'curve': np.array([np.float64(-1)]), 'adp': np.float64(-1)}
    if 'MAE' in metrics:
        mae = MAE.get_results()['mae']
    else:
        mae = np.float64(-1)
    if 'WF' in metrics:
        wfm = WFM.get_results()['wfm']
    else:
        wfm = np.float64(-1)
    if 'HCE' in metrics:
        hce = HCE.get_results()['hce']
    else:
        hce = np.float64(-1)
    if 'MBA' in metrics:
        mba = MBA.get_results()['mba']
    else:
        mba = np.float64(-1)
    if 'BIoU' in metrics:
        biou = BIoU.get_results()['biou']
    else:
        biou = {'curve': np.array([np.float64(-1)])}

    return em, sm, fm, mae, wfm, hce, mba, biou


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    gt = gt > 128
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    return min(2 * matrix.mean(), max_value)


class FMeasure(object):
    def __init__(self, beta: float = 0.3):
        self.beta = beta
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []
        self.changeable_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)

        adaptive_fm = self.cal_adaptive_fm(pred=pred, gt=gt)
        self.adaptive_fms.append(adaptive_fm)

        precisions, recalls, changeable_fms = self.cal_pr(pred=pred, gt=gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)
        self.changeable_fms.append(changeable_fms)

    def cal_adaptive_fm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        binary_predcition = pred >= adaptive_threshold
        area_intersection = binary_predcition[gt].sum()
        if area_intersection == 0:
            adaptive_fm = 0
        else:
            pre = area_intersection / np.count_nonzero(binary_predcition)
            rec = area_intersection / np.count_nonzero(gt)
            adaptive_fm = (1 + self.beta) * pre * rec / (self.beta * pre + rec)
        return adaptive_fm

    def cal_pr(self, pred: np.ndarray, gt: np.ndarray) -> tuple:
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred[gt], bins=bins)
        bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        TPs = fg_w_thrs
        Ps = fg_w_thrs + bg_w_thrs
        Ps[Ps == 0] = 1
        T = max(np.count_nonzero(gt), 1)
        precisions = TPs / Ps
        recalls = TPs / T
        numerator = (1 + self.beta) * precisions * recalls
        denominator = np.where(numerator == 0, 1, self.beta * precisions + recalls)
        changeable_fms = numerator / denominator
        return precisions, recalls, changeable_fms

    def get_results(self) -> dict:
        adaptive_fm = np.mean(np.array(self.adaptive_fms, _TYPE))
        changeable_fm = np.mean(np.array(self.changeable_fms, dtype=_TYPE), axis=0)
        precision = np.mean(np.array(self.precisions, dtype=_TYPE), axis=0)  # N, 256
        recall = np.mean(np.array(self.recalls, dtype=_TYPE), axis=0)  # N, 256
        return dict(fm=dict(adp=adaptive_fm, curve=changeable_fm),
                    pr=dict(p=precision, r=recall))


class MAEMeasure(object):
    def __init__(self):
        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)

        mae = self.cal_mae(pred, gt)
        self.maes.append(mae)

    def cal_mae(self, pred: np.ndarray, gt: np.ndarray) -> float:
        mae = np.mean(np.abs(pred - gt))
        return mae

    def get_results(self) -> dict:
        mae = np.mean(np.array(self.maes, _TYPE))
        return dict(mae=mae)


class SMeasure(object):
    def __init__(self, alpha: float = 0.5):
        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
            sm = max(0, sm)
        return sm

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1], ddof=1)
        score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
        return score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info['weight']
        pred1, pred2, pred3, pred4 = part_info['pred']
        gt1, gt2, gt3, gt4 = part_info['gt']
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def centroid(self, matrix: np.ndarray) -> tuple:
        h, w = matrix.shape
        area_object = np.count_nonzero(matrix)
        if area_object == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            # More details can be found at: https://www.yuque.com/lart/blog/gpbigm
            y, x = np.argwhere(matrix).mean(axis=0).round()
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x, y) -> dict:
        h, w = gt.shape
        area = h * w

        gt_LT = gt[0:y, 0:x]
        gt_RT = gt[0:y, x:w]
        gt_LB = gt[y:h, 0:x]
        gt_RB = gt[y:h, x:w]

        pred_LT = pred[0:y, 0:x]
        pred_RT = pred[0:y, x:w]
        pred_LB = pred[y:h, 0:x]
        pred_RB = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3

        return dict(gt=(gt_LT, gt_RT, gt_LB, gt_RB),
                    pred=(pred_LT, pred_RT, pred_LB, pred_RB),
                    weight=(w1, w2, w3, w4))

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)
        y = np.mean(gt)

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self) -> dict:
        sm = np.mean(np.array(self.sms, dtype=_TYPE))
        return dict(sm=sm)


class EMeasure(object):
    def __init__(self):
        self.adaptive_ems = []
        self.changeable_ems = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]

        changeable_ems = self.cal_changeable_em(pred, gt)
        self.changeable_ems.append(changeable_ems)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.adaptive_ems.append(adaptive_em)

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
        return adaptive_em

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        changeable_ems = self.cal_em_with_cumsumhistogram(pred, gt)
        return changeable_ems

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)

        fg___numel = fg_fg_numel + fg_bg_numel
        bg___numel = self.gt_size - fg___numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel, fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel, pred_bg_numel=bg___numel,
            )

            results_parts = []
            for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):
                align_matrix_value = 2 * (combination[0] * combination[1]) / \
                                     (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts.append(enhanced_matrix_value * part_numel)
            enhanced_matrix_sum = sum(results_parts)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg___numel_w_thrs = self.gt_size - fg___numel_w_thrs

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel_w_thrs
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs, fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs, pred_bg_numel=bg___numel_w_thrs,
            )

            results_parts = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
                align_matrix_value = 2 * (combination[0] * combination[1]) / \
                                     (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(axis=0)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def generate_parts_numel_combinations(self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel

        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size

        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value

        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value)
        ]
        return parts_numel, combinations

    def get_results(self) -> dict:
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=_TYPE))
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=_TYPE), axis=0)
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))


class WeightedFMeasure(object):
    def __init__(self, beta: float = 1):
        self.beta = beta
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.weighted_fms.append(wfm)

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        # MIN_E_EA = E;
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])


        R = 1 - np.mean(Ew[gt == 1])
        P = TPw / (TPw + FPw + _EPS)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (R + self.beta * P + _EPS)

        return Q

    def matlab_style_gauss2D(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m: m + 1, -n: n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self) -> dict:
        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=_TYPE))
        return dict(wfm=weighted_fm)


class HCEMeasure(object):
    def __init__(self):
        self.hces = []

    def step(self, pred: np.ndarray, gt: np.ndarray, gt_ske):
        # pred, gt = _prepare_data(pred, gt)

        hce = self.cal_hce(pred, gt, gt_ske)
        self.hces.append(hce)

    def get_results(self) -> dict:
        hce = np.mean(np.array(self.hces, _TYPE))
        return dict(hce=hce)


    def cal_hce(self, pred: np.ndarray, gt: np.ndarray, gt_ske: np.ndarray, relax=5, epsilon=2.0) -> float:
        # Binarize gt
        if(len(gt.shape)>2):
            gt = gt[:, :, 0]

        epsilon_gt = 128#(np.amin(gt)+np.amax(gt))/2.0
        gt = (gt>epsilon_gt).astype(np.uint8)

        # Binarize pred
        if(len(pred.shape)>2):
            pred = pred[:, :, 0]
        epsilon_pred = 128#(np.amin(pred)+np.amax(pred))/2.0
        pred = (pred>epsilon_pred).astype(np.uint8)

        Union = np.logical_or(gt, pred)
        TP = np.logical_and(gt, pred)
        FP = pred - TP
        FN = gt - TP

        # relax the Union of gt and pred
        Union_erode = Union.copy()
        Union_erode = cv2.erode(Union_erode.astype(np.uint8), disk(1), iterations=relax)

        # --- get the relaxed False Positive regions for computing the human efforts in correcting them ---
        FP_ = np.logical_and(FP, Union_erode) # get the relaxed FP
        for i in range(0, relax):
            FP_ = cv2.dilate(FP_.astype(np.uint8), disk(1))
            FP_ = np.logical_and(FP_, 1-np.logical_or(TP, FN))
        FP_ = np.logical_and(FP, FP_)

        # --- get the relaxed False Negative regions for computing the human efforts in correcting them ---
        FN_ = np.logical_and(FN, Union_erode) # preserve the structural components of FN
        ## recover the FN, where pixels are not close to the TP borders
        for i in range(0, relax):
            FN_ = cv2.dilate(FN_.astype(np.uint8), disk(1))
            FN_ = np.logical_and(FN_, 1-np.logical_or(TP, FP))
        FN_ = np.logical_and(FN, FN_)
        FN_ = np.logical_or(FN_, np.logical_xor(gt_ske, np.logical_and(TP, gt_ske))) # preserve the structural components of FN

        ## 2. =============Find exact polygon control points and independent regions==============
        ## find contours from FP_
        ctrs_FP, hier_FP = cv2.findContours(FP_.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        ## find control points and independent regions for human correction
        bdies_FP, indep_cnt_FP = self.filter_bdy_cond(ctrs_FP, FP_, np.logical_or(TP,FN_))
        ## find contours from FN_
        ctrs_FN, hier_FN = cv2.findContours(FN_.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        ## find control points and independent regions for human correction
        bdies_FN, indep_cnt_FN = self.filter_bdy_cond(ctrs_FN, FN_, 1-np.logical_or(np.logical_or(TP, FP_), FN_))

        poly_FP, poly_FP_len, poly_FP_point_cnt = self.approximate_RDP(bdies_FP, epsilon=epsilon)
        poly_FN, poly_FN_len, poly_FN_point_cnt = self.approximate_RDP(bdies_FN, epsilon=epsilon)

        # FP_points+FP_indep+FN_points+FN_indep
        return poly_FP_point_cnt+indep_cnt_FP+poly_FN_point_cnt+indep_cnt_FN

    def filter_bdy_cond(self, bdy_, mask, cond):

        cond = cv2.dilate(cond.astype(np.uint8), disk(1))
        labels = label(mask) # find the connected regions
        lbls = np.unique(labels) # the indices of the connected regions
        indep = np.ones(lbls.shape[0]) # the label of each connected regions
        indep[0] = 0 # 0 indicate the background region

        boundaries = []
        h,w = cond.shape[0:2]
        ind_map = np.zeros((h, w))
        indep_cnt = 0

        for i in range(0, len(bdy_)):
            tmp_bdies = []
            tmp_bdy = []
            for j in range(0, bdy_[i].shape[0]):
                r, c = bdy_[i][j,0,1],bdy_[i][j,0,0]

                if(np.sum(cond[r, c])==0 or ind_map[r, c]!=0):
                    if(len(tmp_bdy)>0):
                        tmp_bdies.append(tmp_bdy)
                        tmp_bdy = []
                    continue
                tmp_bdy.append([c, r])
                ind_map[r, c] =  ind_map[r, c] + 1
                indep[labels[r, c]] = 0 # indicates part of the boundary of this region needs human correction
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

            for k in range(0, len(tmp_bdies)):
                tmp_bdies[k] =  np.array(tmp_bdies[k])[:, np.newaxis, :]
            if(len(tmp_bdies)>0):
                boundaries.extend(tmp_bdies)

        return boundaries, np.sum(indep)

    # this function approximate each boundary by DP algorithm
    # https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
    def approximate_RDP(self, boundaries, epsilon=1.0):

        boundaries_ = []
        boundaries_len_ = []
        pixel_cnt_ = 0

        # polygon approximate of each boundary
        for i in range(0, len(boundaries)):
            boundaries_.append(cv2.approxPolyDP(boundaries[i], epsilon, False))

        # count the control points number of each boundary and the total control points number of all the boundaries
        for i in range(0, len(boundaries_)):
            boundaries_len_.append(len(boundaries_[i]))
            pixel_cnt_ = pixel_cnt_ + len(boundaries_[i])

        return boundaries_, boundaries_len_, pixel_cnt_


class MBAMeasure(object):
    def __init__(self):
        self.bas = []
        self.all_h = 0
        self.all_w = 0
        self.all_max = 0

    def step(self, pred: np.ndarray, gt: np.ndarray):
        # pred, gt = _prepare_data(pred, gt)
        
        refined = gt.copy()

        rmin = cmin = 0
        rmax, cmax = gt.shape

        self.all_h += rmax
        self.all_w += cmax
        self.all_max += max(rmax, cmax)

        refined_h, refined_w = refined.shape
        if refined_h != cmax:
            refined = np.array(Image.fromarray(pred).resize((cmax, rmax), Image.BILINEAR))

        if not(gt.sum() < 32*32):
            if not((cmax==cmin) or (rmax==rmin)):
                class_refined_prob = np.array(Image.fromarray(pred).resize((cmax-cmin, rmax-rmin), Image.BILINEAR))
                refined[rmin:rmax, cmin:cmax] = class_refined_prob
        
        pred = pred > 128
        gt = gt > 128

        ba = self.cal_ba(pred, gt)
        self.bas.append(ba)
        
    def get_disk_kernel(self, radius):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))

    def cal_ba(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Calculate the mean absolute error.

        :return: ba
        """
    
        gt = gt.astype(np.uint8)
        pred = pred.astype(np.uint8)

        h, w = gt.shape

        min_radius = 1
        max_radius = (w+h)/300
        num_steps = 5

        pred_acc = [None] * num_steps

        for i in range(num_steps):
            curr_radius = min_radius + int((max_radius-min_radius)/num_steps*i)

            kernel = self.get_disk_kernel(curr_radius)
            boundary_region = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0

            gt_in_bound = gt[boundary_region]
            pred_in_bound = pred[boundary_region]

            num_edge_pixels = (boundary_region).sum()
            num_pred_gd_pix = ((gt_in_bound) * (pred_in_bound) + (1-gt_in_bound) * (1-pred_in_bound)).sum()

            pred_acc[i] = num_pred_gd_pix / num_edge_pixels

        ba = sum(pred_acc)/num_steps
        return ba

    def get_results(self) -> dict:
        mba = np.mean(np.array(self.bas, _TYPE))
        return dict(mba=mba)


class BIoUMeasure(object):
    def __init__(self, dilation_ratio=0.02):
        self.bious = []
        self.dilation_ratio = dilation_ratio
            
    def mask_to_boundary(self, mask):
        h, w = mask.shape
        img_diag = np.sqrt(h ** 2 + w ** 2)
        dilation = int(round(self.dilation_ratio * img_diag))
        if dilation < 1:
            dilation = 1
        # Pad image so mask truncated by the image border is also considered as boundary.
        new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
        mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
        # G_d intersects G in the paper.
        return mask - mask_erode

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)

        bious = self.cal_biou(pred=pred, gt=gt)
        self.bious.append(bious)

    def cal_biou(self, pred, gt):
        pred = (pred * 255).astype(np.uint8)
        pred = self.mask_to_boundary(pred)
        gt = (gt * 255).astype(np.uint8)
        gt = self.mask_to_boundary(gt)
        gt = gt > 128
            
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred[gt], bins=bins) # ture positive
        bg_hist, _ = np.histogram(pred[~gt], bins=bins) # false positive
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0) 
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        TPs = fg_w_thrs
        Ps = fg_w_thrs + bg_w_thrs # positives
        Ps[Ps == 0] = 1 
        T = max(np.count_nonzero(gt), 1)
        
        ious = TPs / (T + bg_w_thrs)
        return ious

    def get_results(self) -> dict:
        biou = np.mean(np.array(self.bious, dtype=_TYPE), axis=0)
        return dict(biou=dict(curve=biou))
