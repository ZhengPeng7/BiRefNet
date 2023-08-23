import os
import argparse
from glob import glob
from tqdm import tqdm
import cv2
import torch
from torch import nn

from dataset import MyData
from models.baseline import BSL
from utils import save_tensor_img
from config import Config


config = Config()


def inference(model, data_loader_test, pred_dir, method, testset, device=0):
    model_training = model.training
    if model_training:
        model.eval()
    for batch in tqdm(data_loader_test, total=len(data_loader_test)//config.batch_size_valid) if config.verbose_eval else data_loader_test:
        inputs = batch[0].to(device)
        # gts = batch[1].to(device)
        label_paths = batch[-1]
        with torch.no_grad():
            scaled_preds = model(inputs)[-1].sigmoid()

        os.makedirs(os.path.join(pred_dir, method, testset), exist_ok=True)

        for idx_sample in range(scaled_preds.shape[0]):
            res = nn.functional.interpolate(
                scaled_preds[idx_sample].unsqueeze(0),
                size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2],
                mode='bilinear',
                align_corners=True
            )
            save_tensor_img(res, os.path.join(os.path.join(pred_dir, method, testset), label_paths[idx_sample].replace('\\', '/').split('/')[-1]))   # test set dir + file name
    if model_training:
        model.train()
    return None


def main(args):
    # Init model

    device = config.device
    if args.ckpt_folder:
        print('Testing with models in {}'.format(args.ckpt_folder))
    else:
        print('Testing with model {}'.format(args.ckpt))

    if config.model == 'BSL':
        model = BSL()
    weights_lst = sorted(
        glob(os.path.join(args.ckpt_folder, '*.pth')) if args.ckpt_folder else [args.ckpt],
        key=lambda x: int(x.split('ep')[-1].split('.pth')[0]),
        reverse=True
    )
    for weights in weights_lst:
        print('Inferencing {}...'.format(weights))
        model.load_state_dict(torch.load(weights, map_location='cpu'))
        model = model.to(device)
        for testset in args.testsets.split('+'):
            data_loader_test = torch.utils.data.DataLoader(
                dataset=MyData(data_root=os.path.join(config.data_root_dir, config.dataset, testset), image_size=config.size, is_train=False),
                batch_size=config.batch_size_valid, shuffle=False, num_workers=config.num_workers, pin_memory=True
            )
            inference(model, data_loader_test=data_loader_test, pred_dir=args.pred_dir, method='--'.join([_.rstrip('.pth') for _ in weights.split(os.sep)[-2:]]), testset=testset)


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt', type=str, help='model folder')
    parser.add_argument('--ckpt_folder', default='', type=str, help='model folder')
    parser.add_argument('--pred_dir', default='e_preds', type=str, help='Output folder')
    parser.add_argument('--testsets',
                        default=[
                            'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4',
                            'COD10K+NC4K+CAMO+CHAMELEON',
                            'DIS-VD',
                            'COD10K',
                            'CHAMELEON',
                        ][0],
                        type=str,
                        help="Test all sets: , 'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4'")

    args = parser.parse_args()

    main(args)