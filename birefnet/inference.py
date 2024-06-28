import os
import argparse
from glob import glob
from tqdm import tqdm
import cv2
import torch

from dataset import MyData
from birefnet.models.birefnet import BiRefNet
from birefnet.utils import save_tensor_img, check_state_dict
from birefnet.config import Config


config = Config()


def inference(model, data_loader_test, pred_root, method, testset, device=0):
    model_training = model.training
    if model_training:
        model.eval()
    for batch in tqdm(data_loader_test, total=len(data_loader_test)) if 1 or config.verbose_eval else data_loader_test:
        inputs = batch[0].to(device)
        # gts = batch[1].to(device)
        label_paths = batch[-1]
        with torch.no_grad():
            scaled_preds = model(inputs)[-1].sigmoid()

        os.makedirs(os.path.join(pred_root, method, testset), exist_ok=True)

        for idx_sample in range(scaled_preds.shape[0]):
            res = torch.nn.functional.interpolate(
                scaled_preds[idx_sample].unsqueeze(0),
                size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2],
                mode='bilinear',
                align_corners=True
            )
            save_tensor_img(res, os.path.join(os.path.join(pred_root, method, testset), label_paths[idx_sample].replace('\\', '/').split('/')[-1]))   # test set dir + file name
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

    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=False)
    weights_lst = sorted(
        glob(os.path.join(args.ckpt_folder, '*.pth')) if args.ckpt_folder else [args.ckpt],
        key=lambda x: int(x.split('epoch_')[-1].split('.pth')[0]),
        reverse=True
    )
    for testset in args.testsets.split('+'):
        print('>>>> Testset: {}...'.format(testset))
        data_loader_test = torch.utils.data.DataLoader(
            dataset=MyData(testset, image_size=config.size, is_train=False),
            batch_size=config.batch_size_valid, shuffle=False, num_workers=config.num_workers, pin_memory=True
        )
        for weights in weights_lst:
            if int(weights.strip('.pth').split('epoch_')[-1]) % 1 != 0:
                continue
            print('\tInferencing {}...'.format(weights))
            # model.load_state_dict(torch.load(weights, map_location='cpu'))
            state_dict = torch.load(weights, map_location='cpu')
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            model = model.to(device)
            inference(
                model, data_loader_test=data_loader_test, pred_root=args.pred_root,
                method='--'.join([w.rstrip('.pth') for w in weights.split(os.sep)[-2:]]),
                testset=testset, device=config.device
            )


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt', type=str, help='model folder')
    parser.add_argument('--ckpt_folder', default=sorted(glob(os.path.join('ckpt', '*')))[-1], type=str, help='model folder')
    parser.add_argument('--pred_root', default='e_preds', type=str, help='Output folder')
    parser.add_argument('--testsets',
                        default={
                            'DIS5K': 'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4',
                            'COD': 'TE-COD10K+NC4K+TE-CAMO+CHAMELEON',
                            'HRSOD': 'DAVIS-S+TE-HRSOD+TE-UHRSD+TE-DUTS+DUT-OMRON',
                            'DIS5K+HRSOD+HRS10K': 'DIS-VD',
                            'P3M-10k': 'TE-P3M-500-P+TE-P3M-500-NP',
                            'DIS5K-': 'DIS-VD',
                            'COD-': 'TE-COD10K',
                            'SOD-': 'DAVIS-S+TE-HRSOD+TE-UHRSD',
                        }[config.task + ''],
                        type=str,
                        help="Test all sets: , 'DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4'")

    args = parser.parse_args()

    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')
    main(args)
