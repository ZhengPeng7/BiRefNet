import os
import time
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from models.isnet import ISNetDIS


if __name__ == "__main__":
    dataset_path=""  #Your dataset path
    model_path=""  #Your model path
    result_path=""  #The folder path that you want to save the results
    input_size=[1024,1024]
    net=ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net=net.cuda()
    else:
        net.load_state_dict(torch.load(model_path,map_location="cpu"))

    im_list = glob(dataset_path+"/*.jpg")
    for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
        print("im_path: ", im_path)
        im = io.imread(im_path)
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_shp=im.shape[0:2]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
        im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear")
        image = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0]).type(torch.uint8)
        if torch.cuda.is_available():
            image=image.cuda()
        image = torch.divide(image,255.0)
        result=net(image)
        result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result-mi)/(ma-mi)
        im_name=im_path.split('/')[-1].split('.')[0]
        io.imsave(os.path.join(result_path,im_name+".png"),(result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8))
