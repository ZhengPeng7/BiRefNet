import os
import cv2
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

from preproc import preproc
from config import Config


Image.MAX_IMAGE_PIXELS = None       # remove DecompressionBombWarning
config = Config()


class MyData(data.Dataset):
    def __init__(self, data_root, image_size, is_train=True):

        self.size_train = image_size
        self.size_test = image_size
        self.data_size = (config.size, config.size)
        self.is_train = is_train
        self.load_all = config.load_all
        self.device = torch.device(config.device)
        self.dataset = data_root.replace('\\', '/').split('/')[-1]
        if self.load_all:
            self.transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.transform_label = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform_image = transforms.Compose([
                transforms.Resize(self.data_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.transform_label = transforms.Compose([
                transforms.Resize(self.data_size),
                transforms.ToTensor(),
            ])
        ## 'im' and 'gt' need modifying
        image_root = os.path.join(data_root, 'im')
        self.image_paths = [os.path.join(image_root, p) for p in os.listdir(image_root)]
        self.label_paths = [p.replace('/im/', '/gt/').replace('.jpg', '.png') for p in self.image_paths]
        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                self.images_loaded.append(
                    Image.fromarray(
                        cv2.cvtColor(cv2.resize(cv2.imread(image_path), (config.size, config.size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                    ).convert('RGB')
                )
                self.labels_loaded.append(
                    Image.fromarray(
                        cv2.resize(cv2.imread(label_path, cv2.IMREAD_GRAYSCALE), (config.size, config.size), interpolation=cv2.INTER_LINEAR)
                    ).convert('L')
                )


    def __getitem__(self, index):

        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
        else:
            image = Image.open(self.image_paths[index]).convert('RGB')
            label = Image.open(self.label_paths[index]).convert('L')

        # loading image and label
        if self.is_train:
            image, label = preproc(image, label, preproc_methods=config.preproc_methods)

        image, label = self.transform_image(image), self.transform_label(label)

        if self.is_train:
            return image, label
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)
