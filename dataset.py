import os
import cv2
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torchvision import transforms

from image_proc import preproc
from config import Config
from utils import path_to_image


Image.MAX_IMAGE_PIXELS = None       # remove DecompressionBombWarning
config = Config()
_class_labels_TR_sorted = (
    'Airplane, Ant, Antenna, Archery, Axe, BabyCarriage, Bag, BalanceBeam, Balcony, Balloon, Basket, BasketballHoop, Beatle, Bed, Bee, Bench, Bicycle, '
    'BicycleFrame, BicycleStand, Boat, Bonsai, BoomLift, Bridge, BunkBed, Butterfly, Button, Cable, CableLift, Cage, Camcorder, Cannon, Canoe, Car, '
    'CarParkDropArm, Carriage, Cart, Caterpillar, CeilingLamp, Centipede, Chair, Clip, Clock, Clothes, CoatHanger, Comb, ConcretePumpTruck, Crack, Crane, '
    'Cup, DentalChair, Desk, DeskChair, Diagram, DishRack, DoorHandle, Dragonfish, Dragonfly, Drum, Earphone, Easel, ElectricIron, Excavator, Eyeglasses, '
    'Fan, Fence, Fencing, FerrisWheel, FireExtinguisher, Fishing, Flag, FloorLamp, Forklift, GasStation, Gate, Gear, Goal, Golf, GymEquipment, Hammock, '
    'Handcart, Handcraft, Handrail, HangGlider, Harp, Harvester, Headset, Helicopter, Helmet, Hook, HorizontalBar, Hydrovalve, IroningTable, Jewelry, Key, '
    'KidsPlayground, Kitchenware, Kite, Knife, Ladder, LaundryRack, Lightning, Lobster, Locust, Machine, MachineGun, MagazineRack, Mantis, Medal, MemorialArchway, '
    'Microphone, Missile, MobileHolder, Monitor, Mosquito, Motorcycle, MovingTrolley, Mower, MusicPlayer, MusicStand, ObservationTower, Octopus, OilWell, '
    'OlympicLogo, OperatingTable, OutdoorFitnessEquipment, Parachute, Pavilion, Piano, Pipe, PlowHarrow, PoleVault, Punchbag, Rack, Racket, Rifle, Ring, Robot, '
    'RockClimbing, Rope, Sailboat, Satellite, Scaffold, Scale, Scissor, Scooter, Sculpture, Seadragon, Seahorse, Seal, SewingMachine, Ship, Shoe, ShoppingCart, '
    'ShoppingTrolley, Shower, Shrimp, Signboard, Skateboarding, Skeleton, Skiing, Spade, SpeedBoat, Spider, Spoon, Stair, Stand, Stationary, SteeringWheel, '
    'Stethoscope, Stool, Stove, StreetLamp, SweetStand, Swing, Sword, TV, Table, TableChair, TableLamp, TableTennis, Tank, Tapeline, Teapot, Telescope, Tent, '
    'TobaccoPipe, Toy, Tractor, TrafficLight, TrafficSign, Trampoline, TransmissionTower, Tree, Tricycle, TrimmerCover, Tripod, Trombone, Truck, Trumpet, Tuba, '
    'UAV, Umbrella, UnevenBars, UtilityPole, VacuumCleaner, Violin, Wakesurfing, Watch, WaterTower, WateringPot, Well, WellLid, Wheel, Wheelchair, WindTurbine, Windmill, WineGlass, WireWhisk, Yacht'
)
class_labels_TR_sorted = _class_labels_TR_sorted.split(', ')


class MyData(data.Dataset):
    def __init__(self, datasets, image_size, is_train=True):
        self.size_train = image_size
        self.size_test = image_size
        self.keep_size = not config.size
        self.data_size = config.size
        self.is_train = is_train
        self.load_all = config.load_all
        self.device = config.device
        valid_extensions = ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']

        if self.is_train and config.auxiliary_classification:
            self.cls_name2id = {_name: _id for _id, _name in enumerate(class_labels_TR_sorted)}
        self.transform_image = transforms.Compose([
            transforms.Resize(self.data_size[::-1]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ][self.load_all or self.keep_size:])
        self.transform_label = transforms.Compose([
            transforms.Resize(self.data_size[::-1]),
            transforms.ToTensor(),
        ][self.load_all or self.keep_size:])
        dataset_root = os.path.join(config.data_root_dir, config.task)
        # datasets can be a list of different datasets for training on combined sets.
        self.image_paths = []
        for dataset in datasets.split('+'):
            image_root = os.path.join(dataset_root, dataset, 'im')
            self.image_paths += [os.path.join(image_root, p) for p in os.listdir(image_root) if any(p.endswith(ext) for ext in valid_extensions)]
        self.label_paths = []
        for p in self.image_paths:
            for ext in valid_extensions:
                ## 'im' and 'gt' may need modifying
                p_gt = p.replace('/im/', '/gt/')[:-(len(p.split('.')[-1])+1)] + ext
                file_exists = False
                if os.path.exists(p_gt):
                    self.label_paths.append(p_gt)
                    file_exists = True
                    break
            if not file_exists:
                print('Not exists:', p_gt)

        if len(self.label_paths) != len(self.image_paths):
            set_image_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.image_paths])
            set_label_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.label_paths])
            print('Path diff:', set_image_paths - set_label_paths)
            raise ValueError(f"There are different numbers of images ({len(self.label_paths)}) and labels ({len(self.image_paths)})")

        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            self.class_labels_loaded = []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                _image = path_to_image(image_path, size=config.size, color_type='rgb')
                _label = path_to_image(label_path, size=config.size, color_type='gray')
                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)
                self.class_labels_loaded.append(
                    self.cls_name2id[label_path.split('/')[-1].split('#')[3]] if self.is_train and config.auxiliary_classification else -1
                )

    def __getitem__(self, index):

        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
            class_label = self.class_labels_loaded[index] if self.is_train and config.auxiliary_classification else -1
        else:
            image = path_to_image(self.image_paths[index], size=config.size, color_type='rgb')
            label = path_to_image(self.label_paths[index], size=config.size, color_type='gray')
            class_label = self.cls_name2id[self.label_paths[index].split('/')[-1].split('#')[3]] if self.is_train and config.auxiliary_classification else -1

        # loading image and label
        if self.is_train:
            image, label = preproc(image, label, preproc_methods=config.preproc_methods)
        # else:
        #     if _label.shape[0] > 2048 or _label.shape[1] > 2048:
        #         _image = cv2.resize(_image, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        #         _label = cv2.resize(_label, (2048, 2048), interpolation=cv2.INTER_LINEAR)

        image, label = self.transform_image(image), self.transform_label(label)

        if self.is_train:
            return image, label, class_label
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)
