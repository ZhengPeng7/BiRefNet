import os
import cv2
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torchvision import transforms

from preproc import preproc
from config import Config


Image.MAX_IMAGE_PIXELS = None       # remove DecompressionBombWarning
config = Config()
_class_labels_TR_sorted = 'Airplane, Ant, Antenna, Archery, Axe, BabyCarriage, Bag, BalanceBeam, Balcony, Balloon, Basket, BasketballHoop, Beatle, Bed, Bee, Bench, Bicycle, BicycleFrame, BicycleStand, Boat, Bonsai, BoomLift, Bridge, BunkBed, Butterfly, Button, Cable, CableLift, Cage, Camcorder, Cannon, Canoe, Car, CarParkDropArm, Carriage, Cart, Caterpillar, CeilingLamp, Centipede, Chair, Clip, Clock, Clothes, CoatHanger, Comb, ConcretePumpTruck, Crack, Crane, Cup, DentalChair, Desk, DeskChair, Diagram, DishRack, DoorHandle, Dragonfish, Dragonfly, Drum, Earphone, Easel, ElectricIron, Excavator, Eyeglasses, Fan, Fence, Fencing, FerrisWheel, FireExtinguisher, Fishing, Flag, FloorLamp, Forklift, GasStation, Gate, Gear, Goal, Golf, GymEquipment, Hammock, Handcart, Handcraft, Handrail, HangGlider, Harp, Harvester, Headset, Helicopter, Helmet, Hook, HorizontalBar, Hydrovalve, IroningTable, Jewelry, Key, KidsPlayground, Kitchenware, Kite, Knife, Ladder, LaundryRack, Lightning, Lobster, Locust, Machine, MachineGun, MagazineRack, Mantis, Medal, MemorialArchway, Microphone, Missile, MobileHolder, Monitor, Mosquito, Motorcycle, MovingTrolley, Mower, MusicPlayer, MusicStand, ObservationTower, Octopus, OilWell, OlympicLogo, OperatingTable, OutdoorFitnessEquipment, Parachute, Pavilion, Piano, Pipe, PlowHarrow, PoleVault, Punchbag, Rack, Racket, Rifle, Ring, Robot, RockClimbing, Rope, Sailboat, Satellite, Scaffold, Scale, Scissor, Scooter, Sculpture, Seadragon, Seahorse, Seal, SewingMachine, Ship, Shoe, ShoppingCart, ShoppingTrolley, Shower, Shrimp, Signboard, Skateboarding, Skeleton, Skiing, Spade, SpeedBoat, Spider, Spoon, Stair, Stand, Stationary, SteeringWheel, Stethoscope, Stool, Stove, StreetLamp, SweetStand, Swing, Sword, TV, Table, TableChair, TableLamp, TableTennis, Tank, Tapeline, Teapot, Telescope, Tent, TobaccoPipe, Toy, Tractor, TrafficLight, TrafficSign, Trampoline, TransmissionTower, Tree, Tricycle, TrimmerCover, Tripod, Trombone, Truck, Trumpet, Tuba, UAV, Umbrella, UnevenBars, UtilityPole, VacuumCleaner, Violin, Wakesurfing, Watch, WaterTower, WateringPot, Well, WellLid, Wheel, Wheelchair, WindTurbine, Windmill, WineGlass, WireWhisk, Yacht'
class_labels_TR_sorted = _class_labels_TR_sorted.split(', ')


class MyData(data.Dataset):
    def __init__(self, data_root, image_size, is_train=True):
        self.size_train = image_size
        self.size_test = image_size
        self.keep_size = not config.size
        self.data_size = (config.size, config.size)
        self.is_train = is_train
        self.load_all = config.load_all
        self.device = config.device
        self.dataset = data_root.replace('\\', '/').split('/')[-1]
        if self.is_train and config.auxiliary_classification:
            self.cls_name2id = {_name: _id for _id, _name in enumerate(class_labels_TR_sorted)}
        self.transform_image = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ][self.load_all or self.keep_size:])
        self.transform_label = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
        ][self.load_all or self.keep_size:])
        ## 'im' and 'gt' need modifying
        image_root = os.path.join(data_root, 'im')
        self.image_paths = [os.path.join(image_root, p) for p in os.listdir(image_root)]
        self.label_paths = [p.replace('/im/', '/gt/').replace('.jpg', '.png') for p in self.image_paths]
        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            self.class_labels_loaded = []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                _image = cv2.imread(image_path)
                _label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if not self.keep_size:
                    _image_rs = cv2.resize(_image, (config.size, config.size), interpolation=cv2.INTER_LINEAR)
                    _label_rs = cv2.resize(_label, (config.size, config.size), interpolation=cv2.INTER_LINEAR)
                self.images_loaded.append(
                    Image.fromarray(cv2.cvtColor(_image_rs, cv2.COLOR_BGR2RGB)).convert('RGB')
                )
                self.labels_loaded.append(
                    Image.fromarray(_label_rs).convert('L')
                )
                self.class_labels_loaded.append(
                    self.cls_name2id[label_path.split('/')[-1].split('#')[3]] if self.is_train and config.auxiliary_classification else -1
                )


    def __getitem__(self, index):

        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
            class_label = self.class_labels_loaded[index] if self.is_train and config.auxiliary_classification else -1
        else:
            image = Image.open(self.image_paths[index]).convert('RGB')
            label = Image.open(self.label_paths[index]).convert('L')
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
