import os
import torch
from data_loader_cache import get_im_gt_name_dict, create_dataloaders, GOSNormalize
from models.isnet import ISNetDIS
from config import Config
from train_valid_inference_main import valid


config = Config()
data_root_dir = '/root/autodl-tmp/datasets/dis'


dataset_vd = {"name": "DIS5K-VD",
                "im_dir": os.path.join(data_root_dir, "DIS5K/DIS-VD/im"),
                "gt_dir": os.path.join(data_root_dir, "DIS5K/DIS-VD/gt"),
                "im_ext": ".jpg",
                "gt_ext": ".png",
                "cache_dir": os.path.join(data_root_dir, "DIS5K-Cache/DIS-VD")}
valid_datasets = [dataset_vd]

### --------------- STEP 2: Configuring the hyperparamters for Training, validation and inferencing ---------------
hypar = {}

## -- 2.1. configure the model saving or restoring path --
hypar["mode"] = "train"
## "train": for training,
## "valid": for validation and inferening,
## in "valid" mode, it will calculate the accuracy as well as save the prediciton results into the "hypar["valid_out_dir"]", which shouldn't be ""
## otherwise only accuracy will be calculated and no predictions will be saved
hypar["interm_sup"] = False ## in-dicate if activate intermediate feature supervision

if hypar["mode"] == "train":
    hypar["valid_out_dir"] = "" ## for "train" model leave it as "", for "valid"("inference") mode: set it according to your local directory
    hypar["model_path"] = os.path.join(data_root_dir, "saved_models/IS-Net-test") ## model weights saving (or restoring) path
    hypar["restore_model"] = "" ## name of the segmentation model weights .pth for resume training process from last stop or for the inferencing
    hypar["start_ite"] = 0 ## start iteration for the training, can be changed to match the restored training process
    hypar["gt_encoder_model"] = ""
else: ## configure the segmentation output path and the to-be-used model weights path
    hypar["valid_out_dir"] =  os.path.join(data_root_dir, "your-results/")## os.path.join(data_root_dir, "DIS5K-Results-test") ## output inferenced segmentation maps into this fold
    hypar["model_path"] =  os.path.join(data_root_dir, "saved_models/IS-Net") ## load trained weights from this path
    hypar["restore_model"] = "isnet.pth"##"isnet.pth" ## name of the to-be-loaded weights

# if hypar["restore_model"]!="":
#     hypar["start_ite"] = int(hypar["restore_model"].split("_")[2])

## -- 2.2. choose floating point accuracy --
hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
hypar["seed"] = 0

## -- 2.3. cache data spatial size --
## To handle large size input images, which take a lot of time for loading in training,
#  we introduce the cache mechanism for pre-convering and resizing the jpg and png images into .pt file
hypar["cache_size"] = [config.size, config.size] ## cached input spatial resolution, can be configured into different size
hypar["cache_boost_train"] = False ## "True" or "False", indicates wheather to load all the training datasets into RAM, True will greatly speed the training process while requires more RAM
hypar["cache_boost_valid"] = False ## "True" or "False", indicates wheather to load all the validation datasets into RAM, True will greatly speed the training process while requires more RAM

## --- 2.4. data augmentation parameters ---
hypar["input_size"] = [config.size, config.size] ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
hypar["crop_size"] = [config.size, config.size] ## random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation
hypar["random_flip_h"] = 1 ## horizontal flip, currently hard coded in the dataloader and it is not in use
hypar["random_flip_v"] = 0 ## vertical flip , currently not in use

## --- 2.5. define model  ---
print("building model...")
hypar["model"] = ISNetDIS() #U2NETFASTFEATURESUP()
hypar["early_stop"] = 20 ## stop the training when no improvement in the past 20 validation periods, smaller numbers can be used here e.g., 5 or 10.
hypar["model_save_fre"] = 2000 ## valid and save model weights every 2000 iterations

hypar["batch_size_train"] = config.batch_size ## batch size for training
hypar["batch_size_valid"] = 1 ## batch size for validation and inferencing
print("batch size: ", hypar["batch_size_train"])

hypar["max_ite"] = 10000000 ## if early stop couldn't stop the training process, stop it by the max_ite_num
hypar["max_epoch_num"] = 1000000 ## if early stop and max_ite couldn't stop the training process, stop it by the max_epoch_num




print("--- create valid dataloader ---")
## build dataloader for validation or testing
valid_nm_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
## build dataloader for training datasets
valid_dataloaders, valid_datasets = create_dataloaders(valid_nm_im_gt_list,
                                                        cache_size = hypar["cache_size"],
                                                        cache_boost = hypar["cache_boost_valid"],
                                                        my_transforms = [
                                                                        GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0]),
                                                                        # GOSResize(hypar["input_size"])
                                                                        ],
                                                        batch_size=hypar["batch_size_valid"],
                                                        shuffle=False)
print(len(valid_dataloaders), " valid dataloaders created")
print("--- build model ---")
net = hypar["model"]
if 'cuda' in config.device:
    net.load_state_dict(torch.load(
        '../../../datasets/dis/saved_models/IS-Net-test/gpu_itr_16000_traLoss_0.0555_traTarLoss_0.0555_valLoss_0.2227_valTarLoss_0.2227_maxF1_0.8119_mae_0.077_time_0.034913.pth'
    ))
    net.cuda()
else:
    net.load_state_dict(torch.load(
        '../../../datasets/dis/saved_models/IS-Net-test/gpu_itr_16000_traLoss_0.0555_traTarLoss_0.0555_valLoss_0.2227_valTarLoss_0.2227_maxF1_0.8119_mae_0.077_time_0.034913.pth',
        map_location="cpu"
    ))

res = valid(net, valid_dataloaders, valid_datasets, hypar, eval_all_metrics=config.eval_all_metrics)
if config.eval_all_metrics:
    keys = ['Emax_all_ds', 'Sm_all_ds', 'Fmax_all_ds', 'MAE_all_ds', 'wFm_all_ds', 'val_loss', 'tar_loss', 'i_val']
else:
    keys = ['tmp_sm', 'tmp_mae', 'val_loss', 'tar_loss', 'i_val']
for k, r in zip(keys, res):
    print(k, r)
