import os
import time
import numpy as np
from skimage import io
import time

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from data_loader_cache import get_im_gt_name_dict, create_dataloaders, GOSRandomHFlip, GOSResize, GOSRandomCrop, GOSNormalize #GOSDatasetCache,
from basics import  f1_mae_torch #normPRED, GOSPRF1ScoresCache,f1score_torch,
from models.isnet import ISNetDIS
from config import Config
from evaluation.evaluator import evaluator_online_S, evaluator_online


config = Config()

def train(net, optimizer, train_dataloaders, train_datasets, valid_dataloaders, valid_datasets, hypar):

    model_path = hypar["model_path"]
    model_save_fre = hypar["model_save_fre"]
    max_ite = hypar["max_ite"]
    batch_size_train = hypar["batch_size_train"]
    batch_size_valid = hypar["batch_size_valid"]

    if(not os.path.exists(model_path)):
        os.makedirs(model_path)

    ite_num = hypar["start_ite"] # count the toal iteration number
    ite_num4val = 0 #
    running_loss = 0.0 # count the toal loss
    running_tar_loss = 0.0 # count the target output loss
    last_sm = [0 for x in range(len(valid_dataloaders))]
    last_mae = [0 for x in range(len(valid_dataloaders))]

    train_num = train_datasets[0].__len__()

    net.train()

    start_last = time.time()
    gos_dataloader = train_dataloaders[0]
    epoch_num = hypar["max_epoch_num"]
    notgood_cnt = 0
    best_sm_all_ds, best_mae_all_ds = [], []
    if config.eval_all_metrics:
        best_emax_all_ds, best_fmax_all_ds, best_wfm_all_ds, best_emean_all_ds = [], [], [], []
    for epoch in range(epoch_num): ## set the epoch num as 100000

        for i, data in enumerate(gos_dataloader):

            if(ite_num >= max_ite):
                print("Training Reached the Maximal Iteration Number ", max_ite)
                exit()

            # start_read = time.time()
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            # get the inputs
            inputs, labels = data['image'], data['label']

            if(hypar["model_digit"]=="full"):
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
            else:
                inputs = inputs.type(torch.HalfTensor)
                labels = labels.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # print("time lapse for data preparation: ", time.time()-start_read, ' s')

            # y zero the parameter gradients
            start_inf_loss_back = time.time()
            optimizer.zero_grad()

            # forward + backward + optimize
            ds,_ = net(inputs_v)
            loss2, loss = net.compute_loss(ds, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del outputs, loss
            del ds, loss2, loss
            end_inf_loss_back = time.time()-start_inf_loss_back

            if ite_num % (model_save_fre // 10) == 0:
                print(">>>"+model_path.split('/')[-1]+" - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, time-per-iter: %3f s, time_read: %3f" % (
                    epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val,
                    time.time()-start_last, time.time()-start_last-end_inf_loss_back
                ))
            start_last = time.time()

            if ite_num % model_save_fre == 0:  # validate every 2000 iterations
                notgood_cnt += 1
                net.eval()
                if config.eval_all_metrics:
                    Emax_all_ds, Sm_all_ds, Fmax_all_ds, MAE_all_ds, wFm_all_ds, Emean_all_ds, val_loss, tar_loss, i_val = valid(net, valid_dataloaders, valid_datasets, hypar, epoch, eval_all_metrics=config.eval_all_metrics)
                else:
                    Sm_all_ds, MAE_all_ds, val_loss, tar_loss, i_val = valid(net, valid_dataloaders, valid_datasets, hypar, epoch, eval_all_metrics=config.eval_all_metrics)
                net.train()  # resume train

                print("last_sm, last_mae:", last_sm, last_mae)
                update = 0
                improved_count = 0
                if not best_sm_all_ds:
                    best_sm_all_ds, best_mae_all_ds = Sm_all_ds, MAE_all_ds
                    if config.eval_all_metrics:
                        best_emax_all_ds = Emax_all_ds
                        best_fmax_all_ds = Fmax_all_ds
                        best_wfm_all_ds = wFm_all_ds
                        best_emean_all_ds = Emean_all_ds
                for idx_tmp in range(len(Sm_all_ds)):
                    # Dependin on the Sm of all test sets to decide whether to update the best.
                    if Sm_all_ds[idx_tmp] > best_sm_all_ds[idx_tmp]:
                        improved_count += 1
                if improved_count > len(Sm_all_ds) / 2:
                    for idx_imp in range(len(Sm_all_ds)):
                        best_sm_all_ds[idx_imp] = Sm_all_ds[idx_imp]
                        best_mae_all_ds[idx_imp] = MAE_all_ds[idx_imp]
                        if config.eval_all_metrics:
                            best_emax_all_ds[idx_tmp] = Emax_all_ds[idx_tmp]
                            best_fmax_all_ds[idx_tmp] = Fmax_all_ds[idx_tmp]
                            best_wfm_all_ds[idx_tmp] = wFm_all_ds[idx_tmp]
                            best_emean_all_ds[idx_tmp] = Emean_all_ds[idx_tmp]
                    update = 1
                print("tmp_sm, tmp_mae:", Sm_all_ds, MAE_all_ds)
                print("Best of sm, mae:\t", 
                    list(np.round(best_sm_all_ds, 4)),
                    list(np.round(best_mae_all_ds, 4))
                )
                if config.eval_all_metrics:
                    print("Emax_all_ds, Fmax_all_ds, wFm_all_ds, Emean_all_ds:", Emax_all_ds, Fmax_all_ds, wFm_all_ds, Emean_all_ds)
                    print("Best of emax, fmax, wfm, emean:\t",
                        list(np.round(best_emax_all_ds, 4)),
                        list(np.round(best_fmax_all_ds, 4)),
                        list(np.round(best_wfm_all_ds, 4)),
                        list(np.round(best_emean_all_ds, 4))
                    )
                if update:
                    notgood_cnt = 0
                    last_sm = Sm_all_ds
                    tmp_sm_str = [str(round(smx,4)) for smx in Sm_all_ds]
                    tmp_mae_str = [str(round(mx,4)) for mx in MAE_all_ds]
                    maxsm = '_'.join(tmp_sm_str)
                    meanM = '_'.join(tmp_mae_str)
                    # .cpu().detach().numpy()
                    model_name = "/gpu_itr_"+str(ite_num)+\
                                "_traLoss_"+str(np.round(running_loss / ite_num4val,4))+\
                                "_traTarLoss_"+str(np.round(running_tar_loss / ite_num4val,4))+\
                                "_valLoss_"+str(np.round(val_loss /(i_val+1),4))+\
                                "_valTarLoss_"+str(np.round(tar_loss /(i_val+1),4)) + \
                                "_maxsm_" + maxsm + \
                                "_mae_" + meanM + \
                                ".pth"
                    torch.save(net.state_dict(), model_path + model_name)

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

                if(notgood_cnt >= hypar["early_stop"]):
                    print("No improvements in the last "+str(notgood_cnt)+" validation periods, so training stopped !")
                    exit()

    print("Training Reaches The Maximum Epoch Number")

def valid(net, valid_dataloaders, valid_datasets, hypar, epoch=0, eval_all_metrics=False):
    net.eval()
    print("Validating...")
    epoch_num = hypar["max_epoch_num"]

    val_loss = 0.0
    tar_loss = 0.0
    val_cnt = 0.0

    Emax_all_ds, Sm_all_ds, Fmax_all_ds, MAE_all_ds, wFm_all_ds, Emean_all_ds = [], [], [], [], [], []
    for k in range(len(valid_dataloaders)):

        valid_dataloader = valid_dataloaders[k]
        valid_dataset = valid_datasets[k]

        val_num = valid_dataset.__len__()
        mybins = np.arange(0,256)
        Emax, Sm, Fmax, MAE, wFm, Emean = [], [], [], [], [], []

        for i_val, data_val in enumerate(valid_dataloader):
            val_cnt = val_cnt + 1.0
            imidx_val, inputs_val, labels_val, shapes_val = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape']

            if(hypar["model_digit"]=="full"):
                inputs_val = inputs_val.type(torch.FloatTensor)
                labels_val = labels_val.type(torch.FloatTensor)
            else:
                inputs_val = inputs_val.type(torch.HalfTensor)
                labels_val = labels_val.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_val_v, labels_val_v = Variable(inputs_val.cuda(), requires_grad=False), Variable(labels_val.cuda(), requires_grad=False)
            else:
                inputs_val_v, labels_val_v = Variable(inputs_val, requires_grad=False), Variable(labels_val,requires_grad=False)

            ds_val = net(inputs_val_v)[0]

            # loss2_val, loss_val = muti_loss_fusion(ds_val, labels_val_v)
            loss2_val, loss_val = net.compute_loss(ds_val, labels_val_v)

            # compute F measure
            for t in range(hypar["batch_size_valid"]):
                i_test = imidx_val[t].data.numpy()

                pred_val = ds_val[0][t,:,:,:] # B x 1 x H x W

                ## recover the prediction spatial size to the orignal image size
                pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(shapes_val[t][0],shapes_val[t][1]),mode='bilinear'))

                # pred_val = normPRED(pred_val)
                ma = torch.max(pred_val)
                mi = torch.min(pred_val)
                pred_val = (pred_val-mi)/(ma-mi) # max = 1

                if len(valid_dataset.dataset["ori_gt_path"]) != 0:
                    gt = np.squeeze(io.imread(valid_dataset.dataset["ori_gt_path"][i_test])) # max = 255
                else:
                    gt = np.zeros((shapes_val[t][0],shapes_val[t][1]))
                with torch.no_grad():
                    gt = torch.tensor(gt).to(config.device)

                if eval_all_metrics:
                    em, sm, fm, mae, wfm = evaluator_online(gt, pred_val*255)
                else:
                    sm, mae = evaluator_online_S(gt, pred_val*255)

                Sm.append(sm)
                MAE.append(mae)
                if eval_all_metrics:
                    Emax.append(em['curve'].max())
                    Fmax.append(fm['curve'].max())
                    wFm.append(wfm)
                    Emean.append(em['curve'].mean())

                del ds_val, gt
                gc.collect()
                torch.cuda.empty_cache()

            # if(loss_val.data[0]>1):
            val_loss += loss_val.item()#data[0]
            tar_loss += loss2_val.item()#data[0]


            if (i_val + 1) % 50 == 0:
                print("[validating: %5d/%5d] val_ls:%f, tar_ls: %f, Sm: %f, mae: %f"% (i_val, val_num, val_loss / (i_val + 1), tar_loss / (i_val + 1), sm, mae))

            del loss2_val, loss_val

        mean_sm = np.mean(Sm)
        mean_mae = np.mean(MAE)
        Sm_all_ds.append(mean_sm)
        MAE_all_ds.append(mean_mae)

        if eval_all_metrics:
            mean_emax = np.mean(Emax)
            mean_fmax = np.mean(Fmax)
            mean_wfm = np.mean(wFm)
            mean_emean = np.mean(Emean)
            Emax_all_ds.append(mean_emax)
            Fmax_all_ds.append(mean_fmax)
            wFm_all_ds.append(mean_wfm)
            Emean_all_ds.append(mean_emean)


    if eval_all_metrics:
        return Emax_all_ds, Sm_all_ds, Fmax_all_ds, MAE_all_ds, wFm_all_ds, Emean_all_ds, val_loss, tar_loss, i_val
    else:
        return Sm_all_ds, MAE_all_ds, val_loss, tar_loss, i_val

def main(train_datasets,
         valid_datasets,
         hypar): # model: "train", "test"

    ### --- Step 1: Build datasets and dataloaders ---
    dataloaders_train = []
    dataloaders_valid = []

    if(hypar["mode"]=="train"):
        print("--- create training dataloader ---")
        ## collect training dataset
        train_nm_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        ## build dataloader for training datasets
        train_dataloaders, train_datasets = create_dataloaders(train_nm_im_gt_list,
                                                             cache_size = hypar["cache_size"],
                                                             cache_boost = hypar["cache_boost_train"],
                                                             my_transforms = [
                                                                             GOSRandomHFlip(), ## this line can be uncommented for horizontal flip augmetation
                                                                             # GOSResize(hypar["input_size"]),
                                                                             # GOSRandomCrop(hypar["crop_size"]), ## this line can be uncommented for randomcrop augmentation
                                                                              GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0]),
                                                                              ],
                                                             batch_size = hypar["batch_size_train"],
                                                             shuffle = True)
        # train_dataloaders_val, train_datasets_val = create_dataloaders(train_nm_im_gt_list,
        #                                                      cache_size = hypar["cache_size"],
        #                                                      cache_boost = hypar["cache_boost_train"],
        #                                                      my_transforms = [
        #                                                                       GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0]),
        #                                                                       ],
        #                                                      batch_size = hypar["batch_size_valid"],
        #                                                      shuffle = False)
        print(len(train_dataloaders), " train dataloaders created")

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
    # print(valid_datasets[0]["data_name"])

    ### --- Step 2: Build Model and Optimizer ---
    print("--- build model ---")
    net = hypar["model"]#GOSNETINC(3,1)

    # convert to half precision
    if(hypar["model_digit"]=="half"):
        net.half()
        for layer in net.modules():
          if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    if torch.cuda.is_available():
        net.cuda()

    if(hypar["restore_model"]!=""):
        print("restore model from:")
        print(hypar["model_path"]+"/"+hypar["restore_model"])
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(hypar["model_path"]+"/"+hypar["restore_model"]))
        else:
            net.load_state_dict(torch.load(hypar["model_path"]+"/"+hypar["restore_model"],map_location="cpu"))

    print("--- define optimizer ---")
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    ### --- Step 3: Train or Valid Model ---
    if(hypar["mode"]=="train"):
        train(net,
              optimizer,
              train_dataloaders,
              train_datasets,
              valid_dataloaders,
              valid_datasets,
              hypar,
            #   train_dataloaders_val, train_datasets_val
              )
    else:
        valid(net,
              valid_dataloaders,
              valid_datasets,
              hypar)


if __name__ == "__main__":

    ### --------------- STEP 1: Configuring the Train, Valid and Test datasets ---------------
    ## configure the train, valid and inference datasets
    train_datasets, valid_datasets = [], []
    dataset_1, dataset_1 = {}, {}

    data_root_dir = '/root/autodl-tmp/datasets/dis'
    dataset_tr = {"name": "DIS5K-TR",
                 "im_dir": os.path.join(data_root_dir, "DIS5K/DIS-TR/im"),
                 "gt_dir": os.path.join(data_root_dir, "DIS5K/DIS-TR/gt"),
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "cache_dir": os.path.join(data_root_dir, "DIS5K-Cache/DIS-TR")}

    dataset_vd = {"name": "DIS5K-VD",
                 "im_dir": os.path.join(data_root_dir, "DIS5K/DIS-VD/im"),
                 "gt_dir": os.path.join(data_root_dir, "DIS5K/DIS-VD/gt"),
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "cache_dir": os.path.join(data_root_dir, "DIS5K-Cache/DIS-VD")}

    dataset_te1 = {"name": "DIS5K-TE1",
                 "im_dir": os.path.join(data_root_dir, "DIS5K/DIS-TE1/im"),
                 "gt_dir": os.path.join(data_root_dir, "DIS5K/DIS-TE1/gt"),
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "cache_dir": os.path.join(data_root_dir, "DIS5K-Cache/DIS-TE1")}

    dataset_te2 = {"name": "DIS5K-TE2",
                 "im_dir": os.path.join(data_root_dir, "DIS5K/DIS-TE2/im"),
                 "gt_dir": os.path.join(data_root_dir, "DIS5K/DIS-TE2/gt"),
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "cache_dir": os.path.join(data_root_dir, "DIS5K-Cache/DIS-TE2")}

    dataset_te3 = {"name": "DIS5K-TE3",
                 "im_dir": os.path.join(data_root_dir, "DIS5K/DIS-TE3/im"),
                 "gt_dir": os.path.join(data_root_dir, "DIS5K/DIS-TE3/gt"),
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "cache_dir": os.path.join(data_root_dir, "DIS5K-Cache/DIS-TE3")}

    dataset_te4 = {"name": "DIS5K-TE4",
                 "im_dir": os.path.join(data_root_dir, "DIS5K/DIS-TE4/im"),
                 "gt_dir": os.path.join(data_root_dir, "DIS5K/DIS-TE4/gt"),
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "cache_dir": os.path.join(data_root_dir, "DIS5K-Cache/DIS-TE4")}
    ### test your own dataset
    dataset_demo = {"name": "your-dataset",
                 "im_dir":  os.path.join(data_root_dir, "your-dataset/im"),
                 "gt_dir": "",
                 "im_ext": ".jpg",
                 "gt_ext": "",
                 "cache_dir": os.path.join(data_root_dir, "your-dataset/cache")}

    train_datasets = [dataset_tr] ## users can create mutiple dictionary for setting a list of datasets as training set
    # valid_datasets = [dataset_vd] ## users can create mutiple dictionary for setting a list of datasets as vaidation sets or inference sets
    valid_datasets = [dataset_vd] # dataset_vd, dataset_te1, dataset_te2, dataset_te3, dataset_te4] # and hypar["mode"] = "valid" for inference,

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

    main(train_datasets,
         valid_datasets,
         hypar=hypar)
