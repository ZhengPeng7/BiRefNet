# <p align=center>`Bilateral Reference for High-Resolution Dichotomous Image Segmentation`</p>

|            *DIS-Sample_1*        |             *DIS-Sample_2*        |
| :------------------------------: | :-------------------------------: |
| <img src="https://drive.google.com/thumbnail?id=1ItXaA26iYnE8XQ_GgNLy71MOWePoS2-g&sz=w400" /> |  <img src="https://drive.google.com/thumbnail?id=1Z-esCujQF_uEa_YJjkibc3NUrW4aR_d4&sz=w400" /> |

This repo is the official implementation of "[**Bilateral Reference for High-Resolution Dichotomous Image Segmentation**](https://arxiv.org/pdf/2401.03407.pdf)" (___arXiv 2024___).

> **Authors:**
> [Peng Zheng](https://scholar.google.com/citations?user=TZRzWOsAAAAJ),
> [Dehong Gao](https://scholar.google.com/citations?user=0uPb8MMAAAAJ),
> [Deng-Ping Fan](https://scholar.google.com/citations?user=kakwJ5QAAAAJ),
> [Li Liu](https://scholar.google.com/citations?user=9cMQrVsAAAAJ),
> [Jorma Laaksonen](https://scholar.google.com/citations?user=qQP6WXIAAAAJ),
> [Wanli Ouyang](https://scholar.google.com/citations?user=pw_0Z_UAAAAJ), &
> [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ).


[[**arXiv**](https://arxiv.org/abs/2401.03407)] [[**code**](https://github.com/ZhengPeng7/BiRefNet)] [[**stuff**](https://drive.google.com/drive/u/0/folders/1s2Xe0cjq-2ctnJBR24563yMSCOu4CcxM)] 

Our BiRefNet has achieved SOTA on many similar HR tasks:

 **DIS**: [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/dichotomous-image-segmentation-on-dis-te1)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te1?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/dichotomous-image-segmentation-on-dis-te2)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te2?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/dichotomous-image-segmentation-on-dis-te3)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te3?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/dichotomous-image-segmentation-on-dis-te4)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te4?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/dichotomous-image-segmentation-on-dis-vd)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-vd?p=bilateral-reference-for-high-resolution)

**COD**:[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/camouflaged-object-segmentation-on-cod)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-cod?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/camouflaged-object-segmentation-on-nc4k)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-nc4k?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/camouflaged-object-segmentation-on-camo)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-camo?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/camouflaged-object-segmentation-on-chameleon)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-chameleon?p=bilateral-reference-for-high-resolution)

**HRSOD**: [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/rgb-salient-object-detection-on-davis-s)](https://paperswithcode.com/sota/rgb-salient-object-detection-on-davis-s?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/rgb-salient-object-detection-on-hrsod)](https://paperswithcode.com/sota/rgb-salient-object-detection-on-hrsod?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/rgb-salient-object-detection-on-uhrsd)](https://paperswithcode.com/sota/rgb-salient-object-detection-on-uhrsd?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/salient-object-detection-on-duts-te)](https://paperswithcode.com/sota/salient-object-detection-on-duts-te?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/salient-object-detection-on-dut-omron)](https://paperswithcode.com/sota/salient-object-detection-on-dut-omron?p=bilateral-reference-for-high-resolution)

#### Try our online demos for inference:

+ **Inference and evaluation** of your given weights: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MaEiBfJ4xIaZZn0DqKrhydHB8X97hNXl#scrollTo=DJ4meUYjia6S)
+ **Online Inference with GUI** with adjustable resolutions: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo)  
<img src="https://drive.google.com/thumbnail?id=12XmDhKtO1o2fEvBu4OE4ULVB2BK0ecWi&sz=w1620" />

## Third-Party Creations
> Concerning edge devices with less computing power, we provide a lightweight version with `swin_v1_tiny` as the backbone, which is x4+ faster and x5+ smaller. The details can be found in [this issue](https://github.com/ZhengPeng7/BiRefNet/issues/11#issuecomment-2041033576) and links there.

We found there've been some 3rd party applications based on our BiRefNet. Many thanks for their contribution to the community!  
Choose the one you like to try with clicks instead of codes:  
1. **Applications**:
   + Thanks [**fal.ai/birefnet**](https://fal.ai/models/birefnet): this project on `fal.ai` encapsulates BiRefNet **online** with more useful options in **UI** and **API** to call the model.
     <p align="center"><img src="https://drive.google.com/thumbnail?id=1rNk81YV_Pzb2GykrzfGvX6T7KBXR0wrA&sz=w1620" /></p>

   + Thanks [**ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO**](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO): this project further improves the **UI** for BiRefNet in ComfyUI, especially for **video data**.
     <p align="center"><img src="https://drive.google.com/thumbnail?id=1GOqEreyS7ENzTPN0RqxEjaA76RpMlkYM&sz=w1620" /></p>
     
     <https://github.com/ZhengPeng7/BiRefNet/assets/25921713/3a1c7ab2-9847-4dac-8935-43a2d3cd2671>

   + Thanks [**viperyl/ComfyUI-BiRefNet**](https://github.com/viperyl/ComfyUI-BiRefNet): this project packs BiRefNet as **ComfyUI nodes**, and makes this SOTA model easier use for everyone.
     <p align="center"><img src="https://drive.google.com/thumbnail?id=1KfxCQUUa2y9T-aysEaeVVjCUt3Z0zSkL&sz=w1620" /></p>

   + Thanks [**Rishabh**](https://github.com/rishabh063) for offerring a demo for the [easier single image inference on colab](https://colab.research.google.com/drive/14Dqg7oeBkFEtchaHLNpig2BcdkZEogba?usp=drive_link).

2. **More Visual Comparisons**
   + Thanks [**twitter.com/ZHOZHO672070**](https://twitter.com/ZHOZHO672070) for the comparison with more background-removal methods in images:

     <img src="https://drive.google.com/thumbnail?id=1nvVIFt_Ezs-crPSQxUDqkUBz598fTe63&sz=w1620" />

   + Thanks [**twitter.com/toyxyz3**](https://twitter.com/toyxyz3) for the comparison with more background-removal methods in videos:

    <https://github.com/ZhengPeng7/BiRefNet/assets/25921713/40136198-01cc-4106-81f9-81c985f02e31>

    <https://github.com/ZhengPeng7/BiRefNet/assets/25921713/1a32860c-0893-49dd-b557-c2e35a83c160>


## Usage

#### Environment Setup

```shell
# PyTorch==2.0.1 is used for faster training with compilation.
conda create -n dis python=3.9 -y && conda activate dis
pip install -r requirements.txt
```

#### Dataset Preparation

Download combined training / test sets I have organized well from: [DIS](https://drive.google.com/drive/u/0/folders/1hZW6tAGPJwo9mPS7qGGGdpxuvuXiyoMJ)--[COD](https://drive.google.com/drive/u/0/folders/1EyHmKWsXfaCR9O0BiZEc3roZbRcs4ECO)--[HRSOD](https://drive.google.com/drive/u/0/folders/18_hAE3QM4cwAzEAKXuSNtKjmgFXTQXZN) or the single official ones in the `single_ones` folder, or their official pages. You can also find the same ones on my **BaiduDisk**: [DIS](https://pan.baidu.com/s/1O_pQIGAE4DKqL93xOxHpxw?pwd=PSWD)--[COD](https://pan.baidu.com/s/1RnxAzaHSTGBC1N6r_RfeqQ?pwd=PSWD)--[HRSOD](https://pan.baidu.com/s/1_Del53_0lBuG0DKJJAk4UA?pwd=PSWD).

#### Weights Preparation

Download backbone weights from [my google-drive folder](https://drive.google.com/drive/u/0/folders/1s2Xe0cjq-2ctnJBR24563yMSCOu4CcxM) or their official pages.

#### Run

```shell
# Train & Test & Evaluation
./train_test.sh RUN_NAME GPU_NUMBERS_FOR_TRAINING GPU_NUMBERS_FOR_TEST
# See train.sh / test.sh for only training / test-evaluation.
# After the evluation, run `gen_best_ep.py` to select the best ckpt from a specific metric (you choose it from Sm, wFm, HCE (DIS only)).
```

#### Well-trained weights:

Download the `BiRefNet_{TASK}_{EPOCH}.pth` from [[**stuff**](https://drive.google.com/drive/u/0/folders/1s2Xe0cjq-2ctnJBR24563yMSCOu4CcxM)].

The results might be a bit different from those in the original paper, you can see them in the `performances_all_ckpts` folder in **stuff**. Due to the very high cost I used (A100-80G x 8) which many people cannot afford to (including myself....),  I re-trained BiRefNet on a single A100-40G only and achieve the performance on the same level (even better). It means you can directly train the model on a single GPU with 36.5G+ memory. BTW, 5.5G GPU memory is needed for inference in 1024x1024. (I personally paid a lot for renting an A100-40G to re-train BiRefNet on the three tasks... T_T. Hope it can help you.)

But if you have more and more powerful GPUs, you can set GPU IDs and increase the batch size in `config.py` to accelerate the training. We have made all this kind of things adaptive in scripts to seamlessly switch between single-card training and multi-card training. Enjoy it :)

#### Some of my messages:

This project was originally built for DIS only. But after the updates one by one, I made it larger and larger with many functions embedded together. Finally, you can **use it for any binary image segmentation tasks**, such as DIS/COD/SOD, medical image segmentation, anomaly segmentation, etc. You can eaily open/close below things (usually in `config.py`):
+ Multi-GPU training: open/close with one variable.
+ Backbone choices: Swin_v1, PVT_v2, ConvNets, ...
+ Weighted losses: BCE, IoU, SSIM, MAE, Reg, ...
+ Adversarial loss for binary segmentation (proposed in my previous work [MCCL](https://arxiv.org/pdf/2302.14485.pdf)).
+ Training tricks: multi-scale supervision, freezing backbone, multi-scale input...
+ Data collator: loading all in memory, smooth combination of different datasets for combined training and test.
+ ...
I really hope you enjoy this project and use it in more works to achieve new SOTAs.


### Quantitative Results

![quan_dis](./materials/quan_dis.png)

![quan_cod_hrsod](./materials/quan_cod_hrsod.png)



### Qualitative Results

![qual_dis](./materials/qual_dis.png)

![qual_cod](./materials/qual_cod.png)



### Citation

```
@article{zheng2024birefnet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={arXiv},
  year={2024}
}
```



## Contact

Any question, discussion or even complaint, feel free to leave issues here or send me e-mails (zhengpeng0108@gmail.com).
