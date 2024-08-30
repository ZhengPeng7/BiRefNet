<h1 align="center">Bilateral Reference for High-Resolution Dichotomous Image Segmentation</h1>

<div align='center'>
    <a href='https://scholar.google.com/citations?user=TZRzWOsAAAAJ' target='_blank'><strong>Peng Zheng</strong></a><sup> 1,4,5,6</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=0uPb8MMAAAAJ' target='_blank'><strong>Dehong Gao</strong></a><sup> 2</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=kakwJ5QAAAAJ' target='_blank'><strong>Deng-Ping Fan</strong></a><sup> 1*</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=9cMQrVsAAAAJ' target='_blank'><strong>Li Liu</strong></a><sup> 3</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=qQP6WXIAAAAJ' target='_blank'><strong>Jorma Laaksonen</strong></a><sup> 4</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=pw_0Z_UAAAAJ' target='_blank'><strong>Wanli Ouyang</strong></a><sup> 5</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=stFCYOAAAAAJ' target='_blank'><strong>Nicu Sebe</strong></a><sup> 6</sup>
</div>

<div align='center'>
    <sup>1 </sup>Nankai University&ensp;  <sup>2 </sup>Northwestern Polytechnical University&ensp;  <sup>3 </sup>National University of Defense Technology&ensp; 
    <br />
    <sup>4 </sup>Aalto University&ensp;  <sup>5 </sup>Shanghai AI Laboratory&ensp;  <sup>6 </sup>University of Trento&ensp; 
</div>

<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap;">
  <a href='https://www.sciopen.com/article/pdf/10.26599/AIR.2024.9150038.pdf'><img src='https://img.shields.io/badge/Journal-Paper-red'></a>&ensp; 
  <a href='https://arxiv.org/pdf/2401.03407'><img src='https://img.shields.io/badge/arXiv-Paper-red'></a>&ensp; 
  <a href='https://drive.google.com/file/d/1FWvKDWTnK9RsiywfCsIxsnQzqv-dlO5u/view'><img src='https://img.shields.io/badge/中文版-Paper-red'></a>&ensp; 
  <a href='https://www.birefnet.top'><img src='https://img.shields.io/badge/Page-Project-red'></a>&ensp; 
  <a href='https://drive.google.com/drive/folders/1s2Xe0cjq-2ctnJBR24563yMSCOu4CcxM'><img src='https://img.shields.io/badge/GDrive-Stuff-green'></a>&ensp; 
  <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-yellow'></a>&ensp; 
  <a href='https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HF-Space-blue'></a>&ensp; 
  <a href='https://huggingface.co/ZhengPeng7/BiRefNet'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model-blue'></a>&ensp; 
</div>

<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap;">
  <a href='https://colab.research.google.com/drive/14Dqg7oeBkFEtchaHLNpig2BcdkZEogba'><img src='https://img.shields.io/badge/Multiple_Images_Inference-F9AB00?style=for-the-badge&logo=googlecolab&color=525252'></a>&ensp; 
  <a href='https://colab.research.google.com/drive/1MaEiBfJ4xIaZZn0DqKrhydHB8X97hNXl'><img src='https://img.shields.io/badge/Inference_&_Evaluation-F9AB00?style=for-the-badge&logo=googlecolab&color=525252'></a>&ensp; 
  <a href='https://colab.research.google.com/drive/1B6aKZ3ekcvKMkSBn0N5mCASLUYMp0whK'><img src='https://img.shields.io/badge/Box_Guided_Segmentation-F9AB00?style=for-the-badge&logo=googlecolab&color=525252'></a>&ensp; 
</div>


|            *DIS-Sample_1*        |             *DIS-Sample_2*        |
| :------------------------------: | :-------------------------------: |
| <img src="https://drive.google.com/thumbnail?id=1ItXaA26iYnE8XQ_GgNLy71MOWePoS2-g&sz=w400" /> |  <img src="https://drive.google.com/thumbnail?id=1Z-esCujQF_uEa_YJjkibc3NUrW4aR_d4&sz=w400" /> |

This repo is the official implementation of "[**Bilateral Reference for High-Resolution Dichotomous Image Segmentation**](https://arxiv.org/pdf/2401.03407)" (___CAAI AIR 2024___).

> [!note]
> **We need more GPU resources** to push forward the performance of BiRefNet, especially on *matting* tasks, higher-resolution inference (*2K*), and more *efficient* model design. If you are happy to cooperate, please contact me at zhengpeng0108@gmail.com.

## News :newspaper:
* **`Aug 30, 2024`:** We uploaded notebooks in `tutorials` to run the inference and ONNX conversion locally.
* **`Aug 23, 2024`:** Our BiRefNet is now officially released [online](https://www.sciopen.com/article/10.26599/AIR.2024.9150038) on CAAI AIR journal. And thanks to the [press release](https://www.eurekalert.org/news-releases/1055380).
* **`Aug 19, 2024`:** We uploaded the ONNX model files of all weights in the [GitHub release](https://github.com/ZhengPeng7/BiRefNet/releases/tag/v1) and [GDrive folder](https://drive.google.com/drive/u/0/folders/1kZM55bwsRdS__bdnsXpkmH6QPyza-9-N). Check out the **ONNX conversion** part in [model zoo](https://github.com/ZhengPeng7/BiRefNet?tab=readme-ov-file#model-zoo) for more details.
* **`Jul 30, 2024`:** Thanks to @not-lain for his kind efforts in adding BiRefNet to the official huggingface.js [repo](https://github.com/huggingface/huggingface.js/blob/3a8651fbc6508920475564a692bf0e5b601d9343/packages/tasks/src/model-libraries-snippets.ts#L763).
* **`Jul 28, 2024`:** We released the [Colab demo for box-guided segmentation](https://colab.research.google.com/drive/1B6aKZ3ekcvKMkSBn0N5mCASLUYMp0whK).
* **`Jul 15, 2024`:** We deployed our BiRefNet on [Hugging Face Models](https://huggingface.co/ZhengPeng7/BiRefNet) for users to easily load it in one line code.
* **`Jun 21, 2024`:** We released and uploaded the Chinese version of our original paper to my [GDrive](https://drive.google.com/file/d/1aBnJ_R9lbnC2dm8dqD0-pzP2Cu-U1Xpt/view).
* **`May 28, 2024`:** We hold a [model zoo](https://github.com/ZhengPeng7/BiRefNet?tab=readme-ov-file#model-zoo) with well-trained weights of our BiRefNet in different sizes and for different tasks, including general use, matting segmentation, DIS, HRSOD, COD, etc.
* **`May 7, 2024`:**  We also released the [Colab demo for multiple images inference](https://colab.research.google.com/drive/14Dqg7oeBkFEtchaHLNpig2BcdkZEogba). Many thanks to @rishabh063 for his support on it.
* **`Apr 9, 2024`:**  Thanks to [Features and Labels Inc.](https://fal.ai/) for deploying a cool online BiRefNet [inference API](https://fal.ai/models/fal-ai/birefnet/playground) and providing me with strong GPU resources for 4 months on more extensive experiments!
* **`Mar 7, 2024`:**  We released BiRefNet codes, the well-trained weights for all tasks in the original papers, and all related stuff in my [GDrive folder](https://drive.google.com/drive/folders/1s2Xe0cjq-2ctnJBR24563yMSCOu4CcxM). Meanwhile, we also deployed our BiRefNet on [Hugging Face Spaces](https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo) for easier online use and released the [Colab demo for inference and evaluation](https://colab.research.google.com/drive/1MaEiBfJ4xIaZZn0DqKrhydHB8X97hNXl).
* **`Jan 7, 2024`:**  We released our paper on [arXiv](https://arxiv.org/pdf/2401.03407).


## :rocket: Load BiRefNet in _ONE LINE_ by HuggingFace, check more: [![BiRefNet](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/ZhengPeng7/birefnet)
```python
from transformers import AutoModelForImageSegmentation
birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
```
## :flight_arrival: Inference Partner:
We are really happy to collaborate with [FAL](https://fal.ai) to deploy the **inference API** of BiRefNet. You can access this service via the link below:
+ https://fal.ai/models/fal-ai/birefnet

Our BiRefNet has achieved SOTA on many similar HR tasks:

**DIS**: [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/dichotomous-image-segmentation-on-dis-te1)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te1?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/dichotomous-image-segmentation-on-dis-te2)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te2?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/dichotomous-image-segmentation-on-dis-te3)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te3?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/dichotomous-image-segmentation-on-dis-te4)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-te4?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/dichotomous-image-segmentation-on-dis-vd)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-vd?p=bilateral-reference-for-high-resolution)

<details><summary>Figure of Comparison on DIS Papers with Codes (by the time of this work):</summary>
<img src="https://drive.google.com/thumbnail?id=1DLt6CFXdT1QSWDj_6jRkyZINXZ4vmyRp&sz=w1620" />
<img src="https://drive.google.com/thumbnail?id=1gn5GyKFlJbMIkre1JyEdHDSYcrFmcLD0&sz=w1620" />
<img src="https://drive.google.com/thumbnail?id=16CVYYOtafEeZhHqv0am2Daku1n_exMP6&sz=w1620" />
<img src="https://drive.google.com/thumbnail?id=10K45xwPXmaTG4Ex-29ss9payA9yBnyLn&sz=w1620" />
<img src="https://drive.google.com/thumbnail?id=16EuyqKFJOqwMmagvfnbC9hUurL9pYLLB&sz=w1620" />
</details>
<br />

**COD**:[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/camouflaged-object-segmentation-on-cod)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-cod?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/camouflaged-object-segmentation-on-nc4k)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-nc4k?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/camouflaged-object-segmentation-on-camo)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-camo?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/camouflaged-object-segmentation-on-chameleon)](https://paperswithcode.com/sota/camouflaged-object-segmentation-on-chameleon?p=bilateral-reference-for-high-resolution)

<details><summary>Figure of Comparison on COD Papers with Codes (by the time of this work):</summary>
<img src="https://drive.google.com/thumbnail?id=1DLt6CFXdT1QSWDj_6jRkyZINXZ4vmyRp&sz=w1620" />
<img src="https://drive.google.com/thumbnail?id=1gn5GyKFlJbMIkre1JyEdHDSYcrFmcLD0&sz=w1620" />
<img src="https://drive.google.com/thumbnail?id=16CVYYOtafEeZhHqv0am2Daku1n_exMP6&sz=w1620" />
</details>
<br />

**HRSOD**: [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/rgb-salient-object-detection-on-davis-s)](https://paperswithcode.com/sota/rgb-salient-object-detection-on-davis-s?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/rgb-salient-object-detection-on-hrsod)](https://paperswithcode.com/sota/rgb-salient-object-detection-on-hrsod?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/rgb-salient-object-detection-on-uhrsd)](https://paperswithcode.com/sota/rgb-salient-object-detection-on-uhrsd?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/salient-object-detection-on-duts-te)](https://paperswithcode.com/sota/salient-object-detection-on-duts-te?p=bilateral-reference-for-high-resolution) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bilateral-reference-for-high-resolution/salient-object-detection-on-dut-omron)](https://paperswithcode.com/sota/salient-object-detection-on-dut-omron?p=bilateral-reference-for-high-resolution)

<details><summary>Figure of Comparison on HRSOD Papers with Codes (by the time of this work):</summary>
<img src="https://drive.google.com/thumbnail?id=1hNfQtlTAHT4-AVbk_47852zyRp1NOFLs&sz=w1620" />
<img src="https://drive.google.com/thumbnail?id=1bcVldUAxYkMI3OMTyaP_jNuOugDfYj-d&sz=w1620" />
<img src="https://drive.google.com/thumbnail?id=1p1zgyVz27cGEqQMtOKzm_6zoYK3Sw_Zk&sz=w1620" />
<img src="https://drive.google.com/thumbnail?id=1TubAvcoEbH_mHu3I-AxflnB71nkf35jJ&sz=w1620" />
<img src="https://drive.google.com/thumbnail?id=1A3V9HjVtcMQdnGPwuy-DBVhwKuo0q2lT&sz=w1620" />
</details>
<br />

#### Try our online demos for inference:

+ **Inference and evaluation** of your given weights: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MaEiBfJ4xIaZZn0DqKrhydHB8X97hNXl)
+ **Online Inference with GUI** with adjustable resolutions: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo)  
+ Online **Multiple Images Inference** on Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14Dqg7oeBkFEtchaHLNpig2BcdkZEogba)

<img src="https://drive.google.com/thumbnail?id=12XmDhKtO1o2fEvBu4OE4ULVB2BK0ecWi&sz=w1620" />



## Model Zoo

> For more general use of our BiRefNet, I extended the original academic one to more general ones for better real-life application.
>
> Datasets and datasets are suggested to be downloaded from official pages. But you can also download the packaged ones: [DIS](https://drive.google.com/drive/folders/1hZW6tAGPJwo9mPS7qGGGdpxuvuXiyoMJ), [HRSOD](https://drive.google.com/drive/folders/18_hAE3QM4cwAzEAKXuSNtKjmgFXTQXZN), [COD](https://drive.google.com/drive/folders/1EyHmKWsXfaCR9O0BiZEc3roZbRcs4ECO), [Backbones](https://drive.google.com/drive/folders/1cmce_emsS8A5ha5XT2c_CZiJzlLM81ms).
>
> Find performances (almost all metrics) of all models in the `exp-TASK_SETTINGS` folders in [[**stuff**](https://drive.google.com/drive/folders/1s2Xe0cjq-2ctnJBR24563yMSCOu4CcxM)].



<details><summary>Models in the original paper, for <b>comparison on benchmarks</b>:</summary>

| Task  |        Training Sets        |   Backbone    |                           Download                           |
| :---: | :-------------------------: | :-----------: | :----------------------------------------------------------: |
|  DIS  |          DIS5K-TR           | swin_v1_large | [google-drive](https://drive.google.com/file/d/1J90LucvDQaS3R_-9E7QUh1mgJ8eQvccb/view) |
|  COD  |     COD10K-TR, CAMO-TR      | swin_v1_large | [google-drive](https://drive.google.com/file/d/1tM5M72k7a8aKF-dYy-QXaqvfEhbFaWkC/view) |
| HRSOD |           DUTS-TR           | swin_v1_large | [google-drive](https://drive.google.com/file/d/1f7L0Pb1Y3RkOMbqLCW_zO31dik9AiUFa/view) |
| HRSOD |          HRSOD-TR           | swin_v1_large |                         google-drive                         |
| HRSOD |          UHRSD-TR           | swin_v1_large |                         google-drive                         |
| HRSOD |      DUTS-TR, HRSOD-TR      | swin_v1_large | [google-drive](https://drive.google.com/file/d/1WJooyTkhoDLllaqwbpur_9Hle0XTHEs_/view) |
| HRSOD |      DUTS-TR, UHRSD-TR      | swin_v1_large | [google-drive](https://drive.google.com/file/d/1Pu1mv3ORobJatIuUoEuZaWDl2ylP3Gw7/view) |
| HRSOD |     HRSOD-TR, UHRSD-TR      | swin_v1_large | [google-drive](https://drive.google.com/file/d/1xEh7fsgWGaS5c3IffMswasv0_u-aVM9E/view) |
| HRSOD | DUTS-TR, HRSOD-TR, UHRSD-TR | swin_v1_large | [google-drive](https://drive.google.com/file/d/13FaxyyOwyCddfZn2vZo1xG1KNZ3cZ-6B/view) |

</details>



<details><summary>Models trained with customed data (general, matting), for <b>general use in practical application</b>:</summary>

|           Task            |                        Training Sets                         |   Backbone    | Test Set  | Metric (S, wF[, HCE]) |                           Download                           |
| :-----------------------: | :----------------------------------------------------------: | :-----------: | :-------: | :-------------------: | :----------------------------------------------------------: |
|      **general use**      | DIS5K-TR,DIS-TEs, DUTS-TR_TE,HRSOD-TR_TE,UHRSD-TR_TE, HRS10K-TR_TE, TR-P3M-10k, TE-P3M-500-NP, TE-P3M-500-P, TR-humans | swin_v1_large |  DIS-VD   |  0.911, 0.875, 1069   | [google-drive](https://drive.google.com/file/d/1_IfUnu8Fpfn-nerB89FzdNXQ7zk6FKxc/view) |
|      **general use**      | DIS5K-TR,DIS-TEs, DUTS-TR_TE,HRSOD-TR_TE,UHRSD-TR_TE, HRS10K-TR_TE, TR-P3M-10k, TE-P3M-500-NP, TE-P3M-500-P, TR-humans | swin_v1_tiny |  DIS-VD   |  0.882, 0.830, 1175   | [google-drive](https://drive.google.com/file/d/1fzInDWiE2n65tmjaHDSZpqhL0VME6-Yl/view) |
|      **general use**      |                      DIS5K-TR, DIS-TEs                       | swin_v1_large |  DIS-VD   |  0.907, 0.865, 1059   | [google-drive](https://drive.google.com/file/d/1P6NJzG3Jf1sl7js2q1CPC3yqvBn_O8UJ/view) |
| **matting segmentation** |                           [P3M-10k](https://github.com/JizhiziLi/P3M), [humans](https://huggingface.co/datasets/schirrmacher/humans)                            | swin_v1_large | P3M-500-P |     0.983, 0.989      | [google-drive](https://drive.google.com/file/d/1uUeXjEUoD2XF_6YjD_fsct-TJp7TFiqh) |

</details>



<details><summary>Segmentation with box <b>guidance</b>:</summary>

+ Given box guidance: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1B6aKZ3ekcvKMkSBn0N5mCASLUYMp0whK)

</details>



<details><summary>Model <b>efficiency</b>:</summary>

> Screenshot from the original paper. All tests are conducted on a single A100 GPU.

<img src="https://drive.google.com/thumbnail?id=1mTfSD_qt-rFO1t8DRQcyIa5cgWLf1w2-&sz=h300" />  <img src="https://drive.google.com/thumbnail?id=1F_OURIWILVe4u1rSz-aqt6ur__bAef25&sz=h300" />

</details>



<details><summary><b>ONNX</b> conversion:</summary>

> We converted from `.pth` weights files to `.onnx` files.  
> We referred a lot to the [Kazuhito00/BiRefNet-ONNX-Sample](https://github.com/Kazuhito00/BiRefNet-ONNX-Sample), many thanks to @Kazuhito00.

+ Check our [Colab demo for ONNX conversion](https://colab.research.google.com/drive/1z6OruR52LOvDDpnp516F-N4EyPGrp5om) or the [notebook file for local running](https://drive.google.com/file/d/1cgL2qyvOO5q3ySfhytypX46swdQwZLrJ), where you can do the conversion/inference by yourself and find all relevant info.
+ As tested, BiRefNets with SwinL (default backbone) cost `~90%` more time (the inference costs `~165ms` on an A100 GPU) using ONNX files. Meanwhile, BiRefNets with SwinT (lightweight) cost `~75%` more time (the inference costs `~93.8ms` on an A100 GPU) using ONNX files. Input resolution is `1024x1024` as default.
+ The results of the original pth files and the converted onnx files are slightly different, which is acceptable.
+ Pay attention to the compatibility among `onnxruntime-gpu, CUDA, and CUDNN` (we use `torch==2.0.1, cuda=11.8` here).


</details>

## Third-Party Creations

> Concerning edge devices with less computing power, we provide a lightweight version with `swin_v1_tiny` as the backbone, which is x4+ faster and x5+ smaller. The details can be found in [this issue](https://github.com/ZhengPeng7/BiRefNet/issues/11#issuecomment-2041033576) and links there.

We found there've been some 3rd party applications based on our BiRefNet. Many thanks for their contribution to the community!  
Choose the one you like to try with clicks instead of codes:  
1. **Applications**:
   + Thanks [**lbq779660843/BiRefNet-Tensorrt**](https://github.com/lbq779660843/BiRefNet-Tensorrt) and [**yuanyang1991/birefnet_tensorrt**](https://github.com/yuanyang1991/birefnet_tensorrt): they both provided the project to convert BiRefNet to **TensorRT**, which is faster and better for deployment. Their repos offer solid local establishment (Win and Linux) and [colab demo](https://colab.research.google.com/drive/1r8GkFPyMMO0OkMX6ih5FjZnUCQrl2SHV?usp=sharing), respectively. And @yuanyang1991 kindly offered the comparison among the inference efficiency of naive PyTorch, ONNX, and TensorRT on an RTX 4080S:

| Methods | [Pytorch](https://drive.google.com/file/d/1_IfUnu8Fpfn-nerB89FzdNXQ7zk6FKxc/view) | [ONNX](https://drive.google.com/drive/u/0/folders/1kZM55bwsRdS__bdnsXpkmH6QPyza-9-N) | TensorRT |
|:------------------------------------------------------------------------------------:|:--------------:|:--------------:|:--------------:|
|  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;First Inference Time&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |     0.71s      |     5.32s      |     **0.17s**      |

| Methods | [Pytorch](https://drive.google.com/file/d/1_IfUnu8Fpfn-nerB89FzdNXQ7zk6FKxc/view) | [ONNX](https://drive.google.com/drive/u/0/folders/1kZM55bwsRdS__bdnsXpkmH6QPyza-9-N) | TensorRT |
|:------------------------------------------------------------------------------------:|:--------------:|:--------------:|:--------------:|
|  Avg Inf Time (excluding 1st)   |     0.15s      |     4.43s      |     **0.11s**      |

   + Thanks [**dimitribarbot/sd-webui-birefnet**](https://github.com/dimitribarbot/sd-webui-birefnet): this project allows to add a BiRefNet section to the original **Stable Diffusion WebUI**'s Extras tab.
     <p align="center"><img src="https://drive.google.com/thumbnail?id=159bLXI71FWh4ZsHTvc-wApSN9ytVRmua&sz=w1620" /></p>

   + Thanks [**fal.ai/birefnet**](https://fal.ai/models/birefnet): this project on `fal.ai` encapsulates BiRefNet **online** with more useful options in **UI** and **API** to call the model.
     <p align="center"><img src="https://drive.google.com/thumbnail?id=1rNk81YV_Pzb2GykrzfGvX6T7KBXR0wrA&sz=w1620" /></p>

   + Thanks [**ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO**](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO): this project further improves the **UI** for BiRefNet in ComfyUI, especially for **video data**.
     <p align="center"><img src="https://drive.google.com/thumbnail?id=1GOqEreyS7ENzTPN0RqxEjaA76RpMlkYM&sz=w1620" /></p>
     
     <https://github.com/ZhengPeng7/BiRefNet/assets/25921713/3a1c7ab2-9847-4dac-8935-43a2d3cd2671>

   + Thanks [**viperyl/ComfyUI-BiRefNet**](https://github.com/viperyl/ComfyUI-BiRefNet): this project packs BiRefNet as **ComfyUI nodes**, and makes this SOTA model easier use for everyone.
     <p align="center"><img src="https://drive.google.com/thumbnail?id=1KfxCQUUa2y9T-aysEaeVVjCUt3Z0zSkL&sz=w1620" /></p>

   + Thanks [**Rishabh**](https://github.com/rishabh063) for offering a demo for the [easier multiple images inference on colab](https://colab.research.google.com/drive/14Dqg7oeBkFEtchaHLNpig2BcdkZEogba).

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
conda create -n birefnet python=3.9 -y && conda activate birefnet
pip install -r requirements.txt
```

#### Dataset Preparation

Download combined training / test sets I have organized well from: [DIS](https://drive.google.com/drive/folders/1hZW6tAGPJwo9mPS7qGGGdpxuvuXiyoMJ)--[COD](https://drive.google.com/drive/folders/1EyHmKWsXfaCR9O0BiZEc3roZbRcs4ECO)--[HRSOD](https://drive.google.com/drive/folders/18_hAE3QM4cwAzEAKXuSNtKjmgFXTQXZN) or the single official ones in the `single_ones` folder, or their official pages. You can also find the same ones on my **BaiduDisk**: [DIS](https://pan.baidu.com/s/1O_pQIGAE4DKqL93xOxHpxw?pwd=PSWD)--[COD](https://pan.baidu.com/s/1RnxAzaHSTGBC1N6r_RfeqQ?pwd=PSWD)--[HRSOD](https://pan.baidu.com/s/1_Del53_0lBuG0DKJJAk4UA?pwd=PSWD).

#### Weights Preparation

Download backbone weights from [my google-drive folder](https://drive.google.com/drive/folders/1s2Xe0cjq-2ctnJBR24563yMSCOu4CcxM) or their official pages.

## Run

```shell
# Train & Test & Evaluation
./train_test.sh RUN_NAME GPU_NUMBERS_FOR_TRAINING GPU_NUMBERS_FOR_TEST
# Example: ./train_test.sh tmp-proj 0,1,2,3,4,5,6,7 0

# See train.sh / test.sh for only training / test-evaluation.
# After the evaluation, run `gen_best_ep.py` to select the best ckpt from a specific metric (you choose it from Sm, wFm, HCE (DIS only)).
```

#### Well-trained weights:

Download the `BiRefNet-{TASK}-{EPOCH}.pth` from [[**stuff**](https://drive.google.com/drive/folders/1s2Xe0cjq-2ctnJBR24563yMSCOu4CcxM)]. Info of the corresponding (predicted\_maps/performance/training\_log) weights can be also found in folders like `exp-BiRefNet-{TASK_SETTINGS}` in the same directory.

You can also download the weights from the release of this repo.

The results might be a bit different from those in the original paper, you can see them in the `eval_results-BiRefNet-{TASK_SETTINGS}` folder in each `exp-xx`, we will update them in the following days. Due to the very high cost I used (A100-80G x 8) which many people cannot afford to (including myself....),  I re-trained BiRefNet on a single A100-40G only and achieve the performance on the same level (even better). It means you can directly train the model on a single GPU with 36.5G+ memory. BTW, 5.5G GPU memory is needed for inference in 1024x1024. (I personally paid a lot for renting an A100-40G to re-train BiRefNet on the three tasks... T_T. Hope it can help you.)

But if you have more and more powerful GPUs, you can set GPU IDs and increase the batch size in `config.py` to accelerate the training. We have made all this kind of things adaptive in scripts to seamlessly switch between single-card training and multi-card training. Enjoy it :)

#### Some of my messages:

This project was originally built for DIS only. But after the updates one by one, I made it larger and larger with many functions embedded together. Finally, you can **use it for any binary image segmentation tasks**, such as DIS/COD/SOD, medical image segmentation, anomaly segmentation, etc. You can eaily open/close below things (usually in `config.py`):
+ Multi-GPU training: open/close with one variable.
+ Backbone choices: Swin_v1, PVT_v2, ConvNets, ...
+ Weighted losses: BCE, IoU, SSIM, MAE, Reg, ...
+ Adversarial loss for binary segmentation (proposed in my previous work [MCCL](https://arxiv.org/pdf/2302.14485)).
+ Training tricks: multi-scale supervision, freezing backbone, multi-scale input...
+ Data collator: loading all in memory, smooth combination of different datasets for combined training and test.
+ ...
I really hope you enjoy this project and use it in more works to achieve new SOTAs.


### Quantitative Results

<p align="center"><img src="https://drive.google.com/thumbnail?id=1Ymkh8WN16XMTBOS8dmPTg5eAf-NIl2m5&sz=w1620" /></p>

<p align="center"><img src="https://drive.google.com/thumbnail?id=1W0mi0ZiYbqsaGuohNXU8Gh7Zj4M3neFg&sz=w1620" /></p>



### Qualitative Results

<p align="center"><img src="https://drive.google.com/thumbnail?id=1TYZF8pVZc2V0V6g3ik4iAr9iKvJ8BNrf&sz=w1620" /></p>

<p align="center"><img src="https://drive.google.com/thumbnail?id=1ZGHC32CAdT9cwRloPzOCKWCrVQZvUAlJ&sz=w1620" /></p>



### Citation

```
@article{zheng2024birefnet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  volume = {3},
  pages = {9150038},
  year={2024}
}
```



## Contact

Any questions, discussions, or even complaints, feel free to leave issues here or send me e-mails (zhengpeng0108@gmail.com). You can also join the Discord Group (https://discord.gg/d9NN5sgFrq) or QQ Group (https://qm.qq.com/q/y6WPy7WOIK) if you want to talk a lot publicly.

