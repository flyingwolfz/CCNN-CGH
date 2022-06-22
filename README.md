# CCNN-CGH

![capture](https://user-images.githubusercontent.com/57349703/173181176-ffaf9eb5-addc-4b95-bb6d-ecd2252f09ea.png)

*capture results of CCNN-CGH (pretrained models, zero-padding version)*

Using the pretained models will get the exact results in our paper. This repository is under updating.

## 0. Contents

1, Python code and pretained models (30 loops in DIV2K training dataset) for different networks 

xxxtrain.py will train corresponding network.

xxxload.py will load trained model and test it on a single picture, return average generation time, PSNR, SSIM, simulated reconstruction.

xxxpsnrssim.py will test the trained model using 100 samples of DIV2K validation dataset, return average PSNR, SSIM.

2,Captured results

including captured videos, images and corresponding CGHs

## 1. Set up conda environment 

```
conda create -n ccnncgh python=3.9
conda activate ccnncgh
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install opencv-python
pip install tqdm
pip install scipy
pip install scikit-image
```             

The environment has been tested in Windows 10 and Ubuntu 20.04. We use python 3.9, Pytorch 1.10. opencv is used to operate images, other libraries are not essentional to run our model.

## 2.Comparision with HoloNet

Before running HoloNet, make sure your GPU has more than 10GB memory.

HoloNet is from https://github.com/computational-imaging/neural-holography

The U-Net used in HoloNet is from https://github.com/vsitzmann/pytorch_prototyping

## 3.Comparision with Holo-encoder

Holo-encoder is from:https://github.com/THUHoloLab/Holo-encoder

For pytorch version, refer to https://github.com/flyingwolfz/holoencoder-python-version

## 4. Run CCNN-CGH (Next update)
Next update will include: CCNN-CGH for 1920 and 4K resolution, mini CCNN-CGH for 4K resolution, zero-padding version CCNN-CGH for 1072 resolution
