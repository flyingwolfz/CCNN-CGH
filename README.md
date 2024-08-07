# CCNN-CGH

![capture](https://user-images.githubusercontent.com/57349703/173181176-ffaf9eb5-addc-4b95-bb6d-ecd2252f09ea.png)

**<p align="center">
Real-time CGH using CCNN (zero-padding version)**
</p>

Real-time end-to-end CGH network with average PSNR more than 30dB in DIV2K valitaion dataset. Compared with HoloNet and holo-encoder, we achieve the fasted speed and the best quality using compact CCNN. Moreover, CCNN-CGH is a 4K capable network and mini CCNN-CGH is the first 4K real-time network！The following tests run using RTX 3080.
 
<p align="center">
1920 performance
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/57349703/179686169-69aee351-6af7-4957-8d1e-c1535c1a2ffe.png" alt="1920" width='35%' height='35%'/>
</p>
<p align="center">
4k performance
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/57349703/179686183-ea2af19b-72f4-4ee3-9a60-84cf65800137.png" alt="4k" width='35%' height='35%'/>
</p>

**paper: https://doi.org/10.1109/tvcg.2023.3239670 (If it's useful, consider cite our paper!)**

## 0. Contents

- 1, Python code and pretrained models (30 loops in DIV2K training dataset) for different networks 

  xxxtrain.py will train corresponding network.

  xxxload.py will load trained model and test it on a single picture, return average generation time, PSNR, SSIM, simulated reconstruction.

  xxxpsnrssim.py will test the trained model using 100 samples of DIV2K validation dataset, return average PSNR, SSIM.
  
  **In reorganized codes, some issues are fixed. You can change the ASM version**

- 2,Captured results

  including captured videos, images and corresponding CGHs. If you have the same devices as ours, you can use the CGHs to reproduce our experiments.

## 1. Set up conda environment 

```
conda create -n ccnncgh python=3.9
conda activate ccnncgh
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install opencv-python
pip install tqdm
pip install scipy
pip install scikit-image
```             

The environment has been tested in Windows 10 and Ubuntu 20.04 in 2022.7. We use python 3.9, Pytorch 1.10. 

## 2. Run CCNN-CGH 

For better compatibility with complex values, we use complexPyTorch from https://github.com/wavefrontshaping/complexPyTorch. Their new version should also work.

Change the file path in the code.

Run corresponding python files. CCNN uses 4init, mini CCNN uses 2init. 

## 3.Comparision with other networks

- Comparision with HoloNet

  Before running HoloNet, make sure your GPU has more than 10GB memory.

  HoloNet and some codes are from https://github.com/computational-imaging/neural-holography

  The U-Net used in HoloNet is from https://github.com/vsitzmann/pytorch_prototyping

- Comparision with Holo-encoder

  Holo-encoder is from:https://github.com/THUHoloLab/Holo-encoder

  For pytorch version, we use https://github.com/flyingwolfz/holoencoder-python-version

- Comparision with tensor holography

  We run tensor holography in Ubuntu 20.04 using their code and pretrained model from https://github.com/liangs111/tensor_holography
  
  Matlab code is used for simulation. See ASM: https://github.com/flyingwolfz/angular-spectrum-method



