# CCNN-CGH

![capture](https://user-images.githubusercontent.com/57349703/173181176-ffaf9eb5-addc-4b95-bb6d-ecd2252f09ea.png)

**<p align="center">
Real-time CGH using CCNN**
</p>

Currently state-of-the-art end-to-end CGH network (proposed in 2021， finished in 2022.2, repository created in 2022.6, latest update in 2022.7). Compared with HoloNet and holo-encoder, we achieve the fasted speed and the best quality using compact CCNN. Moreover, CCNN-CGH is the second 4K capable network and mini CCNN-CGH is the first 4K real-time network！The following tests run using RTX 3080.

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

**paper: now under review in IEEE TVCG.(If it's useful, consider cite our paper!)(Since it has not been accepted by any journal yet, papers based on our work should wait until its acceptance.)**

## 0. Contents

- 1, Python code and pretrained models (30 loops in DIV2K training dataset) for different networks 

  xxxtrain.py will train corresponding network.

  xxxload.py will load trained model and test it on a single picture, return average generation time, PSNR, SSIM, simulated reconstruction.

  xxxpsnrssim.py will test the trained model using 100 samples of DIV2K validation dataset, return average PSNR, SSIM.

- 2,Captured results

  including captured videos, images and corresponding CGHs

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

The environment has been tested in Windows 10 and Ubuntu 20.04 in 2022.7. We use python 3.9, Pytorch 1.10. opencv is used to operate images, other libraries are not essentional to run our model. This environment also supports HoloNet, Holo-encoder.

## 2. Run CCNN-CGH 

For better compatibility with complex values, we use an old version of complexPyTorch from https://github.com/wavefrontshaping/complexPyTorch. Their new version should also work.

Change the file path to DIV2K dataset.

Run corresponding python files. CCNN uses 4init, mini CCNN uses 2init. 

## 3.Comparision with other networks

- Comparision with HoloNet

  Before running HoloNet, make sure your GPU has more than 10GB memory.

  HoloNet is from https://github.com/computational-imaging/neural-holography

  The U-Net used in HoloNet is from https://github.com/vsitzmann/pytorch_prototyping

- Comparision with Holo-encoder

  Holo-encoder is from:https://github.com/THUHoloLab/Holo-encoder

  For pytorch version, refer to https://github.com/flyingwolfz/holoencoder-python-version

- Comparision with tensor holography

  We run tensor holography in Ubuntu 20.04 using their code and pretrained model from https://github.com/liangs111/tensor_holography

## 4.Tips

- 1, It is greatly recommended to use our zero-padding version directly. Because of better quality and easier experiment.

- 2, Try larger CCNN according to your GPU ability. Although more GPU memory is needed, more DS/US layers and more init channels will bring better quality easily. Meanwhile smaller CCNN still performes better than existing networks, reducing parameters also works if you want faster speed and lower cost.

- 3, under updating......
