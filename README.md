# CCNN-CGH

![capture](https://user-images.githubusercontent.com/57349703/173181176-ffaf9eb5-addc-4b95-bb6d-ecd2252f09ea.png)


Code and pretained model (30 loops in DIV2K dataset) for CCNN-CGH, including captured videos, images and corresponding CGHs. Using the pretained model will get the exact results in our paper. This repository is under updating.


1. Set up conda environment using:

       conda create -n ccnncgh python=3.9
       conda activate ccnncgh
       conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
       pip install opencv-python
       pip install tqdm
       pip install scipy
       pip install scikit-image
             

The environment has been tested in Windows 10 and Ubuntu 20.04. We use python 3.9, Pytorch 1.10. opencv is used to operate images, other libraries are not essentional to run our model.

2. python code will be released soon, it will include: CCNN-CGH for 1920 1072 and 4K resolution, mini CCNN-CGH for 4K resolution, zero-padding version CCNN-CGH for 1920 1072 resolution
