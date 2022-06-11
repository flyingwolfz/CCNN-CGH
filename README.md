# CCNN-CGH

Code and pretained model (30 loops in DIV2K dataset) for CCNN-CGH, including captured videos, images and corresponding CGHs. Using the pretained model will get the exact results in our paper. 


1 Set up conda environment using:

    conda create -n ccnncgh python=3.9
    conda activate ccnncgh
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    pip install opencv-python
    pip install tqdm
    pip install scipy
    pip install scikit-image
    
The complexPytorch for better better compatibility with complex values is from: https://github.com/wavefrontshaping/complexPyTorch

We use python 3.9, Pytorch 1.10. opencv is used to operate images, other libraries are not essentional to run our model.

2 Run corresponding ****.py 

our codes include:    
