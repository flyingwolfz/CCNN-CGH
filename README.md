# CCNN-CGH

Code and pretained model (30 loops in DIV2K dataset) for CCNN-CGH, including captured videos, images and corresponding CGHs. Using the pretained model will get the exact results in our paper. 


1 Set up connda environment using:

    conda create -n ccnncgh python=3.9
    conda activate ccnncgh
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    pip install opencv-python
    pip install tqdm
    pip install scipy
    pip install scikit-image
The complexPytorch is from: https://github.com/wavefrontshaping/complexPyTorch

2 Run corresponding ****.py 

our codes include:    
