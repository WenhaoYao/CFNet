# CFNet：Conditional Filter Learning with Dynamic Noise Estimation for Real Image Denoising
A mainstream type of the state of the arts (SOTAs) based on convolutional neural network (CNN) for real image denoising contains two sub-problems, i.e., noise estimation and non-blind denoising. This paper considers real noise approximated by heteroscedastic Gaussian/Poisson Gaussian distributions with in-camera signal processing pipelines. The related works always exploit the estimated noise prior via channel-wise concatenation followed by a convolutional layer with spatially sharing kernels. Due to the variable modes of noise strength and frequency details of all feature positions, this design cannot adaptively tune the corresponding denoising patterns. To address this problem, we propose a novel conditional filter in which the optimal kernels for different feature positions can be adaptively inferred by local features from the image and the noise map. Also, we bring the thought that alternatively performs noise estimation and non-blind denoising into CNN structure, which continuously updates noise prior to guide the iterative feature denoising. In addition, according to the property of heteroscedastic Gaussian distribution, a novel affine transform block is designed to predict the stationary noise component and the signal-dependent noise component. Compared with SOTAs, extensive experiments are conducted on five synthetic datasets and four real datasets, which shows the improvement of the proposed CFNet.

## Quick Start
Download the dataset from <a href="https://drive.google.com/drive/folders/1-e2nPCr_eP1cTDhFFes27Rjj-QXzMk5u?usp=sharing" title="GoogleDrive">GoogleDrive</a>.

Extract the files to data folder folder as follow:

```
~/
data/
    SIDD_train/
      ... (scene id)
    Syn_train/
      ... (id)
    DND/
      images_srgb/
        ... (mat files)
      ... (mat files)
```
Train the model:
```python train.py```

Test the model:
```python test.py```

## Network Structure
<img width="1058" alt="截屏2023-12-30 11 58 44" src="https://github.com/WenhaoYao/CFNet/assets/26796148/592caed6-2309-45cb-88f1-1921845db86b">

## Result
<img width="1081" alt="截屏2023-12-30 12 00 07" src="https://github.com/WenhaoYao/CFNet/assets/26796148/171379f0-7883-4550-a34c-628ae65e6487">
