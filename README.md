# Feature Modulation Network for Haze Scenes Perception Enhancement in Maritime Intelligent Transportation Systems
Yubo Wang, Zaijin You, Xi Xia, and Jingchun Zhou.
## Abstract
Maritime visual instrument plays an important role in environment perception and navigation assistance. However, it always captures degraded images suffering from poor visibility, lost detail, and distorted color in haze scenes. To effectively enhance the visual perception, we propose a feature modulation image dehazing network named FMID, designed to restore image quality through optimized feature extraction. We use the dual attention residual block (DARB) to perform encoder-decoder-based feature extraction. The core part of DARB has two modules, feature enhancement channel attention (FEAC) and fusion complementary spatial attention (FCSA). Specifically, we use convolutions at different scales to obtain more discriminative feature maps, thus promoting channel attention to learn dynamic weights and focus on valuable signals. Furthermore, a feature fusion complementary attention is developed to effectively integrate spatial information. Features of different scales are fused to preliminarily obtain attention weights, and complementary features are extracted in both horizontal and vertical directions to dynamically guide the aggregation weights. Extensive experiments on standard and maritime-related datasets demonstrate that our FMID can outperform several state-of-the-art methods, enhance visual perception, and effectively improve the performance of maritime-related object detection and segmentation in haze scenes. 

## Installation
The project is built with Python 3.9.2, PyTorch 2.5.0, CUDA 11.8, cuDNN 9.1.0.
For installing, follow these instructions:
~~~
conda install pytorch=2.5.0 torchvision=0.20.0 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~
## Train
~~~
python main.py --mode train --data_dir your_path/RESIDE
~~~
## Test
~~~
python main.py --mode test --data_dir your_path/RESIDE --test_model your_path/RESIDE/checkpoint/reside.pkl
~~~
## Eval

You can use `psnrssim.py` to calculate PSNR and SSIM.

## Trained weights
To test on RESIDE-SOTS outdoor, we provide the trained weights in the `checkpoint`. 

You can download the train and test RESIDE dataset, and trained weights from [[gdrive](https://drive.google.com/drive/folders/1GFycRaUHnvt8BAkQ5QjUw-Xrjovu0b2U?usp=drive_link)].

Other data be provided after the paper is accpeted.

For training and testing on your own dataset, your directory structure should look like this:

`Your path` <br/>
`├──RESIDE` <br/>
     `├──train`  <br/>
          `├──gt`  <br/>
          `└──hazy`  
     `└──test`  <br/>
          `├──gt`  <br/>
          `└──hazy` 



## Citation
If you find this project useful for your research, please consider citing:
~~~
@article{ ,
  title={ },
  author={},
  journal={},
  volume={},
  pages={},
  year={},
  publisher={}
}
~~~

### Acknowledgments
Our code is developed based on ChaIR. We thank the awesome work provided by [[ChaIR](https://github.com/c-yn/ChaIR)].
And great thanks to editors and anonymous reviewers for future useful feedback.

## Contact
Should you have any question, please contact Yubo Wang (wang_yb@dlmu.edu.cn).
