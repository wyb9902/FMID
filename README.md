# Feature enhancement and fusion complementary modulation model for maritime image dehazing
Yubo Wang, Zaijin You, Xi Xia, and Jingchun Zhou.
## Abstract
Maritime optical imaging systems are critical for environment perception and the execution of key maritime op erations. However, optical imaging devices always capture degraded images in haze scenes, such as poor visibil ity, lost detail, and distorted color. To effectively enhance the imaging quality, we propose a feature modulation image dehazing model (FMID), designed to restore the reliability of visual information in haze scenes. The core of our model is the dual attention residual block (DARB) that comprises a feature enhancement channel attention (FECA) module and a fusion complementary spatial attention (FCSA) module. FECA employs multi-scale convo lutions to learn dynamic channel weights, effectively aggregating contextual information. FCSA integrates multi scale features to obtain spatial attention weights and extracts complementary information along horizontal and vertical directions to guide the aggregation, thereby accurately reconstructing structural detail. Extensive experi ments on standard and maritime-related datasets demonstrate that FMID can outperform several state-of-the-art methods, enhance imaging quality, and effectively improve the performance of downstream vision-based mar itime tasks, including vessel detection and maritime scene segmentation, providing robust technical guarantee for maritime traffic and intelligent port surveillance.

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
Wang, Y., You, Z., Xia, X., Zhou, J., & Li, Y. (2026). Feature enhancement and fusion complementary modulation model for maritime image dehazing. Optics & Laser Technology, 202, 115352.
~~~

### Acknowledgments
Our code is developed based on ChaIR. We thank the awesome work provided by [[ChaIR](https://github.com/c-yn/ChaIR)].
And great thanks to editors and anonymous reviewers for future useful feedback.

## Contact
Should you have any question, please contact Yubo Wang (wang_yb@dlmu.edu.cn).
