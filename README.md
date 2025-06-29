# FMID: Dual-Attention Feature Modulation Dehazing Network for Maritime Perception Enhancement
Yubo Wang, Zaijin You, and Xi Xia.

## Installation
The project is built with PyTorch 3.9.2, PyTorch 2.5.0, CUDA 11.8, cuDNN 9.1.0
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

python main.py --mode train --data_dir your_path/RESIDE

## Test

python main.py --mode test --data_dir your_path/RESIDE --test_model path_to_weight

## Eval

You can use psnrssim.py to calculate PSNR and SSIM.

## Trained weights
To test on RESIDE-SOTS outdoor, we provide the Trained weights in the checkpoint. 

You can download the train and test dataset from https://drive.google.com/drive/folders/1GFycRaUHnvt8BAkQ5QjUw-Xrjovu0b2U?usp=drive_link

Other data be provided after the paper is accpeted.

For training and testing on your own dataset, your directory structure should look like this

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
Our code is developed based on ChaIR. We thank the awesome work provided by ChaIR https://github.com/c-yn/ChaIR.
And great thanks to editors and anonymous reviewers for future useful feedback.

## Contact
Should you have any question, please contact Yubo Wang (wang_yb@dlmu.edu.cn).
