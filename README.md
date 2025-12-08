# Dual-Attention Feature Modulation for Maritime Vision-Based Measurement Data Recovery
Yubo Wang, Zaijin You, Xi Xia, and Jingchun Zhou.
## Abstract
Maritime visual instruments, such as navigation cameras and monitoring sensors, are pivotal for providing accurate visual data in maritime vision-based measurement systems (MVMS). However, in frequent hazy conditions, these instruments capture degraded images where the fidelity of the visual data is severely compromised, resulting substantial errors in downstream measurement tasks like object detection and scene segmentation. To mitigate this measurement problem, we propose a feature modulation image dehazing network named FMID, designed as a pre-processing module to restore image data quality for maritime visual instruments. The core of our network is the dual attention residual block (DARB) that comprises a feature enhancement channel attention (FECA) module and a fusion complementary spatial attention (FCSA) module. FECA employs multi-scale convolutions to learn dynamic channel weights, effectively aggregating contextual information. FCSA integrates multi-scale features to obtain spatial attention weights and extracts complementary information along horizontal and vertical directions to guide the aggregation, thereby accurately reconstructing structural details crucial for measurement. Extensive experiments on standard and maritime-related datasets demonstrate that FMID outperforms several state-of-the-art methods in terms of image quality metrics. More importantly, application results confirm that the dehazed images produced by FMID significantly enhance the performance and reliability of critical maritime vision-based measurement tasks, including object detection and scene segmentation.

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
