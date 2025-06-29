import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


folder1 = "gt path"
folder2 = "result path"

files1 = os.listdir(folder1)
files2 = os.listdir(folder2)
psnr_values = []
ssim_values = []

for idx, (file1, file2) in enumerate(zip(files1, files2), start=1):
    img1 = cv2.imread(os.path.join(folder1, file1))
    img2 = cv2.imread(os.path.join(folder2, file2))

    #PSNR
    psnr_val = psnr(img1,img2, data_range=255)
    psnr_values.append(psnr_val)

    #SSIM
    ssim_val = ssim(img1, img2,data_range=255,channel_axis=-1,win_size=11)
    ssim_values.append(ssim_val)

    print(f"Processing pair {idx}: {file1} and {file2} psnr:{psnr_val} ssim:{ssim_val}")

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")