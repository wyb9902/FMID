import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from data.data_load import test_dataloader
import time
import torch.nn.functional as f

import cv2
# ---------------------------------------------------

def _test(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()

    model.eval()
    factor = 4
    with torch.no_grad():


        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            tm = time.time()
            pred = model(input_img)[2]
            pred = pred[:,:,:h,:w]
            elapsed = time.time() - tm

            pred_clip = torch.clamp(pred, 0, 1)
            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

            print('time: %f' % (elapsed))

