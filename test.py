import PIL.Image
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2
from torch.backends import cudnn
import glob
from model.CFNet import Network
cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from utils.common import *

parser = argparse.ArgumentParser(description = 'Test')
# parser.add_argument('input_filename', type=str)
# parser.add_argument('output_filename', type=str)
args = parser.parse_args()

save_dir = 'Save_model/save_model_500/'
cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Network().cuda(device=device)
# model.cuda()
# model = nn.DataParallel(model)

model.eval()
test_fns = glob.glob(r'/home/xjc/PycharmProjects/CBDNet/data/test/RNI15/Dog.png')
for i in range(len(test_fns)):
    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        model.load_state_dict(model_info['state_dict'])
    else:
        print('Error: no trained model detected!')
        exit(1)
    img = cv2.imread(test_fns[i])
    img = img[:, :, ::-1] / 255.0
    img = np.array(img).astype('float32')
    input_var =  torch.from_numpy(hwc_to_chw(img)).unsqueeze(0).cuda()

    with torch.no_grad():
        _, output = model(input_var)

    output_image = chw_to_hwc(output[0,...].cpu().numpy())
    output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))
    output_image = PIL.Image.fromarray(output_image)
    output_image.save("/home/xjc/PycharmProjects/CBDNet/data/test/dog"+'/'+str(i)+'.png')



