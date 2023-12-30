from torch.backends import cudnn

from real import denoise_srgb, bundle_submissions_srgb, SIDD_denoise
import torch
from model.New_TestNet6_1111stage import Network
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str, default='/media/data/ywh/CBDNet/Save_model/New_Test6/New_Test6_1111stage', help="Checkpoints directory,  (default:./checkpoints)")
parser.add_argument('--Isreal', default=True, help='Location to save checkpoint models')
parser.add_argument('--data_folder', type=str, default='/media/data/ywh/CBDNet/data', help='Location to save checkpoint models')
parser.add_argument('--out_folder', type=str, default='/media/data/ywh/CBDNet/data', help='Location to save checkpoint models')
parser.add_argument('--model', type=str, default='/media/data/ywh/CBDNet/Save_model/New_Test6/New_Test6_1111stage/epoch_2000.pth', help='Location to save checkpoint models')
parser.add_argument('--Type', type=str, default='SIDD', help='To choose the testing benchmark dataset, SIDD or Dnd')
args = parser.parse_args()
use_gpu = True
cudnn.benchmark = True
device = torch.device('cuda:0')
print('Loading the Model')
net = Network().cuda(device=device)
checkpoint = torch.load(args.model)
net.load_state_dict(checkpoint['state_dict'])
model = torch.nn.DataParallel(net).cuda(device=device)
model.eval()

if args.Type == 'Dnd':
    denoise_srgb(model, args.data_folder, args.out_folder)
    bundle_submissions_srgb(args.out_folder)
else:
    SIDD_denoise.test(args)