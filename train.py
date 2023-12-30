import os, time, shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils.common import AverageMeter
from dataset.loader import Real, Syn
# from model.one_atb_compine import Network
# from model.New_TestNet6_3atb import Network
# from model.New_TestNet6_1atb import Network
from model.CFNet import Network
# from model.New_TestNet6_112stage import Network
# from model.New_TestNet6_1111stage import Network
# from model.New_TestNet6_114stage import Network

# from model.Restormer import Restormer
from model.loss import *
import time
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description = 'Train')
parser.add_argument('--bs', default=10, type=int, help='batch size')
parser.add_argument('--ps', default=128, type=int, help='patch size')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')#
parser.add_argument('--epochs', default =2000, type=int, help='sum of epochs')
# parser.add_argument('--psnr-lr', type=float, default=1e-3)
args = parser.parse_args()

""" GPU  """
cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    
def train(train_loader, model, criterion, optimizer,epoch):
	losses = AverageMeter()
	model.train()

	for (noise_img, clean_img, sigma_img, flag) in train_loader:
		input_var = noise_img.cuda()
		target_var = clean_img.cuda()
		sigma_var = sigma_img.cuda()
		flag_var = flag.cuda()

		noise_level_est, output = model(input_var)

		loss = criterion(output, target_var, noise_level_est, sigma_var, flag_var,epoch)

		losses.update(loss.item())


		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	return losses.avg

if __name__ == '__main__':
	save_dir = '/media/data/ywh/CFNet/Save_model/'

	model = Network().cuda(device)

	print("加载预处理模型")
	if os.path.exists(os.path.join(save_dir, 'epoch_1950.pth')):# epoch_{}.pth
		# load existing model
		model_info = torch.load(os.path.join(save_dir, 'epoch_1950.pth'))
		print('==> loading existing model:', os.path.join(save_dir, 'epoch_1950.pth'))# epoch_{}.pth
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		# for param_group in optimizer.param_groups:
		# 	param_group['lr'] = 7e-6
		optimizer.load_state_dict(model_info['optimizer'])
		# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1050, 1100, 1150], gamma=0.5)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
		# scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[1300,1500,1700,1900], gamma=0.5)
		# scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[1200,1500,1800], gamma=0.5)
		scheduler.load_state_dict(model_info['scheduler'])
		cur_epoch = model_info['epoch']

	else:
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		# create model
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
		# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1050, 1100, 1150], gamma=0.5)
		#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.65)
		cur_epoch = 0

	criterion = fixed_loss()
	criterion.cuda()
	print_network(model)
	print("-----------------start train-------------")
	train_dataset = Syn('data/train/synthetic_train_4744/', 4744, args.ps) + Real('data/train/SIDD_train/', 320, args.ps)
	# train_dataset = Real('data/train/SIDD_train/', 320, args.ps)
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
	# tensorboard
	# writer = SummaryWriter("logs")
	epoch = 0
	LOSS  = []
	for epoch in range(cur_epoch, args.epochs + 1):
		begin = time.perf_counter()
		loss= train(train_loader, model, criterion, optimizer,epoch=epoch)
		# 梯度裁剪
		# torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=20)
		scheduler.step()
		# writer.add_scalars("loss",loss,epoch)
		if (epoch) % 50 == 0:
			torch.save({'epoch': epoch + 1,
						'state_dict': model.state_dict(),
						'optimizer' : optimizer.state_dict(),
						'scheduler' : scheduler.state_dict()},
						# os.path.join(save_dir, 'checkpoint.pth.tar'))# 'epoch_{}.pth'.format(epoch)
						os.path.join(save_dir, 'epoch_{}.pth'.format(epoch)))
		end = time.perf_counter()

		print('114stage Epoch [{0}]\t'
		'lr:{lr:.5f}\t'
		'Loss: {loss:.6f}\t'
		'time{time:.2f}\t'
		'total epoch{total:.2f}'.format(epoch, lr=optimizer.param_groups[-1]['lr'], loss=loss, time=end - begin, total=args.epochs))
		# LOSS.append(loss)
		# print(loss.shape)
		# data_frame = pd.DataFrame(data={'epoch': epoch, 'LOSS': LOSS}, index=range(0, epoch + 1))
		# data_frame.to_csv(os.path.join('/DISK/xjc/PycharmProject/CBDNet', 'training_logs.csv'), index_label='index')
		# if epoch == 200:
		# 	for param_group in optimizer.param_groups:
		# 		param_group['lr'] /= 2.0
		if epoch == 100:
			for param_group in optimizer.param_groups:
				param_group['lr'] /= 2.0
		# if epoch == 1001:
		# 	for param_group in optimizer.param_groups:
		# 		param_group['lr'] = 1e-6

		# if epoch == 280:
		# 	for param_group in optimizer.param_groups:
		# 		param_group['lr'] /= 2.0
		# if epoch == 800:
		# 	for param_group in optimizer.param_groups:
		# 		param_group['lr'] /=2.0
		# if epoch == 850:
		# 	for param_group in optimizer.param_groups:
		# 		param_group['lr'] /=2.

