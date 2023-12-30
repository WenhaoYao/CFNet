import torch.nn as nn
import torch
import torch.nn.functional as F
# import pytorch_ssim as ssim
from train import *
class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym,epoch):

        l2_loss = F.mse_loss(out_image, gt_image)
        l1_loss = F.l1_loss(out_image, gt_image)
        # smooth_loss = F.smooth_l1_loss(out_image, gt_image)
        # ssim_loss = 1- ssim.SSIM(out_image,gt_image)
        L_loss = l1_loss
        # if epoch <= 100:
        #     # a ,L_loss= 1,l2_loss
        #     a = 1
        # if epoch > 100 and epoch <= 300:
        #     a =  0.7
        # if epoch > 300 and epoch <= 500:
        #     a = 0.3
        # if epoch > 500 :
        #     a = 0
        if epoch <= 1500:
            a = 1
        if epoch > 1500 and epoch <= 1700:
             a =  0.7
        if epoch > 1700 and epoch <= 1900:
             a =  0.4
        if epoch > 1900:
            a = 0
        asym_loss = torch.mean(
            if_asym * torch.abs(0.25 - F.relu(gt_noise - est_noise)) * torch.pow(est_noise - gt_noise, 2))
            # 0.3--->0.25
        # h_x = est_noise.size()[2]
        # w_x = est_noise.size()[3]
        # count_h = self._tensor_size(est_noise[:, :, 1:, :])
        # count_w = self._tensor_size(est_noise[:, :, :, 1:])
        # h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x - 1, :]), 2).sum()
        # w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x - 1]), 2).sum()
        # tvloss = h_tv / count_h + w_tv / count_w

        # loss = l1_loss + 0.5 * asym_loss + 0.05 * tvloss
        loss = L_loss + 0.5 * a * asym_loss
        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
# class fixed_loss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, out_image, gt_image):

#         l2_loss = F.mse_loss(out_image, gt_image)
#         l1_loss = F.l1_loss(out_image, gt_image)
#         # smooth_loss = F.smooth_l1_loss(out_image, gt_image)
#         # ssim_loss = 1- ssim.SSIM(out_image,gt_image)
#         L_loss = l1_loss
#         # if epoch <= 100:
#         #     # a ,L_loss= 1,l2_loss
#         #     a = 1
#         # if epoch > 100 and epoch <= 300:
#         #     a =  0.7
#         # if epoch > 300 and epoch <= 500:
#         #     a = 0.3
#         # if epoch > 500 :
#         #     a = 0
#         # # if epoch <= 1500:
#         # #     a = 1
#         # # if epoch > 1500 and epoch <= 1700:
#         # #      a =  0.7
#         # # if epoch > 1700 and epoch <= 1900:
#         # #      a =  0.4
#         # # if epoch > 1900:
#         # #     a = 0
#         # asym_loss = torch.mean(
#         #     if_asym * torch.abs(0.25 - F.relu(gt_noise - est_noise)) * torch.pow(est_noise - gt_noise, 2))
#             # 0.3--->0.25
#         # h_x = est_noise.size()[2]
#         # w_x = est_noise.size()[3]
#         # count_h = self._tensor_size(est_noise[:, :, 1:, :])
#         # count_w = self._tensor_size(est_noise[:, :, :, 1:])
#         # h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x - 1, :]), 2).sum()
#         # w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x - 1]), 2).sum()
#         # tvloss = h_tv / count_h + w_tv / count_w

#         # loss = l1_loss + 0.5 * asym_loss + 0.05 * tvloss
#         loss = L_loss
#         return loss

#     def _tensor_size(self, t):
#         return t.size()[1] * t.size()[2] * t.size()[3]
