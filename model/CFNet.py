"""
不变的：
1》噪声估计分层，cat起来（原本就有的），
2》通道数，64，128，256，其中噪声估计层里面的阶段是64不变的。
改变：
在best model 上将 JTF共享变成不共享
k = 3
best model = 1000
ssid_noisy 7
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
    def forward(self, x1):
        x1 = self.up(x1)
        return x1

#---------新的激活函数FRelu------------------
# class FReLU(nn.Module):
#    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
#    """
#    def __init__(self, in_channels):
#        super().__init__()
#        self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
#        self.bn_frelu = nn.BatchNorm2d(in_channels)
#
#    def forward(self, x):
#        x1 = self.conv_frelu(x)
#        x1 = self.bn_frelu(x1)
#        x = torch.maximum(x, x1)
#        return x
#---------输出的 1*1 in=64-->out=3卷积 ，没有激活层--------------
class outconv(nn.Module):
    def __init__(self):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(64, 3, 1)
    def forward(self, x):
        x = self.conv(x)
        return x
# -----------浅层特征提取FE--------------------------
class FE(nn.Module):
    def __init__(self):
        super(FE, self).__init__()
        self.fe = nn.Sequential(
            nn.Conv2d(3, 64 ,3, padding=1), nn.PReLU(),
            nn.Conv2d(64, 64,3 ,padding=1), nn.PReLU(),
            nn.Conv2d(64, 64,3, padding=1), nn.PReLU()
        )
    def forward(self,x):
        x = self.fe(x)
        return x
# -----------深层特征提取Deep_FE--------------------------
class Deep_FE(nn.Module):
    def __init__(self,in_channles,out_channles):
        super(Deep_FE, self).__init__()
        self.fe = nn.Sequential(
            nn.Conv2d(in_channels=in_channles, out_channels=out_channles, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(in_channels=out_channles, out_channels=out_channles, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(in_channels=out_channles, out_channels=out_channles, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(in_channels=out_channles, out_channels=out_channles, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(in_channels=out_channles, out_channels=out_channles, kernel_size=1)
        )
    def forward(self,x):
        img = self.fe(x)
        x = x + img
        return x
#-----------噪声估计层FCN1-------------------------
class FCN(nn.Module):
    def __init__(self,in_channles,out_channles):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(in_channles, out_channles,3, padding=1),
            nn.PReLU(),
            nn.Conv2d(out_channles, out_channles, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(out_channles, out_channles, 3, padding=1),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.fcn(x)
        return x
# ----------仿射变换层SFT---------------------------
class SFTLayer(nn.Module):
    def __init__(self,in_channles,out_channles):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_channles, out_channles, 1)
        self.SFT_scale_conv1 = nn.Conv2d(out_channles, out_channles, 1)
        self.SFT_shift_conv0 = nn.Conv2d(out_channles, out_channles, 1)
        self.SFT_shift_conv1 = nn.Conv2d(out_channles, out_channles, 1)
        self.Sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.Sigmoid(self.SFT_scale_conv1(self.SFT_scale_conv0(x)))
        shift = self.SFT_shift_conv1((self.SFT_shift_conv0(x)))
        return x * (scale + 1) + shift

# ----------整合 噪声估计FCN + SFT --------------------
class FCN_SFT(nn.Module):
    def __init__(self):
        super(FCN_SFT, self).__init__()
        self.fcn1 = FCN(64,64)
        self.fcn2 = FCN(64,64)
        self.sft1 = SFTLayer(64,64)
        self.sft2 = SFTLayer(64,64)
        self.con =nn.Conv2d(128,64,1)
    def forward(self,x):
        x1 = self.sft1(self.fcn1(x))
        x2 = self.sft2(self.fcn2(x1))
        x  = torch.cat([x1,x2],dim=1)
        x  = self.con(x)
        return x
# ----------  JTF--------------------------------

# def default_conv(in_channels, out_channels, kernel_size, padding=1, bias=True):
#     return nn.Conv2d(
#         in_channels, out_channels, kernel_size,
#         padding=padding, bias=bias)
## Kernel_generation (KG)
class KG(nn.Module):
    # n_feat: number of feature maps
    # kernel_size: kernel size of joint trilateral filter
    def __init__(self, n_feat, kernel_size,group_num, act=nn.PReLU()):
        super(KG, self).__init__()
        self.conv_first = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, padding=1), act,
                                        nn.Conv2d(n_feat, n_feat, 3, padding=1), act,
                                        nn.Conv2d(n_feat, n_feat, 3, padding=1), act,
                                        nn.Conv2d(n_feat, n_feat, 3, padding=1), act,
                                        nn.Conv2d(n_feat, kernel_size * kernel_size * group_num, kernel_size=1)
                                        )


    def forward(self, x):
        y = self.conv_first(x)
        return y
## Joint Biliteral Filter (JBF)
class JTF(nn.Module):
    def __init__(self, n_feat, kernel_size,group_num):
        super(JTF, self).__init__()
        self.kernel_size = kernel_size
        self.g =group_num
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, 1)
        self.target_kg = KG( n_feat, kernel_size, group_num,act=nn.PReLU())
        self.guidance_kg = KG( n_feat, kernel_size,group_num, act=nn.PReLU())
        # self.concat_kg = KG(conv, n_feat, kernel_size, bias=True, act=nn.PReLU())
        self.jbf_conv = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, padding=1), nn.PReLU(),
                                      nn.Conv2d(n_feat, n_feat, 3, padding=1))

    def forward(self, image, guidance_noise):
        target = image
        # image_ten = self.bottleneck(image)
        target = self.target_kg(target)
        guidance = self.guidance_kg(guidance_noise)
        b,c, h,w = image.shape
        # Group = 8
        bi_kernel = (target * guidance).view(b, self.g, self.kernel_size * self.kernel_size,h,w).unsqueeze(2)
        patch = self.unfold(image).view(b, c , self.kernel_size * self.kernel_size, h, w)
        patch = patch.view(b,self.g,c//self.g,self.kernel_size * self.kernel_size,h,w)  # b,group,c//group,k*k,h,w
        # print(patch.shape,bi_kernel.shape)
        jbf_new = (patch * bi_kernel).view(b, self.kernel_size * self.kernel_size, c, h, w).sum(dim=1)
        jbf_new = self.jbf_conv(jbf_new)
        image = jbf_new + image
        return image
# -------------------------总体network-----------------------------------------------------------------------
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.endconv = outconv()
        self.FE1 = FE()
        self.conv1 = nn.Sequential(nn.Conv2d(3,64,1),nn.PReLU())
        # -------stage1----------------
        self.Deep_FE1 = Deep_FE(64,64)
        self.FCN_SFT1 = FCN_SFT() # 估计
        self.JTF1 = JTF(64,3,16) # nfeat,kernel
        self.JTF11 = JTF(64,3,16)
        self.pool1 = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(64,128,1)
        # -------stage2----------------
        self.Deep_FE2 = Deep_FE(128,128)
        self.JTF2 = JTF(128,3,16) # nfeat,kernel
        self.JTF21 = JTF(128,3,16)
        self.pool2 = nn.AvgPool2d(2,2)
        self.conv3 = nn.Conv2d(128,256,1)
        self.conv_noise2 = nn.Conv2d(128,64,1)
        # ------stage3-----------------
        self.Deep_FE3 = Deep_FE(256,256)
        self.JTF3 = JTF(256,3,16)
        self.JTF31 = JTF(256,3,16)
        self.conv_noise3 = nn.Conv2d(256,64,1)
        self.conv_noise4 = nn.Conv2d(64,256,1)
        # -------stage4与stage3没有变化
        self.JTF4 = JTF(256,3,16)
        self.JTF41 = JTF(256,3,16)
        # -------stage5----------------
        self.up1 = up(256)#256-->128
        self.JTF5 = JTF(128,3,16)
        self.JTF51 = JTF(128,3,16)
        # -------stage6---------------
        self.up2 = up(128)# 128-->64
        self.JTF6 =JTF(64,3,16)
        self.JTF61 = JTF(64,3,16)

    def forward(self,x):
        # --------stage1-----------------
        noise1 = self.FCN_SFT1(self.conv1(x)) # (64,64)

        x1 = self.Deep_FE1(self.FE1(x))       # (64,64)

        feature1 = self.JTF1(x1,noise1)       #

        feature1 = self.JTF11(feature1,noise1)
        # feature1 = feature1 + self.FE1(x)
        feature111= feature1
        # -------作为监督信息noise1---
        noise_out = self.endconv(noise1)
        feature1 = self.pool1(feature1)

        feature11 = self.conv2(feature1) # (64,128)

        # noise2   = self.pool1(noise1)
        noise2   = self.conv2(feature1)
        # -------stage2-------------------
        noise2   = self.conv_noise2(noise2)#(128,64)
        noise2   = self.FCN_SFT1(noise2)#(64,64)
        noise2   = self.conv2(noise2)# (64,128)
        feature2 = self.Deep_FE2(feature11)
        feature2 = self.JTF2(feature2,noise2)
        feature2 = self.JTF21(feature2,noise2) # 128,128
        # feature211 = feature11 + feature2
        feature211 = feature2
        feature2 = self.pool2(feature211)
        feature21= self.conv3(feature2)#128-->256
        # noise2 = self.pool2(noise2)# down -->h/2,w/2
        noise2 =self.conv3(feature2)# 128-->256
        #---------stage3------------------------
        noise3 = self.conv_noise3(noise2)# 256-->64
        noise3 = self.FCN_SFT1(noise3)
        noise3 = self.conv_noise4(noise3)#64-->256
        feature3 = self.Deep_FE3(feature21)
        feature3 = self.JTF3(feature3,noise3)
        feature3=self.JTF31(feature3,noise3)#256-->256
        # feature31 = feature3 +feature21
        feature31 = feature3
        #---stage4--------------------------
        noise4 = self.conv_noise3(feature3) # 256-->64
        noise4 = self.FCN_SFT1(noise4)
        noise4 = self.conv_noise4(noise4) #64-->256
        feature4 = self.Deep_FE3(feature31)#256-->256
        feature4 = self.JTF4(feature4,noise4)
        feature4 = self .JTF41(feature4,noise4)
        # feature41 =feature4 + feature31 # 256-->256
        feature41 = feature4
        # ------stage5-----------------------
        feature5 = self.up1(feature41)#256-->128
        feature5 = feature5 + feature211 # 3残差
        # noise5 = self.up1(feature4)  # 256-->128
        noise5 = self.conv_noise2(feature5)  # 128-->64
        noise5 = self.FCN_SFT1(noise5)
        noise5 = self.conv2(noise5)  # 64-->128
        feature5 = self.Deep_FE2(feature5)  # 256-->256
        feature5 = self.JTF5(feature5,noise5)
        feature5 = self.JTF51(feature5,noise5)
        feature5 = feature5 +feature211    # 128,h/2,w/2
        # ----stage6---------------------------
        feature6 = self.up2(feature5)# 128-->64,h,w
        feature6 = feature6 + feature111 # 2残差
        noise6 = self.FCN_SFT1(feature6)
        feature6 = self.Deep_FE1(feature6)#
        # noise6 = self.up2(noise5)  # 128-->64,h,w
        feature6 = self.JTF6(feature6,noise6)
        feature6 = self.JTF61(feature6,noise6)
        feature6 = feature6 +feature111 # 64,h,w

        # ---out-----
        img = self.endconv(feature6)
        img = x+img # 1残差
        return noise_out , img

