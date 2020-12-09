import os
import sys
import tqdm 
import math
import time
import torch
import datetime
import itertools
import torchvision

import numpy as np
import torch.nn as nn
import skimage.io as io
import SimpleITK as sitk
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import autograd


def list_volumes(mri_list, cryo_list, DATA_DIR, MRI_DIR, CRYO_DIR):
    new_mri_list, new_cryo_list = [], []
    for i in range(len(cryo_list)):
        np_cryo = sitk.GetArrayFromImage(sitk.ReadImage(DATA_DIR+MRI_DIR+cryo_list[i]))
        non_zero = np.count_nonzero(np_cryo) / (32*32*32)
        if non_zero >= 0.9:
            new_mri_list.append(mri_list[i])
            new_cryo_list.append(cryo_list[i])

    return new_mri_list, new_cryo_list

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_3d_gaussian(window_size, sigma=5):
    gauss_1 = gaussian(window_size, sigma)
    
    gaussian_kernel = torch.zeros((window_size, window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            for k in range(window_size):
                gaussian_kernel[i,j,k] = gauss_1[i] * gauss_1[j] * gauss_1[k]
    
    gaussian_kernel = gaussian_kernel.float()
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)
    
    return gaussian_kernel

class SSIM(torch.nn.Module):
    def __init__(self, device, size_average=True, window_size=11, sigma=1.5):
        super(SSIM, self).__init__()
        self.kernel = create_3d_gaussian(window_size, sigma).to(device)
        self.size_average = size_average
        
    def forward(self, img1, img2):
        mu1 = F.conv3d(img1, self.kernel, padding = self.kernel.shape[2]//2, groups = 1)
        mu2 = F.conv3d(img2, self.kernel, padding = self.kernel.shape[2]//2, groups = 1)

        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2

        sigma1_sq = F.conv3d(img1**2, self.kernel, padding = self.kernel.shape[2]//2, groups = 1) - mu1_sq
        sigma2_sq = F.conv3d(img2**2, self.kernel, padding = self.kernel.shape[2]//2, groups = 1) - mu2_sq    
        sigma12 = F.conv3d(img1*img2, self.kernel, padding = self.kernel.shape[2]//2, groups = 1) - mu1_mu2

        L = 1
        C1, C2 = (0.01*L)**2, (0.03*L)**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return torch.mean(ssim_map) + 1 # ssim varies from -1 to 1
        else:
            ssim_map

def calc_gradient_penalty(netD, real_data, fake_data, device, channels=1, LAMBDA=5.0):
    #print real_data.size()
    alpha = torch.rand(real_data.shape[0], channels, 1, 1, 1)
    # print(alpha.shape)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    # print(interpolates.shape)
    
    interpolates = interpolates.to(device)
    # interpolates.required_grad = True
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetEncoder, self).__init__()

        # INPUT = (h,w) OUTPUT = (h/2,w/2)
        model2 = [nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), stride=1, padding=True)]
        model2 += [nn.ReLU(inplace=True)]
        model2 += [nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding=True)]
        model2 += [nn.ReLU(inplace=True)]
        model2 += [nn.BatchNorm3d(64)]
        model2down = [nn.MaxPool3d(kernel_size=(2,2,2), stride=2)]


        # INPUT = (h/2,w/2) OUTPUT = (h/4,w/4)
        model3 = [nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=True)]
        model3 += [nn.ReLU(inplace=True)]
        model3 += [nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=1, padding=True)]
        model3 += [nn.ReLU(inplace=True)]
        model3 += [nn.BatchNorm3d(128)]
        model3down = [nn.MaxPool3d(kernel_size=(2,2,2), stride=2)]


        # INPUT = (h/4,w/4) OUTPUT = (h/8,w/8)
        model4 = [nn.Conv3d(128, 128, kernel_size=(3,3,3), stride=1, padding=True)]
        model4 += [nn.ReLU(inplace=True)]
        model4 += [nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=1, padding=True)]
        model4 += [nn.ReLU(inplace=True)]
        model4 += [nn.BatchNorm3d(256)]
        model4down = [nn.MaxPool3d(kernel_size=(2,2,2), stride=2)]


        # INPUT = (h/8,w/8) OUTPUT = (h/16,w/16)
        model5 = [nn.Conv3d(256, 256, kernel_size=(3,3,3), stride=1, padding=True)]
        model5 += [nn.ReLU(inplace=True)]
        model5 += [nn.Conv3d(256, 512, kernel_size=(3,3,3), stride=1, padding=True)]
        model5 += [nn.ReLU(inplace=True)]
        model5 += [nn.Conv3d(512, 512, kernel_size=(3,3,3), stride=1, padding=True)]
        model5 += [nn.ReLU(inplace=True)]
        model5 += [nn.BatchNorm3d(512)]
        model5down = [nn.MaxPool3d(kernel_size=(2,2,2), stride=2)]

        
        model5up = [nn.Upsample(scale_factor=2, mode='trilinear')]


        model11 = [nn.Conv3d(768, 256, kernel_size=(3,3,3), stride=1, padding=True)]
        model11 += [nn.ReLU(inplace=True)]
        model11 += [nn.Conv3d(256, 256, kernel_size=(3,3,3), stride=1, padding=True)]
        model11 += [nn.ReLU(inplace=True)]
        model11 += [nn.BatchNorm3d(256)]

        model11up = [nn.Upsample(scale_factor=2, mode='trilinear')]

        model6 = [nn.Conv3d(384, 128, kernel_size=(3,3,3), stride=1, padding=True)]
        model6 += [nn.ReLU(inplace=True)]
        model6 += [nn.Conv3d(128, 128, kernel_size=(3,3,3), stride=1, padding=True)]
        model6 += [nn.ReLU(inplace=True)]
        model6 += [nn.BatchNorm3d(128)]

        model6up = [nn.Upsample(scale_factor=2, mode='trilinear')]

        model7 = [nn.Conv3d(192, 64, kernel_size=(3,3,3), stride=1, padding=True)]
        model7 += [nn.ReLU(inplace=True)]
        model7 += [nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=True)]
        model7 += [nn.ReLU(inplace=True)]
        model7 += [nn.BatchNorm3d(64)]

        model7up = [nn.Upsample(scale_factor=2, mode='trilinear')]

        model9 = [nn.Conv3d(64, out_channels, kernel_size=(1,1,1), stride=1, padding=False), nn.Sigmoid()]

        self.model2down = nn.Sequential(*model2down)
        self.model3down = nn.Sequential(*model3down)
        self.model4down = nn.Sequential(*model4down)
        self.model5down = nn.Sequential(*model5down)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model11 = nn.Sequential(*model11)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model9 = nn.Sequential(*model9)

        self.model5up = nn.Sequential(*model5up)
        self.model11up = nn.Sequential(*model11up)
        self.model6up = nn.Sequential(*model6up)
        self.model7up = nn.Sequential(*model7up)

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        out_2 = self.model2(x)
        out_2_down = self.model2down(out_2) # h/2,w/2,64

        out_3 = self.model3(out_2_down)
        out_3_down = self.model3down(out_3) # h/4,w/4,128

        out_4 = self.model4(out_3_down)
        out_4_down = self.model4down(out_4) # h/8,w/8,256

        out_5 = self.model5(out_4_down)
        out_5_down = self.model5down(out_5) # h/16,w/16,512

        out_5_up = self.model5up(out_5_down)
        out_11 = self.model11(torch.cat((out_5_up, out_4_down), dim=1)) # h/8,w/8,256
        
        out_11_up = self.model11up(out_11)
        out_6 = self.model6(torch.cat((out_11_up, out_3_down), dim=1)) #h/4,w/4,128

        out_6_up = self.model6up(out_6) 
        out_7 = self.model7(torch.cat((out_6_up, out_2_down), dim=1)) #h/2,w/2,64

        out_7_up = self.model7up(out_7)
        out_rgb = self.model9(out_7_up)
        return out_rgb


# In[4]:

class Discriminator2(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator2, self).__init__()
        model = [   nn.Conv3d(in_channels, 128, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]
        model += [  nn.Conv3d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm3d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        model += [  nn.Conv3d(256, 512, 4, stride=2, padding=1),
                    nn.InstanceNorm3d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        model += [  nn.Conv3d(512, 512, 4, padding=1),
                    nn.InstanceNorm3d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        model += [nn.Conv3d(512, 1, 4, padding=1),]
        model += [nn.Sigmoid()]
        self.model = nn.Sequential(*model)
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, leaky_relu=False):
        super(Discriminator, self).__init__()
        model = [   nn.Conv3d(in_channels, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]
        model += [  nn.Conv3d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm3d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        model += [  nn.Conv3d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm3d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        model += [  nn.Conv3d(256, 512, 4, padding=1),
                    nn.InstanceNorm3d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        model += [nn.Conv3d(512, 1, 4, padding=1),]
        if not leaky_relu:
            model += [nn.Sigmoid()]
        else:
            model += [nn.LeakyReLU()]
        self.model = nn.Sequential(*model)
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)


# In[5]:


class Sobel3d(nn.Module):
    def __init__(self, device):
        super(Sobel3d, self).__init__()
        self.filter_z = np.array([[[-1, 0, 1], [-2, 0, 2], [-1,0, 1]], [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])
        self.filter_x = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],[[-2, -4, -2], [0, 0, 0], [2, 4, 2]], [[-1, -2, -1], [0,0,0], [1,2,1]]])
        self.filter_y  = np.array([[[-1 , -2, -1], [-2, -4, -2], [-1, -2, -1]],[[0, 0, 0], [0, 0, 0],[0, 0, 0]], [[1, 2, 1], [2, 4, 2],[1, 2, 1]]])

        self.weights_x = torch.from_numpy(self.filter_x).unsqueeze(0).unsqueeze(0).float().to(device)
        self.weights_y = torch.from_numpy(self.filter_y).unsqueeze(0).unsqueeze(0).float().to(device)
        self.weights_z = torch.from_numpy(self.filter_z).unsqueeze(0).unsqueeze(0).float().to(device)

    def forward(self, out, target):

        g1_x = nn.functional.conv3d(out, self.weights_x, padding=1)
        g1_y = nn.functional.conv3d(out, self.weights_y, padding=1)
        g1_z = nn.functional.conv3d(out, self.weights_z, padding=1)

        g2_x = nn.functional.conv3d(target, self.weights_x, padding=1)
        g2_y = nn.functional.conv3d(target, self.weights_y, padding=1)
        g2_z = nn.functional.conv3d(target, self.weights_z, padding=1)

        g_1 = torch.abs(g1_x) + torch.abs(g1_y) + torch.abs(g1_z)
        g_2 = torch.abs(g2_x) + torch.abs(g2_y) + torch.abs(g2_z)

        return torch.mean(torch.abs(g_1 - g_2))

    
class Encoder(nn.Module):
    def __init__(self, input_channels=1):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv3d(input_channels, 64, 3, padding=1, stride=2)
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv5 = nn.Conv3d(256, 512, 3, padding=1, stride=2)  
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm3d(512)

        for m in self.modules():
            if isinstance(m,nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
        out = self.bn1(F.leaky_relu(self.conv1(x)))
        out = F.dropout2d(out, p=0.4, training=self.training)

        out = self.bn2(F.leaky_relu(self.conv2(out)))
        out = F.dropout2d(out, p=0.3, training=self.training)

        out = self.bn4(F.leaky_relu(self.conv4(out)))
        out = F.dropout2d(out, p=0.3, training=self.training)

        out = self.bn5(F.leaky_relu(self.conv5(out)))
        out = F.dropout2d(out, p=0.4, training=self.training)
        return out


class ColorDecoderConvTrans(nn.Module):
    def __init__(self, out_channels=1):
        super(ColorDecoderConvTrans, self).__init__()

        self.conv1 = nn.ConvTranspose3d(512, 256, 3, padding=1, stride=2, output_padding=1)
        self.conv2 = nn.ConvTranspose3d(256, 128, 3, padding=1, stride=2, output_padding=1)
        self.conv4 = nn.ConvTranspose3d(128, 128, 3, padding=1, stride=2, output_padding=1)
        self.conv7 = nn.Conv3d(128, 64, 3, padding=1)
        self.conv8 = nn.Conv3d(64, out_channels, 3, padding=1)

        self.bn1 = nn.BatchNorm3d(256)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(128)
        self.bn7 = nn.BatchNorm3d(64)

        for m in self.modules():
            if isinstance(m,nn.Conv3d) or isinstance(m, nn.Linear):
                print('Initializing', m)
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        out = self.bn1(F.leaky_relu(self.conv1(x)))
        out = F.dropout2d(out, p=0.3, training=self.training)
        out = self.bn2(F.leaky_relu(self.conv2(out)))
        out = F.dropout2d(out, p=0.3, training=self.training)
        out = self.bn4(F.leaky_relu(self.conv4(out)))
        out = F.dropout2d(out, p=0.3, training=self.training)
        out = self.bn7(F.leaky_relu(self.conv7(out)))
        out = F.sigmoid(self.conv8(out))

        return out



class AESkip2(nn.Module):

    def __init__(self, input_channels=1, out_channels=1, CHANNEL_SIZE=64):
        super(AESkip2, self).__init__()

        self.conv1 = nn.Conv3d(input_channels, CHANNEL_SIZE, 3, padding=1, stride=1)
        self.conv2 = nn.Conv3d(CHANNEL_SIZE, CHANNEL_SIZE*2, 3, padding=1, stride=2)
        self.conv4 = nn.Conv3d(CHANNEL_SIZE*2, CHANNEL_SIZE*2*2, 3, padding=1, stride=2)
        self.conv5 = nn.Conv3d(CHANNEL_SIZE*2*2, CHANNEL_SIZE*2*2*2, 3, padding=1, stride=2)  
        self.conv6 = nn.Conv3d(CHANNEL_SIZE*2*2*2, CHANNEL_SIZE*2*2*2*2, 3, padding=1, stride=1)  
        self.bn1 = nn.BatchNorm3d(CHANNEL_SIZE)
        self.bn2 = nn.BatchNorm3d(CHANNEL_SIZE*2)
        self.bn3 = nn.BatchNorm3d(CHANNEL_SIZE*2*2)
        self.bn4 = nn.BatchNorm3d(CHANNEL_SIZE*2*2*2)
        self.bn5 = nn.BatchNorm3d(CHANNEL_SIZE*2*2*2*2)



        #### decode
#         self.convt0 = nn.Conv3d(CHANNEL_SIZE*2*2*2*2, CHANNEL_SIZE*2*2*2, 3, padding=1, stride=1)
        self.convt1 = nn.ConvTranspose3d(CHANNEL_SIZE*2*2*2*2, CHANNEL_SIZE*2*2, 3, padding=1, stride=2, output_padding=1)
        self.convt2 = nn.ConvTranspose3d(CHANNEL_SIZE*2*2*2, CHANNEL_SIZE*2, 3, padding=1, stride=2, output_padding=1)
        self.convt3 = nn.ConvTranspose3d(CHANNEL_SIZE*2*2, CHANNEL_SIZE*2, 3, padding=1, stride=2, output_padding=1)
        self.convt4 = nn.Conv3d(CHANNEL_SIZE*2+CHANNEL_SIZE//2*2, CHANNEL_SIZE, 3, padding=1)
        self.convt5 = nn.Conv3d(CHANNEL_SIZE, out_channels, 3, padding=1)

        
        self.bnt0 = nn.BatchNorm3d(CHANNEL_SIZE*2*2*2)
        self.bnt1 = nn.BatchNorm3d(CHANNEL_SIZE*2*2)
        self.bnt2 = nn.BatchNorm3d(CHANNEL_SIZE*2)
        self.bnt3 = nn.BatchNorm3d(CHANNEL_SIZE*2)
        self.bnt4 = nn.BatchNorm3d(CHANNEL_SIZE)

        for m in self.modules():
            if isinstance(m,nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x, half_size, quarter_size):
        out1 = self.bn1(F.leaky_relu(self.conv1(x)))
        out1 = F.dropout3d(out1, p=0.4, training=self.training)

        # print('conv1 shape', out.shape)

        out2 = self.bn2(F.leaky_relu(self.conv2(out1)))
        out2 = F.dropout3d(out2, p=0.3, training=self.training)

        # print('conv2 shape', out.shape)

        out3 = self.bn3(F.leaky_relu(self.conv4(out2)))
        out3 = F.dropout3d(out3, p=0.3, training=self.training)

        # print('conv3 shape', out.shape)

        out4 = self.bn4(F.leaky_relu(self.conv5(out3)))
        out4 = F.dropout3d(out4, p=0.4, training=self.training)

        # print('conv4 shape', out.shape)

        out5 = self.bn5(F.leaky_relu(self.conv6(out4)))
        out5 = F.dropout3d(out5, p=0.4, training=self.training)

        
#         print('out1:' , out1.shape, 'out2: ', out2.shape, 'out3: ', out3.shape, 'out4: ', out4.shape, 'out5: ', out5.shape)
        
#         out = self.bnt0(F.leaky_relu(self.convt0(out)))
#         out = F.dropout3d(out, p=0.3, training=self.training)
        
#         print(out.shape)
        
        out_1 = self.bnt1(F.leaky_relu(self.convt1(out5)))
        out_1 = F.dropout3d(out_1, p=0.3, training=self.training)
        
        ############ 3: T1 : 8, 2: T2 : 16 , 1 : T3 : 32
        
        
        out_1 = torch.cat([out3, out_1], dim=1)
        

#         out = torch.cat((out,quarter_size), 1)
        # print('convt1 shape', out.shape)

        out_2 = self.bnt2(F.leaky_relu(self.convt2(out_1))) 
        out_2 = F.dropout3d(out_2, p=0.3, training=self.training)
        
#         print('out T 1:', out_1.shape, 'out T 2:', out_2.shape)
        
        
        out_2 = torch.cat([out_2, out2], dim=1)

#         out = torch.cat((out, half_size), 1)
#         # print('convt2 shape', out.shape)

        out_3 = self.bnt3(F.leaky_relu(self.convt3(out_2)))
        out_3 = F.dropout3d(out_3, p=0.3, training=self.training)
#         # print('convt3 shape', out.shape)


        out_3 = torch.cat([out_3, out1], dim=1) 

        out_4 = self.bnt4(F.leaky_relu(self.convt4(out_3)))
        # print('convt4 shape', out8.shape)

        out = F.sigmoid(self.convt5(out_4))

        
#         print('out T 3: ', out_3.shape, 'out T 4:', out_4.shape)
#         val = input()
        return out




class AutoEncoder(nn.Module):
    def __init__(self, out_channels=1, train=True):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = ColorDecoderConvTrans(out_channels=out_channels)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

class AESkip(nn.Module):

    def __init__(self, input_channels=1, out_channels=1, CHANNEL_SIZE=64):
        super(AESkip, self).__init__()

        self.conv1 = nn.Conv3d(input_channels, CHANNEL_SIZE, 3, padding=1, stride=1)
        self.conv2 = nn.Conv3d(CHANNEL_SIZE, CHANNEL_SIZE*2, 3, padding=1, stride=2)
        self.conv4 = nn.Conv3d(CHANNEL_SIZE*2, CHANNEL_SIZE*2*2, 3, padding=1, stride=2)
        self.conv5 = nn.Conv3d(CHANNEL_SIZE*2*2, CHANNEL_SIZE*2*2*2, 3, padding=1, stride=2)  
        #self.conv6 = nn.Conv3d(CHANNEL_SIZE*2*2*2, CHANNEL_SIZE*2*2*2*2, 3, padding=1, stride=1)  
        self.bn1 = nn.BatchNorm3d(CHANNEL_SIZE)
        self.bn2 = nn.BatchNorm3d(CHANNEL_SIZE*2)
        self.bn3 = nn.BatchNorm3d(CHANNEL_SIZE*2*2)
        self.bn4 = nn.BatchNorm3d(CHANNEL_SIZE*2*2*2)
        #self.bn5 = nn.BatchNorm3d(CHANNEL_SIZE*2*2*2*2)



        #### decode
#         self.convt0 = nn.Conv3d(CHANNEL_SIZE*2*2*2*2, CHANNEL_SIZE*2*2*2, 3, padding=1, stride=1)
        self.convt1 = nn.ConvTranspose3d(CHANNEL_SIZE*2*2*2, CHANNEL_SIZE*2*2, 3, padding=1, stride=2, output_padding=1)
        self.convt2 = nn.ConvTranspose3d(CHANNEL_SIZE*2*2+1, CHANNEL_SIZE*2, 3, padding=1, stride=2, output_padding=1)
        self.convt3 = nn.ConvTranspose3d(CHANNEL_SIZE*2+1, CHANNEL_SIZE*2, 3, padding=1, stride=2, output_padding=1)
        self.convt4 = nn.Conv3d(CHANNEL_SIZE*2, CHANNEL_SIZE, 3, padding=1)
        self.convt5 = nn.Conv3d(CHANNEL_SIZE, out_channels, 3, padding=1)

        
        #self.bnt0 = nn.BatchNorm3d(CHANNEL_SIZE*2*2*2)
        self.bnt1 = nn.BatchNorm3d(CHANNEL_SIZE*2*2)
        self.bnt2 = nn.BatchNorm3d(CHANNEL_SIZE*2)
        self.bnt3 = nn.BatchNorm3d(CHANNEL_SIZE*2)
        self.bnt4 = nn.BatchNorm3d(CHANNEL_SIZE)

        for m in self.modules():
            if isinstance(m,nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x, half_size, quarter_size):
        out1 = self.bn1(F.leaky_relu(self.conv1(x)))
        out1 = F.dropout3d(out1, p=0.4, training=self.training)

        # print('conv1 shape', out.shape)

        out2 = self.bn2(F.leaky_relu(self.conv2(out1)))
        out2 = F.dropout3d(out2, p=0.3, training=self.training)

        # print('conv2 shape', out.shape)

        out3 = self.bn3(F.leaky_relu(self.conv4(out2)))
        out3 = F.dropout3d(out3, p=0.3, training=self.training)

        # print('conv3 shape', out.shape)

        out4 = self.bn4(F.leaky_relu(self.conv5(out3)))
        out4 = F.dropout3d(out4, p=0.4, training=self.training)

        # print('conv4 shape', out.shape)

#         out5 = self.bn5(F.leaky_relu(self.conv6(out4)))
#         out5 = F.dropout3d(out5, p=0.4, training=self.training)

        
#         print('out1:' , out1.shape, 'out2: ', out2.shape, 'out3: ', out3.shape, 'out4: ', out4.shape, 'out5: ', out5.shape)
        
#         out = self.bnt0(F.leaky_relu(self.convt0(out)))
#         out = F.dropout3d(out, p=0.3, training=self.training)
        
#         print(out.shape)
        
        out_1 = self.bnt1(F.leaky_relu(self.convt1(out4)))
        out_1 = F.dropout3d(out_1, p=0.3, training=self.training)
        
        ############ 3: T1 : 8, 2: T2 : 16 , 1 : T3 : 32
        
        
        #out_1 = torch.cat([out3, out_1], dim=1)
        

        out_1 = torch.cat((out_1,quarter_size), 1)
        # print('convt1 shape', out.shape)

        out_2 = self.bnt2(F.leaky_relu(self.convt2(out_1))) 
        out_2 = F.dropout3d(out_2, p=0.3, training=self.training)
        
#         print('out T 1:', out_1.shape, 'out T 2:', out_2.shape)
        
        
        #out_2 = torch.cat([out_2, out2], dim=1)

        out_2 = torch.cat((out_2, half_size), 1)
#         # print('convt2 shape', out.shape)

        out_3 = self.bnt3(F.leaky_relu(self.convt3(out_2)))
        out_3 = F.dropout3d(out_3, p=0.3, training=self.training)
#         # print('convt3 shape', out.shape)


        #out_3 = torch.cat([out_3, out1], dim=1) 

        out_4 = self.bnt4(F.leaky_relu(self.convt4(out_3)))
        # print('convt4 shape', out8.shape)

        out = F.sigmoid(self.convt5(out_4))

        
#         print('out T 3: ', out_3.shape, 'out T 4:', out_4.shape)
#         val = input()
        return out




class Discriminator_old(nn.Module):
    def __init__(self, dim, in_channels=1, CHANNEL_SIZE=256, sig=False):
        super(Discriminator_old, self).__init__()
        self.sig = sig
        self.conv1 = nn.Conv3d(in_channels, CHANNEL_SIZE, 3, padding=1,stride=2)
        self.conv2 = nn.Conv3d(CHANNEL_SIZE, CHANNEL_SIZE, 3, padding=1)
        self.conv3 = nn.Conv3d(CHANNEL_SIZE, CHANNEL_SIZE//2, 3, padding=1, stride=2)
        self.conv4 = nn.Conv3d(CHANNEL_SIZE//2, CHANNEL_SIZE//4, 3, padding=1, stride=4)
        
        self.linear1 = nn.Linear(CHANNEL_SIZE//4 * int(dim/16) * int(dim/16) * int(dim/16), 1)

        self.bn1 = nn.BatchNorm3d(CHANNEL_SIZE)
        self.bn2 = nn.BatchNorm3d(CHANNEL_SIZE)
        self.bn3 = nn.BatchNorm3d(CHANNEL_SIZE//2)
        self.bn4 = nn.BatchNorm3d(CHANNEL_SIZE//4) 

        for m in self.modules():
            if isinstance(m,nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x))
        out = F.dropout2d(out, p=0.4, training=self.training)
        out = self.bn1(out)
        out = F.leaky_relu(self.conv2(out))
        out = F.dropout2d(out, p=0.4, training=self.training)
        out = self.bn2(out)
        out = F.leaky_relu(self.conv3(out))
        out = F.dropout2d(out, p=0.4, training=self.training)
        out = self.bn3(out)
        out = F.leaky_relu(self.conv4(out))
        out = F.dropout2d(out, p=0.4, training=self.training)
        out = self.bn4(out)
        out = out.view(x.shape[0], -1)
        out = F.dropout(out, p=0.4, training=self.training)
        if not self.sig:
            out = F.leaky_relu(self.linear1(out))
        else:
            out = F.sigmoid(self.linear1(out))
        return out

class FCN3D(nn.Module):
    def __init__(self, in_channels=1):
        
        super(FCN3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv3d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv3d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv3d(512, 256, 3, padding=1)
        self.conv7 = nn.Conv3d(256, 128, 3, padding=1)
        self.conv8 = nn.Conv3d(128, 64, 3, padding=1)
        self.conv9 = nn.Conv3d(64, 1, 3, padding=1)
    
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out))
        out = F.leaky_relu(self.conv5(out))
        
        out = F.leaky_relu(self.conv6(out))
        out = F.leaky_relu(self.conv7(out))
        out = F.leaky_relu(self.conv8(out))
        out = F.sigmoid(self.conv9(out))
        
        return out

class FCNBN3D(nn.Module):
    def __init__(self, in_channels=1):
        
        super(FCNBN3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv3d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv3d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv3d(512+128, 256, 3, padding=1)
        self.conv7 = nn.Conv3d(256, 128, 3, padding=1)
        self.conv8 = nn.Conv3d(128, 64, 3, padding=1)
        self.conv9 = nn.Conv3d(64, 1, 3, padding=1)

        self.dp = nn.BatchNorm3d(64)
        self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm3d(512)
        self.bn5 = nn.BatchNorm3d(256)
        self.bn6 = nn.BatchNorm3d(128)
        self.bn7 = nn.BatchNorm3d(64)


    def forward(self, x):
        out = self.dp(F.leaky_relu(self.conv1(x)))
        out1 = self.bn1(F.leaky_relu(self.conv2(out)))
        out = self.bn2(F.leaky_relu(self.conv3(out1)))
        out = self.bn3(F.leaky_relu(self.conv4(out)))
        out = self.bn4(F.leaky_relu(self.conv5(out)))
        
        
        out = self.bn5(F.leaky_relu(self.conv6(torch.cat([out, out1], dim=1))))
        
        out = self.bn6(F.leaky_relu(self.conv7(out)))
        out = self.bn7(F.leaky_relu(self.conv8(out)))
        out = F.sigmoid(self.conv9(out))
        
        return out
class FCNBN3DNSKIP(nn.Module):
    def __init__(self, in_channels=1):
        
        super(FCNBN3DNSKIP, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv3d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv3d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv3d(512, 256, 3, padding=1)
        self.conv7 = nn.Conv3d(256, 128, 3, padding=1)
        self.conv8 = nn.Conv3d(128, 64, 3, padding=1)
        self.conv9 = nn.Conv3d(64, 1, 3, padding=1)

        self.dp = nn.BatchNorm3d(64)
        self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm3d(512)
        self.bn5 = nn.BatchNorm3d(256)
        self.bn6 = nn.BatchNorm3d(128)
        self.bn7 = nn.BatchNorm3d(64)


    def forward(self, x):
        out = self.dp(F.leaky_relu(self.conv1(x)))
        out = self.bn1(F.leaky_relu(self.conv2(out)))
        out = self.bn2(F.leaky_relu(self.conv3(out)))
        out = self.bn3(F.leaky_relu(self.conv4(out)))
        out = self.bn4(F.leaky_relu(self.conv5(out)))
        
        
        out = self.bn5(F.leaky_relu(self.conv6(out)))
        
        out = self.bn6(F.leaky_relu(self.conv7(out)))
        out = self.bn7(F.leaky_relu(self.conv8(out)))
        out = F.sigmoid(self.conv9(out))
        
        return out

class FCN3DNew(nn.Module):
    def __init__(self, device, in_channels=1):
        
        super(FCN3DNew, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv3 = nn.Conv3d(128, 128, 3, padding=1, stride=2)
        self.conv4 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv5 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        self.conv6 = nn.ConvTranspose3d(512, 256, 3, padding=1, stride=2, output_padding=1)
        self.conv7 = nn.ConvTranspose3d(256, 128, 3, padding=1, stride=2, output_padding=1)
        self.conv8 = nn.ConvTranspose3d(128, 64, 3, padding=1, stride=2, output_padding=1)
        self.conv9 = nn.ConvTranspose3d(64, 1, 3, padding=1, stride=2, output_padding=1)
        
        self.bn = [
            nn.BatchNorm3d(64).to(device),
            nn.BatchNorm3d(128).to(device),
            nn.BatchNorm3d(128).to(device),
            nn.BatchNorm3d(256).to(device),
            nn.BatchNorm3d(512).to(device),
            nn.BatchNorm3d(256).to(device),
            nn.BatchNorm3d(128).to(device),
            nn.BatchNorm3d(64).to(device),
        ]
        self.dropout = [
            nn.Dropout3d(p=0.3).to(device),
            nn.Dropout3d(p=0.4).to(device),
            nn.Dropout3d(p=0.4).to(device),
            nn.Dropout3d(p=0.4).to(device),
            
            nn.Dropout3d(p=0.3).to(device),
            nn.Dropout3d(p=0.3).to(device),
            nn.Dropout3d(p=0.3).to(device),
            nn.Dropout3d(p=0.3).to(device)
            
        ]
        for m in self.modules():
            if isinstance(m,nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                
    def forward(self, x):
        out = self.dropout[0](self.bn[0](F.leaky_relu(self.conv1(x))))
        out = self.dropout[1](self.bn[1](F.leaky_relu(self.conv2(out))))
        out = self.dropout[2](self.bn[2](F.leaky_relu(self.conv3(out))))
        out = self.dropout[3](self.bn[3](F.leaky_relu(self.conv4(out))))
        out = self.dropout[4](self.bn[4](F.leaky_relu(self.conv5(out))))
        
        out = self.dropout[5](self.bn[5](F.leaky_relu(self.conv6(out))))
        out = self.dropout[6](self.bn[6](F.leaky_relu(self.conv7(out))))
        out = self.dropout[7](self.bn[7](F.leaky_relu(self.conv8(out))))

        out = F.sigmoid(self.conv9(out))

        return out
def test_rand_model(model_name):
    if model_name == 'AEskip':
        model = AESkip2()
    print(model)
    device = torch.device("cuda")
    input = torch.randn(1,1,128,128,128)
    input_t = input.to(device)
    h_input = input_t[:, :, ::2, ::2, ::2]
    q_input = h_input[:,:, ::2 , ::2 , ::2] 
    model = model.to(device)

    out = model(input_t, h_input, q_input)

    print('Output shape', out.shape)


if __name__ == "__main__":
    test_rand_model('AEskip')
