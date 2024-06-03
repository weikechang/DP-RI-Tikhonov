# -*- coding: utf-8 -*-
"""
@author: weike Chang
"""
import numpy as np
from math import exp
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from numpy.linalg import norm 
from scipy.io import loadmat,savemat
from skimage.metrics import structural_similarity
import copy
import math
def translation(img, blur_image, k):
    m,l,n = img.shape
    x = img[0:m-k,0:l-k,0:n-k]
    f = int(k/2)
    x_new = copy.copy(blur_image)
    x_new[f:m-f, f:l-f, f:n-f] = x
    return x_new
################################Deep denoiser prior#############################
import torch
device = torch.device("cuda")
from Unet import Unet
#################model#################
def prediect(data):
    state_dict = torch.load('../modle_save/model_tlstage.pth',map_location='cuda')
    model = Unet()  
    model.load_state_dict(state_dict)
    model.to(device)

    data = data.to(device)
    model.eval()
    with torch.no_grad():
        XH1_image = model(data)
    return XH1_image

def DDP_denoising(noisy, h1, w1, ll):
    noisy = noisy.reshape(h1, w1, ll)
    noisy[noisy < 0] = 0 
 
    n = 8
    k1 = int(noisy.shape[0] / n)
    train_data1 = noisy[0:k1*n]
    data1_size = train_data1.shape[0]
    data2_size = train_data1.shape[1]
    data3_size = train_data1.shape[2]
    train_data1 = train_data1.reshape(-1, 1, data1_size, data2_size, data3_size)
    train_data1 = torch.from_numpy(train_data1)
    net_output = prediect(train_data1)
    net_output = net_output.data.squeeze().float().cpu().numpy()
    net_output[net_output<0] = 0   
    denoise = np.concatenate([net_output, noisy[k1*n:noisy.shape[0]]],axis = 0)

    return denoise.flatten()

###############################################################################
def NRMSE(img, label):
    dif = label - img
    mse = norm(dif.flatten())**2
    la = norm(label.flatten())**2;
    return np.sqrt(mse / la)  
###############################################################################
blur_image = np.fromfile('../data/6_10_7_Brain.dat', dtype = 'float32')
blur_image = blur_image.reshape(90,160,160)
label = np.fromfile('../data/labelBrain.dat', dtype = 'float32')
label = label.reshape(90,160,160)
rint(NRMSE(noisy,label),structural_similarity(noisy[15:65,50:130,50:115],label[15:65,50:130,50:115]))
###############################################################################
A = loadmat('../psf/psf3.mat')
ker = A['psf'].astype('float32')
######################如果核的长度与图像不相等#####################
[m,n,l] = blur_image.shape
[m2,n2,l2] = ker.shape
ker = np.pad(ker,((0,m-m2),(0,n-n2),(0,l-l2)),'constant',constant_values = (0,0))
ker = ker.flatten()
######fft##########
ker_ft = fft(ker)
ker_ft_T = ker_ft.conjugate()
blur_image_ft = fft(blur_image.flatten())
##############正则化参数#######################
num = 200
reg_param = np.zeros((int(num)))
for j in range(num):
    reg_param[j] = 10 * exp(-15*(j/(num-1)))
reg_param = reg_param[::-1]
beta = reg_param[152] 
print(beta)
##############RI-Tikhonov#####################
alpha = 0.08
K = 50
C = 1
ker_Tydelta = ker_ft_T * blur_image_ft  #T'y
ker_ker = ker_ft_T * ker_ft # T*T
inv = ker_ker + beta # (T*T + BETA)
###################初始化x#####################
x_lamda = blur_image.copy()
f1 = int((m2 - 1) / 2)
k1 = int(m2 - 1)
x_lamda[0:m-k1,0:n-k1,0:l-k1] = blur_image[f1:m-f1, f1:n-f1, f1:l-f1]
x_lamda = x_lamda.flatten()
##################################################
x_0 = copy.copy(x_lamda)
all_alpha = np.zeros(K).astype('float32')
RR = np.linalg.norm(blur_image.flatten()) 
Evalu_nrmse = np.zeros(K,dtype=np.float32)
for k in range(K): 
    RIT_DDP = copy.copy(x_lamda)
    RIT_DDP[RIT_DDP < 0] = 0
    RIT_DDP = RIT_DDP.reshape(m,n,l)
    RIT_DDP = translation(RIT_DDP, blur_image, m2 -1)
    nrmse = NRMSE(RIT_DDP,label)
    Evalu_nrmse[k] = nrmse
    all_alpha[k] = alpha
    print('##########################################################################################')
    print(k + 1, alpha, np.max(x_lamda))
    print('NRMSE=%.*f'%(6,nrmse)) 
    RIT_DDP.tofile('../results/6_10_7DPRIT'+str(0)+str(k + 1)+'.dat')
#################################################################
    fir = alpha * (beta +  ker_ker) * fft(x_lamda)
    VK = DDP_denoising(x_lamda, m, n, l)
    sec = (1 - alpha) * (ker_Tydelta + beta * fft(VK))
    x_lamda = (fir + sec) / inv
    x_lamda = np.real(ifft(x_lamda))
#######choosing alpha################################## 
    
    tj1 = (ker_Tydelta + beta * fft(DDP_denoising(x_lamda, m, n, l) - x_lamda) - ker_ker * fft(x_lamda)) / inv
    tj1 = np.real(ifft(tj1))
    tj1 = norm(tj1)
    tj2 = (ker_Tydelta + beta * fft(VK - x_0) - ker_ker * fft(x_0)) / inv
    tj2 = np.real(ifft(tj2))
    tj2 = C * alpha * norm(tj2)
    if tj1 > tj2: 
      alpha = 1 - (1 - alpha) * (tj2/tj1)
    else:
      alpha = alpha
##############################################################################################
# =======================early stopping======================================================
    '''    
    if k >= 1:
        err = norm(x_lamda-x_0)/norm(RR)
        print(err)
        if err <= 0.001:
            break
    '''    
    err = norm(x_lamda-x_0)/norm(RR)
    print(err)
# ============================================================================= 
    x_0 = x_lamda
##############################################################################
##################################################################################
loc = np.where(Evalu_nrmse == np.min(Evalu_nrmse))
print(Evalu_nrmse[loc],loc)
x_lamda[x_lamda<0] = 0 
x_lamda = x_lamda.reshape(m,n,l)
x_lamda = translation(x_lamda, blur_image, m2 - 1)

plt.figure()
plt.plot(np.arange(K),Evalu_nrmse)
plt.figure()
plt.plot(np.arange(K),all_alpha)
plt.show()
'''
plt.figure()
plt.imshow(x_lamda[:,80,:], cmap = 'Greys', vmin = 0, vmax = 0.7)
plt.figure()
plt.imshow(blur_image[:,80,:], cmap = 'Greys', vmin = 0, vmax = 0.7)
plt.figure()
plt.imshow(label[:,80,:], cmap = 'Greys', vmin = 0, vmax = 0.7)
plt.show()
'''