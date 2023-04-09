import torch
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import transforms
from torch import nn, optim
from matplotlib import pyplot as plt
from tqdm import tnrange
# from model_make import MLP
import os
import os
#用到的与QT通讯的代码

import socket
import threading
from sklearn.preprocessing import  MinMaxScaler
from sklearn.preprocessing import  MinMaxScaler
from MLP_Res import ResMLP
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#
print(device)
#count_filename = "count.txt" ##用于统计运行模型迭代了几次
from torch import nn
l = 20  # 厚度
d = 120  # 直径
fr = 18500  # 剩余磁场
u0 = torch.tensor(4 * math.pi * (10 ** -7))
pi = torch.tensor(math.pi)
m1 = (((math.pi) * d * d * l * fr) / (4 * u0))
model_dir="model_Mutual_RES_Y_TO_X.txt"
input_size_mag=5
input_size_mag=torch.tensor(input_size_mag)
output_size_mag=96
output_size_mag=torch.tensor(output_size_mag)
hidden_size_mag=512 #隐藏层的节点数量
hidden_size_mag=torch.tensor(hidden_size_mag)
num_layers_mag=5#隐藏层数量
num_layers_mag=torch.tensor(num_layers_mag)

input_size_position=96
input_size_position=torch.tensor(input_size_position)
output_size_position=5
output_size_position=torch.tensor(output_size_position)
hidden_size_position=1024 #隐藏层的节点数量
hidden_size_position=torch.tensor(hidden_size_position)
num_layers_position=5#隐藏层数量
num_layers_position=torch.tensor(num_layers_position)
#     # 读取 model.txt 文件的最后一行，获取上次运行的 pkl 文件名
with open('model_Mutual_RES_X_TO_Y.txt', 'r') as f:
     lines1 = f.readlines()
     if lines1:
         last_line_mag = lines1[-1].strip()
     else:
         last_line_mag= ""
#  # 读取 model.txt 文件的最后一行，获取上次运行的 pkl 文件名
# with open('model_new_split_angle_zhengzehua_4_5.txt', 'r') as f:
#      lines2 = f.readlines()
#      if lines2:
#          last_line_angle = lines2[-1].strip()
#      else:
#          last_line_angle = ""
#          # 读取 model.txt 文件的最后一行，获取上次运行的 pkl 文件名
with open('model_Mutual_RES_Y_TO_X.txt', 'r') as f:
     lines2 = f.readlines()
     if lines2:
         last_line_location = lines2[-1].strip()
     else:
         last_line_location = ""

single_file=1

T=0
model_5D =ResMLP(input_size_position, output_size_position, hidden_size_position, num_layers_position,device).to(device) #96->5
model_Mag =ResMLP(input_size_mag, output_size_mag, hidden_size_mag, num_layers_mag,device).to(device) #96->5
# modelfy = torch.load('finlc128.3.pkl', map_location=torch.device('cpu'))
# modelfx = torch.load('finlf128.3.pkl', map_location=torch.device('cpu'))
# model_5D.load_state_dict(torch.load(last_line_location ,map_location=device))
# model_Mag.load_state_dict(torch.load(last_line_mag,map_location=device))
model_5D.load_state_dict(torch.load('Model_RES_MUTUAL/model_pkl_RES_MUTUAL/MUTUAL_Y_X_0.0001_80_1000_1_0.1.pkl' ,map_location=device))
model_Mag.load_state_dict(torch.load('Model_RES_MUTUAL/model_pkl_RES_MUTUAL/MUTUAL_X_Y_0.0001_80_1000_1_0.1.pkl',map_location=device))
# # X1 = np.loadtxt('l0_4.669.txt',delimiter=',') #
# Y1 = np.loadtxt('d0_4.669.txt',delimiter=' ') #
# X1 = np.loadtxt('l0_-26.082.txt',delimiter=',') #
# Y1 = np.loadtxt('d0_-26.082.txt',delimiter=' ') #
# X1 = np.loadtxt('l0_0.txt',delimiter=',') #
# Y1 = np.loadtxt('d0_0.txt',delimiter=' ') #
# X1 = np.loadtxt('data_verify/cc45.txt',delimiter=',') #0_60
# Y1 = np.loadtxt('data_verify/ss45.txt',delimiter=',') #

    
if(single_file==1):
        Location_data= np.loadtxt('y_train_Y_TO_X_ONLY.txt') #0_60
        Mag_data = np.loadtxt('x_train_Y_TO_X_ONLY.txt') #
        # Location_data= np.loadtxt('data_verify_lowz/cc'+str(angle)+'_clean_no_outliers_delete.txt',delimiter=',') #0_60
        # Mag_data = np.loadtxt('data_verify_lowz/ss'+str(angle)+'_clean_no_outliers_delete.txt',delimiter=',') #
        # Location_data= np.loadtxt('data_verify/cc'+str(angle)+'.txt',delimiter=',') #0_60
        # Mag_data = np.loadtxt('data_verify/ss'+str(angle)+'.txt',delimiter=',') #
        Location_data=torch.tensor(np.array(Location_data,dtype=np.float32).reshape(-1,5)).to(device)
        Mag_data=torch.tensor(np.array(Mag_data,dtype=np.float32).reshape(-1,96)).to(device)
        mask = Location_data[:, 2] <= 37.5
        Location_data,Mag_data=Location_data[mask,:],Mag_data[mask,:]
        len1 = Location_data.shape[0]
        error = 0
        error1 = 0
        error2 = 0
        error3 = 0
        Mag_input = Mag_data
        output_5D = model_5D(Mag_input,device)
        LOCATION_OUT=Location_data
        LOCATION_3D=LOCATION_OUT[:,:3]
        ANGLE_2D=LOCATION_OUT[:,-2:]
        Position_output=output_5D[:,:3]
        Angle_output=output_5D[:,-2:]
        Location_loss= Position_output-LOCATION_3D
        Angle_loss= Angle_output-ANGLE_2D
        error2 = torch.mean(torch.linalg.norm(Location_loss,axis=1)**2,axis=0)
        error3 = torch.mean(torch.linalg.norm(Angle_loss,axis=1)**2,axis=0)
        print("训练:location loss:")
        print((error2)**0.5)
        print("训练:jiaodu loss:")
        print((error3)**0.5)
        Location_data= np.loadtxt('y_test_Y_TO_X_ONLY.txt') #0_60
        Mag_data = np.loadtxt('x_test_Y_TO_X_ONLY.txt') #
        # Location_data= np.loadtxt('data_verify_lowz/cc'+str(angle)+'_clean_no_outliers_delete.txt',delimiter=',') #0_60
        # Mag_data = np.loadtxt('data_verify_lowz/ss'+str(angle)+'_clean_no_outliers_delete.txt',delimiter=',') #
        # Location_data= np.loadtxt('data_verify/cc'+str(angle)+'.txt',delimiter=',') #0_60
        # Mag_data = np.loadtxt('data_verify/ss'+str(angle)+'.txt',delimiter=',') #
        Location_data=torch.tensor(np.array(Location_data,dtype=np.float32).reshape(-1,5)).to(device)
        Mag_data=torch.tensor(np.array(Mag_data,dtype=np.float32).reshape(-1,96)).to(device)
        mask = Location_data[:, 2] <= 37.5
        Location_data,Mag_data=Location_data[mask,:],Mag_data[mask,:]
        len1 = Location_data.shape[0]
        error = 0
        error1 = 0
        error2 = 0
        error3 = 0
        Mag_input = Mag_data
        output_5D = model_5D(Mag_input,device)
        LOCATION_OUT=Location_data
        LOCATION_3D=LOCATION_OUT[:,:3]
        ANGLE_2D=LOCATION_OUT[:,-2:]
        Position_output=output_5D[:,:3]
        Angle_output=output_5D[:,-2:]
        Location_loss= Position_output-LOCATION_3D
        Angle_loss= Angle_output-ANGLE_2D
        error2 = torch.mean(torch.linalg.norm(Location_loss,axis=1)**2,axis=0)
        error3 = torch.mean(torch.linalg.norm(Angle_loss,axis=1)**2,axis=0)
        print("验证:location loss:")
        print((error2)**0.5)
        print("验证:jiaodu loss:")
        print((error3)**0.5)
else:
    for i in range(36):
        angle=-175+10*i
        Location_data= np.loadtxt('DATASET_TEST/cc'+str(angle)+'.txt',delimiter=',') #0_60
        Mag_data = np.loadtxt('DATASET_TEST/ss'+str(angle)+'.txt',delimiter=',') #
        # Location_data= np.loadtxt('data_verify_lowz/cc'+str(angle)+'_clean_no_outliers_delete.txt',delimiter=',') #0_60
        # Mag_data = np.loadtxt('data_verify_lowz/ss'+str(angle)+'_clean_no_outliers_delete.txt',delimiter=',') #
        # Location_data= np.loadtxt('data_verify/cc'+str(angle)+'.txt',delimiter=',') #0_60
        # Mag_data = np.loadtxt('data_verify/ss'+str(angle)+'.txt',delimiter=',') #
        Location_data=torch.tensor(np.array(Location_data,dtype=np.float32).reshape(-1,5)).to(device)
        Mag_data=torch.tensor(np.array(Mag_data,dtype=np.float32).reshape(-1,96)).to(device)
        mask = Location_data[:, 2] <= 37.5
        Location_data,Mag_data=Location_data[mask,:],Mag_data[mask,:]
        len1 = Location_data.shape[0]
        error = 0
        error1 = 0
        error2 = 0
        error3 = 0
        Mag_input = Mag_data
        output_5D = model_5D(Mag_input,device)
        LOCATION_OUT=Location_data
        LOCATION_3D=LOCATION_OUT[:,:3]
        ANGLE_2D=LOCATION_OUT[:,-2:]
        Position_output=output_5D[:,:3]
        Angle_output=output_5D[:,-2:]
        Location_loss= Position_output-LOCATION_3D
        Angle_loss= Angle_output-ANGLE_2D
        error2 = torch.mean(torch.linalg.norm(Location_loss,axis=1)**2,axis=0)
        error3 = torch.mean(torch.linalg.norm(Angle_loss,axis=1)**2,axis=0)
        print(str(angle)+":location loss:")
        print((error2)**0.5)
        print(str(angle)+":jiaodu loss:")
        print((error3)**0.5)
        location_loss=(error2)**0.5
        Angle_loss=(error3)**0.5
        location_loss=location_loss.cpu().detach().numpy()
        Angle_loss=Ang=Angle_loss.cpu().detach().numpy()
        a = np.array([angle,location_loss,Angle_loss])
        # with open('Final_loss_valid_RES_4_9_2.txt', 'a+') as f:
        #     np.savetxt(f,a.reshape(1,3))#np.savetxt(r'test.txt', x, fmt='%d', newline='-|-')
        #     f.close()
    location_av=np.mean(a[:,1])
    angle_av=np.mean(a[:,2])
    print





