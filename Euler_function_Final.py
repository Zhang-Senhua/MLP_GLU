import numpy as np
import torch
import math
from scipy.linalg import block_diag
# 定义device变量


def Mag_Caculate(Pole_location,Sensor_Location,Mag_m1,device):
    """
       计算磁场强度矩阵函数
       :param Pole_location: 磁极位置信息矩阵
       :type Pole_location: tensor五维磁极信息（5*n）
       :param Sensor_Location: 传感器矩阵坐标
       :type Sensor_Location: tensor三维传感器信息（3*32）
       :param Mag_m1: 磁常数
       :type Mag_m1: null
       :param device: 运行设备
       :type device: device
       :return: 输出的result维度为3*（32*n）,32组传感器在n个磁极坐标下的磁场强度,
       格式为：一个磁极坐标有32个磁场强度，[第一组（第一个磁极坐标产生的磁场强度），第二组（第二个磁极坐标产生的磁场强度）]
       :rtype: tensor
       """
    # STEP 1按块复制s
    u0 = 4 * torch.Tensor([math.pi]) * (10 ** -7)
    #提取一个共有多少组磁极坐标位置
    Pole_Count=Pole_location.shape[1]
    #对传感器位置进行复制重构，按块复制，一块表示32组传感器坐标位置
    #Sensor_final = torch.tile(Sensor_Location, (Pole_Count, 0)).T.to(device)
    Sensor_final = Sensor_Location.repeat(1, Pole_Count).clone().requires_grad_(True)
    #Sensor_final=Sensor_Location.repeat(1,Pole_Count).requires_grad_(True)
    #对磁极坐标进行复制重构，原始数据列需要复制32份（32个传感器坐标）
    repeated_Pole_location = torch.repeat_interleave(Pole_location, 32, dim=1).to(device).clone().requires_grad_(True)
    #repeated_Pole_location = torch.repeat_interleave(Pole_location, 32, dim=1).to(device).requires_grad_(True)
    #提取3维磁极坐标信息
   # p = repeated_Pole_location[:3, :]
    p = repeated_Pole_location[:3, :].clone()
    #提取磁极角度信息
    #theta = repeated_Pole_location[-2:-1, :]
    theta = repeated_Pole_location[-2:-1, :].clone()
    theta = torch.deg2rad(theta)
    #弧度转换
    #theta = torch.deg2rad(theta)
    # phi = repeated_Pole_location[-1:, :]
    # phi = torch.deg2rad(phi)
    phi = repeated_Pole_location[-1:, :].clone()
    phi = torch.deg2rad(phi)

    #求解m矩阵，重构为3*（32*n矩阵）
    # m = torch.stack([Mag_m1 * torch.sin(theta) * torch.cos(phi), Mag_m1 * torch.sin(theta) * torch.sin(phi)
    #                  , Mag_m1 * torch.cos(theta)]).requires_grad_(True)
    m = torch.stack([Mag_m1 * torch.sin(theta) * torch.cos(phi), Mag_m1 * torch.sin(theta) * torch.sin(phi)
                        , Mag_m1 * torch.cos(theta)]).clone().requires_grad_(True)

    m=m.reshape(3,-1)
    #求解s-p

    diff = torch.Tensor(Sensor_final - p).to(device)
    diff_norm = torch.norm(Sensor_final - p,dim=0)
    k = m.T
    TEMP = torch.matmul(k, diff)
    ##对于TEMP ##对于TEMP乘（s-p）,注意我们只需要实现TEMP每一列，乘以diff每一行得到1*（32*n）的数组
    # 提取对角线上的元素
    diagonal_elements = torch.diagonal(TEMP)
    # 将结果转换为 1 x 1024 的矩阵
    M_T_P = diagonal_elements.reshape(1, -1)
    # 将 a 矩阵扩展为 3 x 1024 的矩阵
    a_expanded = torch.tile(M_T_P, (3, 1))
    # 直接相乘
    result = a_expanded * diff
    a = 3 * result
    b = (diff_norm ** 2)*m
    c = diff_norm ** 5
    #最后输出的result维度为3*（32*n）,32组传感器在n个磁极坐标下的磁场强度
    #格式为：一个磁极坐标有32个磁场强度，[第一组（第一个磁极坐标产生的磁场强度），第二组（第二个磁极坐标产生的磁场强度）]
    #result = (u0 / (4 * torch.pi)) * ((a - b) / c).requires_grad_(True)
    result = (u0 / (4 * torch.pi)) * ((a - b) / c).clone().requires_grad_(True)
    #保留梯度
    result.retain_grad()
    return result
def Euler_Vectorlized(GHO_arams,Pole_location,Sensor_Location,Mag_m1,device):
    """
         欧拉方程
         :param GHO_arams: 磁极位置信息矩阵
         :type arg1: tensor五维磁极信息（5*n）
         :param arg2: 传感器矩阵坐标
         :type arg2: tensor五维磁极信息（3*32）
         :param arg3: 磁常数
         :type arg3: null
         :param arg4: 运行设备
         :type arg4: device
         :return: 输出的result维度为3*（32*n）,32组传感器在n个磁极坐标下的磁场强度,
         格式为：一个磁极坐标有32个磁场强度，[第一组（第一个磁极坐标产生的磁场强度），第二组（第二个磁极坐标产生的磁场强度）]
         :rtype: tensor
         """
    GHO_arams=torch.Tensor(GHO_arams).requires_grad_(True)
    Pole_location=torch.Tensor(Pole_location)
    Sensor_Location=torch.Tensor(Sensor_Location)
    Pole_Count = Pole_location.shape[1]
    GHO_arams=GHO_arams.reshape(32,9)
    G =  GHO_arams[:,:3]
    G=G.reshape(1,96)
    H =  GHO_arams[:,3:6]
    H= H.reshape(32, 3)
    O = GHO_arams[:,-3:]
    O=O.T
    O=torch.tile(O,(1,Pole_Count))
    O=O.reshape(3,32*Pole_Count)
    G_diag_matrix = torch.diag(G[0])  # G是1*96维
    # 生成随机输入矩阵
   # G_diag_matrix=torch.tensor(G_diag_matrix)
    G_diag_matrix = torch.diag(G[0]).clone()
    input_matrix = H * (math.pi / 180)
    cos_z = torch.cos(input_matrix[:, 2])
    sin_z = torch.sin(input_matrix[:, 2])
    cos_y = torch.cos(input_matrix[:, 1])
    sin_y = torch.sin(input_matrix[:, 1])
    cos_x = torch.cos(input_matrix[:, 0])
    sin_x = torch.sin(input_matrix[:, 0])

    Rz = torch.stack([cos_z, -sin_z, torch.zeros_like(cos_z),
                      sin_z, cos_z, torch.zeros_like(cos_z),
                      torch.zeros_like(cos_z), torch.zeros_like(cos_z), torch.ones_like(cos_z)], dim=1).view(-1, 3, 3)

    Ry = torch.stack([cos_y, torch.zeros_like(cos_y), sin_y,
                      torch.zeros_like(cos_y), torch.ones_like(cos_y), torch.zeros_like(cos_y),
                      -sin_y, torch.zeros_like(cos_y), cos_y], dim=1).view(-1, 3, 3)

    Rx = torch.stack([torch.ones_like(cos_x), torch.zeros_like(cos_x), torch.zeros_like(cos_x),
                      torch.zeros_like(cos_x), cos_x, -sin_x,
                      torch.zeros_like(cos_x), sin_x, cos_x], dim=1).view(-1, 3, 3)
    # 三个旋转矩阵相乘得到转换矩阵
    transform_matrix = torch.bmm(torch.bmm(Rz, Ry), Rx)
    #为了能和G相乘，需要将R阵对角化
    R_DIAG = torch.block_diag(*transform_matrix)
    R_DIAG=torch.tensor(R_DIAG).to(device)
    output_matrix =  transform_matrix.reshape(-1, 3)
    TEMP_GH=torch.matmul(G_diag_matrix,R_DIAG)
    #去对角化
    GH_FINAL=c = torch.einsum('ijik->ijk', TEMP_GH.reshape(32, 3, 32, 3)).reshape(96, 3)
    MAG=Mag_Caculate(Pole_location,Sensor_Location,Mag_m1,device)
    #Euler_FINAL=Mag_Caculate(Pole_location,Sensor_Location,Mag_m1,device).requires_grad_(True)#测试行
    TEMP_GH=torch.cat([GH_FINAL]*Pole_Count, dim=0)
    #TEMP_1=TEMP_GH.numpy()
    # 将a矩阵转换为三维数组
    a = TEMP_GH.reshape((32*Pole_Count, 3, 3))
    # 将b矩阵转置
    # 将b矩阵转置
    b = MAG.T
    # 使用广播机制进行矩阵乘法
    c = a * b[:, None]
    # 对第二个轴求和
    c = c.sum(axis=2)
    # 保留梯度,返回格式：3*（32*n）
    Euler_FINAL=c.T+O
    Euler_FINAL = Euler_FINAL.flatten().requires_grad_(True)
    #Euler_FINAL=torch.ravel(Euler_FINAL, order='F').requires_grad_(True)
    Euler_FINAL= torch.reshape(Euler_FINAL, (-1, 1)).requires_grad_(True)
    Euler_FINAL=torch.reshape(Euler_FINAL,(-1,96)).requires_grad_(True)
    Euler_FINAL.requires_grad_(True)

    return Euler_FINAL
if __name__ == "__main__":
    l = 1  # 厚度
    d = 6  # 直径
    fr = 18500  # 剩余磁场
    u0 = 4 * math.pi * (10 ** -7)
    pi = math.pi
    m1 = (((math.pi) * d * d * l * fr) / (4 * u0))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pole_test=np.loadtxt('pole_test.txt')
    pole_test=torch.tensor(pole_test).to(device)
    Sensor_Location=np.loadtxt('Sensor_Location.txt')
    Sensor_Location = torch.tensor(Sensor_Location).to(device)
    GHO=np.loadtxt('GHO_test.txt')
    GHO = torch.tensor(GHO).to(device)
    Result=Euler_Vectorlized(GHO, pole_test, Sensor_Location, m1, device)

