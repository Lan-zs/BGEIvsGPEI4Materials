import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

'''
储存材料数据集的基本参数：SVR最优超参数、GP超参数
'''

class EnergyStorageDensity:
    def __init__(self):
        self.name = 'EnergyStorageDensity'
        self.parameters = {'C': [1, 10, 100], 'gamma': [0.5, 1, 1.5]} #有/无PCA
        self.gp_kernel=C(1, (0.1, 10)) * RBF(1, (10, 30))+WhiteKernel(1,(1,2))


class CurieTemperature:
    def __init__(self):
        self.name = 'CurieTemperature'
        self.parameters = {'C': [1, 10, 100], 'gamma': [0.005, 0.01, 0.015]} #有/无PCA
        self.gp_kernel = C(1, (0.1, 10)) * RBF(1, (12, 20))+WhiteKernel(0.1,(0.1,0.15))

class PerovskiteStability:
    def __init__(self):
        self.name = 'PerovskiteStability'
        # self.parameters = {'C': [1, 10, 100], 'gamma': [0.05, 0.1, 0.15]} #无PCA
        self.parameters = {'C': [1, 10, 100], 'gamma': [5,10,15]}
        self.gp_kernel=C(1, (0.1, 100)) * RBF(1, (0.5, 3))+WhiteKernel(80,(50,80))

class DiffusionActivationEnergies:
    def __init__(self):
        self.name = 'DiffusionActivationEnergies'
        self.parameters = {'C': [1, 10, 100], 'gamma': [0.5, 1, 1.5]}
        # self.parameters = {'C': [1, 10, 100], 'gamma': [0.05, 0.1, 0.15]}  #无PCA
        self.gp_kernel = C(1, (0.1, 1)) * RBF(1, (2.4,5))+WhiteKernel(0.01,(0.01,0.2))

class PerovskiteStabilityReduced:
    def __init__(self):
        self.name = 'PerovskiteStabilityReduced'
        # self.parameters = {'C': [1, 10, 100], 'gamma': [0.05, 0.1, 0.15]} #无PCA
        self.parameters = {'C': [1, 10, 100], 'gamma': [5,10,15]}
        self.gp_kernel=C(1, (0.1, 100)) * RBF(1, (0.5, 3))+WhiteKernel(80,(50,80))
