import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


class SinglePeakFunction:
    def __init__(self):
        self.name = 'SinglePeakFunction'
        self.input_range = (-3, 3)  # 自变量的取值范围
        self.num_variables = 2  # 自变量的数量
        self.dt = 0.1
        self.parameters = {'C': [1, 10, 100], 'gamma': [0.5, 1, 1.5]}
        self.gp_kernel=C(1, (0.1, 10)) * RBF(1, (0.1, 10))


    def calculate_output(self, x):
        output = 0.5* (1 - x[0]) ** 2 * np.exp(-(x[0] ** 2) - (x[1] + 1) ** 2)
        return output

class SumSquaresFunction:
    def __init__(self):
        self.name = 'SumSquaresFunction'
        self.input_range = (-2, 2)  # SUM SQUARES FUNCTION的取值范围
        self.num_variables = 2  # 自变量数量
        self.dt = 0.1
        self.parameters = {'C': [1, 10, 100], 'gamma': [0.05, 0.1, 0.15]}
        self.gp_kernel=C(1, (0.1, 10)) * RBF(1, (0.1, 10))

    def calculate_output(self, x):
        result = np.sum(np.square(x))
        return result

class PeaksFunction:
    def __init__(self):
        self.name = 'PeaksFunction'
        self.input_range = (-3, 3)  # 自变量的取值范围
        self.dt = 0.1
        self.num_variables = 2  # 自变量的数量
        self.parameters = {'C': [1, 10, 100], 'gamma': [0.5, 1, 1.5]}
        self.gp_kernel =C(1, (0.1, 10)) * RBF(1, (0.1, 10))

    def calculate_output(self, x):
        output = 3 * (1 - x[0]) ** 2 * np.exp(-(x[0] ** 2) - (x[1] + 1) ** 2) - 10 * (x[0] / 5 - x[0] ** 3 - x[1] ** 5) * np.exp(
            -x[0] ** 2 - x[1] ** 2) - 1 / 3 * np.exp(-(x[0] + 1) ** 2 - x[1] ** 2)
        return output

class SixHumpCamelFunction:
    def __init__(self):
        self.name = "SixHumpCamelFunction"
        self.input_range = (-1, 1) # 取值范围
        self.dt = 0.05
        self.num_variables = 2  # 自变量数量
        self.parameters = {'C': [1, 10, 100], 'gamma': [0.5, 1, 1.5]}
        self.gp_kernel =C(1, (0.1, 10)) * RBF(1, (0.1, 10))

    def calculate_output(self, x):
        # 输入自变量 x，输出函数的值
        x1, x2 = x[0], x[1]
        term1 = (4 - 2.1 * x1**2 + (x1**4)/3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2
        return term1 + term2 + term3

class ShubertFunction:
    def __init__(self):
        self.name = 'ShubertFunction'
        self.input_range = (-2,2)  # SHUBERT FUNCTION的取值范围
        self.num_variables = 2  # 自变量数量
        # self.dt = 0.01  # 采样分度值
        self.dt = 0.1
        self.parameters = {'C': [1, 10, 100], 'gamma': [5, 10, 15]}
        self.gp_kernel =C(1, (0.1, 10)) * RBF(1, (0.5, 1)) #0.1的核太窄了
        # self.gp_kernel =C(1.0, constant_value_bounds="fixed") * RBF( 1, length_scale_bounds="fixed" )

    def calculate_output(self, x):
        sum1 = 0
        sum2 = 0
        for i in range(1, 6):
            term1 = i * np.cos((i + 1) * x[0] + i)
            term2 = i * np.cos((i + 1) * x[1] + i)
            sum1 += term1
            sum2 += term2
        result = sum1 * sum2
        return result

class Hartmann3Function:
    def __init__(self):
        self.name = 'Hartmann3Function'
        self.num_variables = 3
        self.input_range = (0, 1)  # 自变量的取值范围
        self.dt = 0.05
        self.parameters = {'C': [1, 10, 100], 'gamma': [0.05, 0.1, 0.15]}
        self.gp_kernel =C(1, (0.1, 10)) * RBF(1, (0.1, 1))

    def calculate_output(self, x):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[3.0, 10, 30],
                      [0.1, 10, 35],
                      [3.0, 10, 30],
                      [0.1, 10, 35]])
        P = 0.0001 * np.array([[3689, 1170, 2673],
                               [4699, 4387, 7470],
                               [1091, 8732, 5547],
                               [381, 5743, 8828]])

        outer_sum = 0
        for i in range(4):
            inner_sum = 0
            for j in range(3):
                inner_sum += A[i, j] * ((x[j] - P[i, j]) ** 2)
            outer_sum += alpha[i] * np.exp(-inner_sum)

        return -outer_sum
