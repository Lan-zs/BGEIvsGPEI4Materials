import test_function
import numpy as np
import random

# 生成网格数据，虚拟样本空间,同时用于画图数据,输入测试函数类的名字+分度值
def testfunc_grid_generation(testfunction_name, dt=None):
    test_func = getattr(test_function, testfunction_name)() #取出测试函数类并实例化
    if dt == None:
        dt=test_func.dt
    # print(test_func.name)
    # print(test_func.input_range)
    ranges = [np.arange(test_func.input_range[0],test_func.input_range[1]+dt,step=dt) for n in range(test_func.num_variables)]
    # print(ranges)
    grid = np.meshgrid(*ranges) #生成网格
    # print(grid[0])
    # print(grid[1])
    grid = np.stack(grid, axis=-1)
    # print(grid)
    grid_points = grid.reshape(-1, grid.shape[-1]) #将网格展开成数据条
    # print(len(grid_points))
    # print(grid_points)
    # x1_grid, x2_grid = np.meshgrid(x1, x2)
    return grid_points

# 测试函数初始样本采集：随机采样
def testfunc_initial_sample(grid_points, testfunction_name, num):
    test_func = getattr(test_function, testfunction_name)()  #取出测试函数类并实例化
    sample_index = random.sample([i for i in range(len(grid_points))], num) #随机抽取数据索引,numpy可以用len
    sample_points = grid_points[sample_index]
    # print(sample_points)
    y = np.array([test_func.calculate_output(sp) for sp in sample_points])
    # print(y.shape)
    return sample_points, y, sample_index

# 计算给定样本的测试函数y值
def testfunc_value(X, testfunction_name):
    test_func = getattr(test_function, testfunction_name)()  # 取出测试函数类并实例化
    y = np.array([test_func.calculate_output(sp) for sp in X])
    # print(y)
    # print(y.shape)
    return y

def dataset_read(dataset_name,pca=True):
    if not pca:
        X_all = np.load('MaterialDataset\\'+dataset_name+'\\all_x_values.npy')
        y_all = np.load('MaterialDataset\\'+dataset_name+'\\all_y_values.npy')
    else:
        X_all = np.load('MaterialDataset\\' + dataset_name + '\\all_x_values_pca.npy')
        y_all = np.load('MaterialDataset\\' + dataset_name + '\\all_y_values.npy')
    return X_all,y_all

# 数据集初始样本采集：随机采样
def dataset_initial_sample(X_all, y_all, num):
    sample_index = random.sample([i for i in range(len(X_all))], num) #随机抽取数据索引
    X_sample = X_all[sample_index]
    y_sample =y_all[sample_index]
    return X_sample, y_sample, sample_index




if __name__ == '__main__':
    testfunction_name = 'Hartmann3Function'
    X_all = testfunc_grid_generation(testfunction_name)
    print(len(X_all))
    X_train, y_train, sample_index = testfunc_initial_sample(X_all, testfunction_name, 10)
    y_all = testfunc_value(X_all, testfunction_name)

    # test_func = getattr(test_function, testfunction_name)()
    # X_all_values=np.array([test_func.calculate_output(sp) for sp in X_all]).reshape(-1,1)
    #
    # import matplotlib.pyplot as plt
    #
    # # 绘制三维散点图
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(x, y, z, c=values, s=100*values)
    # ax.scatter(np.asarray(X_all)[:, 0], np.asarray(X_all)[:, 1], np.asarray(X_all)[:, 2], c=X_all_values, cmap='rainbow')
    #
    # # 设置图形参数
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('3D Scatter Plot')
    #
    # # 显示图形
    # plt.show()
