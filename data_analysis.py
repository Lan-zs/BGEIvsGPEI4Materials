from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from data_generation import testfunc_grid_generation, testfunc_initial_sample
import test_function


def find_peaks(X, y_values,without_boundary=False):
    points = np.array(X)
    tri = Delaunay(points)  # 三角形分割
    hull = ConvexHull(points)  # 凸包边界
    boundary_indices = hull.vertices  # 边界节点索引
    vertex_neighbors = tri.vertex_neighbor_vertices
    peaks_index = []
    # 寻找采样中的峰值（极值），无边界限制
    if not without_boundary:
        for p_index in range(len(points)):
            start_index = vertex_neighbors[0][p_index]
            end_index = vertex_neighbors[0][p_index + 1]
            neighbors_indices = vertex_neighbors[1][start_index:end_index]  # 该点邻居索引
            neighbors_value = y_values[neighbors_indices]  # 该点邻居值
            # plt.triplot(points[:, 0], points[:, 1], tri.simplices)
            # plt.plot(points[p_index, 0], points[p_index, 1], 'o')
            # plt.scatter(points[neighbors_indices, 0], points[neighbors_indices, 1], color='r')
            # plt.show()
            if y_values[p_index] > np.max(neighbors_value) or y_values[p_index] < np.min(neighbors_value):
                peaks_index.append(p_index)
    else:
        # 寻找采样中的峰值（极值），边界限制
        for p_index in range(len(points)):
            if p_index not in boundary_indices:
                start_index = vertex_neighbors[0][p_index]
                end_index = vertex_neighbors[0][p_index + 1]
                neighbors_indices = vertex_neighbors[1][start_index:end_index]  # 该点邻居索引
                neighbors_value = y_values[neighbors_indices]  # 该点邻居值
                # plt.triplot(points[:, 0], points[:, 1], tri.simplices)
                # plt.plot(points[p_index, 0], points[p_index, 1], 'o')
                # plt.scatter(points[neighbors_indices, 0], points[neighbors_indices, 1], color='r')
                # plt.show()
                if y_values[p_index] > np.max(neighbors_value) or y_values[p_index] < np.min(neighbors_value):
                    peaks_index.append(p_index)

    # # 极值示意图  仅二维测试函数可用
    # fig = plt.figure()
    # ax = p3d.Axes3D(fig)
    # fig.add_axes(ax)  # 版本问题，不加该行，显示为空白
    # ax.scatter(np.asarray(X)[:, 0], np.asarray(X)[:, 1], y_values, color='k')
    # ax.scatter(np.asarray(X)[peaks_index, 0], np.asarray(X)[peaks_index, 1], y_values[peaks_index],s=70, color='r',marker='*')
    #
    # ax.triplot(points[:, 0], points[:, 1], tri.simplices)
    # ax.scatter(points[peaks_index, 0], points[peaks_index, 1], color='r')
    # plt.show()

    return peaks_index

# 寻找采样中的极小值，边界限制
def find_localmin_peaks(X, y_values):
    points = np.array(X)
    tri = Delaunay(points)  # 三角形分割
    hull = ConvexHull(points)  # 凸包边界
    boundary_indices = hull.vertices  # 边界节点索引
    vertex_neighbors = tri.vertex_neighbor_vertices
    peaks_index = []

    for p_index in range(len(points)):
        if p_index not in boundary_indices:
            start_index = vertex_neighbors[0][p_index]
            end_index = vertex_neighbors[0][p_index + 1]
            neighbors_indices = vertex_neighbors[1][start_index:end_index]  # 该点邻居索引
            neighbors_value = y_values[neighbors_indices]  # 该点邻居值
            # plt.triplot(points[:, 0], points[:, 1], tri.simplices)
            # plt.plot(points[p_index, 0], points[p_index, 1], 'o')
            # plt.scatter(points[neighbors_indices, 0], points[neighbors_indices, 1], color='r')
            # plt.show()
            if y_values[p_index] < np.min(neighbors_value):
                peaks_index.append(p_index)
    return peaks_index

# 寻找采样中的极大值，边界限制
def find_localmax_peaks(X, y_values):
    points = np.array(X)
    tri = Delaunay(points)  # 三角形分割
    hull = ConvexHull(points)  # 凸包边界
    boundary_indices = hull.vertices  # 边界节点索引
    vertex_neighbors = tri.vertex_neighbor_vertices
    peaks_index = []

    for p_index in range(len(points)):
        if p_index not in boundary_indices:
            start_index = vertex_neighbors[0][p_index]
            end_index = vertex_neighbors[0][p_index + 1]
            neighbors_indices = vertex_neighbors[1][start_index:end_index]  # 该点邻居索引
            neighbors_value = y_values[neighbors_indices]  # 该点邻居值
            # plt.triplot(points[:, 0], points[:, 1], tri.simplices)
            # plt.plot(points[p_index, 0], points[p_index, 1], 'o')
            # plt.scatter(points[neighbors_indices, 0], points[neighbors_indices, 1], color='r')
            # plt.show()
            if y_values[p_index] > np.max(neighbors_value):
                peaks_index.append(p_index)
    return peaks_index


# 计算Cook距离
def find_strong_influences(X, y):
    # 添加常数列
    X = sm.add_constant(X)

    # 构建线性回归模型
    model = sm.OLS(y, X)

    # 拟合模型
    results = model.fit()

    # 计算Cook's距离
    influence = OLSInfluence(results)
    cooks_distance = influence.cooks_distance
    # 找到强影响点的索引
    # strong_influences = np.where(cooks_distance[0] > 4 / len(y))[0]
    strong_influences = np.where(cooks_distance[0] > 4 / len(y))[0]
    # print(cooks_distance[0])
    # print(strong_influences)
    if strong_influences.shape[0]<4:
        strong_influences=np.argsort(cooks_distance[0])[-3:] # 取前三强影响的样本作为强影响点

    # print("强影响点的索引：", strong_influences)
    return list(strong_influences), cooks_distance[0]

# 计算Cook距离
def calculate_cooks_distance(X, y):
    # 添加常数列
    X = sm.add_constant(X)

    # 构建线性回归模型
    model = sm.OLS(y, X)

    # 拟合模型
    results = model.fit()

    # 计算Cook's距离
    influence = OLSInfluence(results)
    cooks_distance = influence.cooks_distance

    return cooks_distance[0]

# 高斯核
def gaussian_kernel(x, xi, gamma):
    return np.exp(-gamma * np.linalg.norm(x - xi) ** 2)

# 计算残差影响
def residual_influences_gamma(xi, yi, x_all, gamma):
    res_list = [abs(y - np.mean(yi)) for y in yi]  # 计算残差
    res_influences = []
    for j in x_all:
        j_ri = []
        for i_index, i in enumerate(xi):
            j_ri.append(gaussian_kernel(i, j, gamma) * res_list[i_index])
        res_influences.append(sum(j_ri))
    return np.array(res_influences)

# 计算残差影响
def residual_influences(xi, yi, x_all, testfunction_name):
    test_func = getattr(test_function, testfunction_name)()  # 取出测试函数类并实例化
    gamma = test_func.statistics_gamma
    res_list = [abs(y - np.mean(yi)) for y in yi]  # 计算残差
    res_influences = []
    for j in x_all:
        j_ri = []
        for i_index, i in enumerate(xi):
            j_ri.append(gaussian_kernel(i, j, gamma) * res_list[i_index])
        res_influences.append(sum(j_ri))
    return np.array(res_influences)


# 计算cook影响
def cook_influences(xi, xi_cooks_value, x_all, testfunction_name):
    test_func = getattr(test_function, testfunction_name)()  # 取出测试函数类并实例化
    gamma = test_func.statistics_gamma
    co_influences = []
    for j in x_all:
        j_ci = []
        for i_index, i in enumerate(xi):
            j_ci.append(gaussian_kernel(i, j, gamma) * xi_cooks_value[i_index])
        co_influences.append(sum(j_ci))
    return np.array(co_influences)

# 邻居残差影响
def neighbor_residual_influences_gamma(xi, yi, x_all, gamma):
    points = np.array(xi)
    tri = Delaunay(points)  # 三角形分割
    vertex_neighbors = tri.vertex_neighbor_vertices
    neighbor_residual_list = []
    no_neighbors_index_list=[]
    # 寻找采样中的峰值（极值），无边界限制
    for p_index in range(len(points)):
        start_index = vertex_neighbors[0][p_index]
        end_index = vertex_neighbors[0][p_index + 1]
        neighbors_indices = vertex_neighbors[1][start_index:end_index]  # 该点邻居索引
        # print('neighbors_indices: ', neighbors_indices)
        # 去除无邻居的点
        if len(neighbors_indices)!=0:
            neighbors_value = yi[neighbors_indices]  # 该点邻居值
            neighbor_residual=np.abs(yi[p_index]-np.mean(neighbors_value))
            neighbor_residual_list.append(neighbor_residual)
        else:
            no_neighbors_index_list.append(p_index)
    xi=np.delete(xi,no_neighbors_index_list,axis=0) # 最后统一删除
    # print(neighbor_residual_list)
    n_res_influences = []
    for j in x_all:
        j_ri = []
        for i_index, i in enumerate(xi):
            j_ri.append(gaussian_kernel(i, j, gamma) * neighbor_residual_list[i_index])
        n_res_influences.append(sum(j_ri))
    # print(n_res_influences)
    return n_res_influences

if __name__ == '__main__':
    testfunction_name = 'SumSquaresFunction'
    grid_points = testfunc_grid_generation(testfunction_name, 0.1)
    sample_points, y, sample_index = testfunc_initial_sample(grid_points, testfunction_name, 10)
    # peaks_index = dataset_find_peaks(sample_points, y)  # 测试find peaks
    peaks_index = find_peaks(sample_points, y)  # 测试find peaks
    strong_influences, cooks_distance = find_strong_influences(sample_points, y)  # 测试find peaks
    print("极值的索引：", peaks_index)
    print("强影响点的索引：", strong_influences)

    # # 验证对于高维函数可以正确找到极值
    # testfunction_name = 'Hartmann3Function'
    # X_all = grid_generation(testfunction_name)
    # X_train, y_train, sample_index = initial_sample(X_all, testfunction_name, 20)
    #
    # import codes.test_function
    #
    # test_func = getattr(codes.test_function, testfunction_name)()
    # X_all_values = np.array([test_func.calculate_output(sp) for sp in X_all]).reshape(-1, 1)
    # print(X_all_values)
    #
    # global_peaks_index = find_peaks(X_all, X_all_values)
    # peaks_index = find_peaks(X_train, y_train)
    #
    # import matplotlib.pyplot as plt
    #
    # # 绘制三维散点图
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(x, y, z, c=values, s=100*values)
    # ax.scatter(np.asarray(X_all)[:, 0], np.asarray(X_all)[:, 1], np.asarray(X_all)[:, 2], c=X_all_values, cmap='rainbow', alpha=0.15)
    # ax.scatter(np.asarray(X_all)[global_peaks_index, 0], np.asarray(X_all)[global_peaks_index, 1], np.asarray(X_all)[global_peaks_index, 2], s=70, color='k', marker='*', alpha=1)
    #
    # # ax.scatter(np.asarray(X_all)[sample_index, 0], np.asarray(X_all)[sample_index, 1], np.asarray(X_all)[sample_index, 2], c=X_all_values[sample_index], cmap='rainbow', alpha=1)
    # # ax.scatter(np.asarray(X_train)[peaks_index, 0], np.asarray(X_train)[peaks_index, 1], np.asarray(X_train)[peaks_index, 2], s=100, color='grey', marker='*', alpha=0.4)
    #
    # print(X_all[peaks_index])
    #
    # # 设置图形参数
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Hartmann3Function')
    #
    # # 显示图形
    # plt.show()
