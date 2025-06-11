import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.svm import SVR
from codes.data_generation import dataset_read
import material_dataset
from sklearn.model_selection import GridSearchCV
import codes.model_code
from codes.sampling_strategy import ego,prediction
from codes.model_code import bagging_fit_predict
from codes.visualization_code import model_surface_plot
from sklearn.gaussian_process import GaussianProcessRegressor
import codes.test_function
import numpy as np
import copy
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from sklearn.metrics import r2_score
from scipy.spatial import Delaunay, ConvexHull
import joblib
from sklearn.manifold import TSNE
from matplotlib.patches import FancyArrowPatch
import os

def find_peaks(X, y_values, without_boundary=False):
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
            # 遇到无邻居的情况时跳过
            if len(neighbors_indices) == 0:
                continue
            if y_values[p_index] > np.max(neighbors_value) or y_values[p_index] < np.min(neighbors_value):
            # if y_values[p_index] < np.min(neighbors_value):  #极小值
                peaks_index.append(p_index)
    else:
        # 寻找采样中的峰值（极值），边界限制
        for p_index in range(len(points)):
            if p_index not in boundary_indices:
                start_index = vertex_neighbors[0][p_index]
                end_index = vertex_neighbors[0][p_index + 1]
                neighbors_indices = vertex_neighbors[1][start_index:end_index]  # 该点邻居索引
                neighbors_value = y_values[neighbors_indices]  # 该点邻居值
                if list(neighbors_value):
                    if y_values[p_index] > np.max(neighbors_value) or y_values[p_index] < np.min(neighbors_value):
                        peaks_index.append(p_index)
    return peaks_index

def find_neighbor_points(points_index ,X_all, y_all,neighbor_layer):
    tri = Delaunay(X_all)  # 三角形分割
    hull = ConvexHull(X_all)  # 凸包边界
    boundary_indices = hull.vertices  # 边界节点索引
    vertex_neighbors = tri.vertex_neighbor_vertices
    neighbor_index = set()

    for p_index in points_index:
        neighbor_index.add(p_index)
        neighbor_list=[p_index]
        #提取极值周围多层邻居,按neighbor_layer
        for nl in range(neighbor_layer):
            _list=[]
            for ni in neighbor_list:
                # print(ni)
                start_index = vertex_neighbors[0][ni]
                end_index = vertex_neighbors[0][ni + 1]
                pni_neighbors_indices = vertex_neighbors[1][start_index:end_index]  # 该点邻居索引
                for pnini in pni_neighbors_indices:
                    _list.append(pnini)
            # print(_list)
            for _i in _list:
                neighbor_list.append(_i)
                neighbor_index.add(_i)

    neighbor_index=list(neighbor_index)
    # print(neighbor_index)
    # # 极值示意图  仅二维测试函数可用
    # fig = plt.figure()
    # ax = p3d.Axes3D(fig)
    # fig.add_axes(ax)  # 版本问题，不加该行，显示为空白
    # # ax.scatter(np.asarray(X)[:, 0], np.asarray(X)[:, 1], y_values, color='k')
    # ax.scatter(np.asarray(X_all)[neighbor_index, 0], np.asarray(X_all)[neighbor_index, 1], y_all[neighbor_index],s=70, color='r',marker='*')
    #
    # ax.triplot(X_all[:, 0], X_all[:, 1], tri.simplices)
    # ax.scatter(X_all[neighbor_index, 0], X_all[neighbor_index, 1], color='r')
    # plt.show()
    return neighbor_index


def generate_random_points_around(data_point, radius, feature_num, n_samples):
    # 在特定半径内生成随机点
    random_directions = np.random.randn(n_samples, feature_num)
    random_directions /= np.linalg.norm(random_directions, axis=1).reshape(-1, 1)  # 单位向量化
    random_distances = np.random.uniform(0, radius, size=(n_samples, 1))  # 随机距离
    random_points = data_point + random_directions * random_distances  # 生成随机点
    return random_points


def sample_and_predict(model, X_StdScaler, y_StdScaler, data_points, feature_num, radius, n_samples):
    all_sampled_points = np.empty((0, feature_num))
    all_predictions = []

    for point in data_points:
        # 在每个数据点周围采样随机点
        sampled_points = generate_random_points_around(point, radius, feature_num, n_samples)


        # 使用模型预测采样点
        sampled_points_nor = _X_StdScaler.transform(sampled_points)
        predictions = model.predict(sampled_points_nor)
        predictions = y_StdScaler.inverse_transform(predictions.reshape(-1, 1))

        # 存储预测结果
        all_sampled_points = np.vstack([all_sampled_points, sampled_points])
        all_predictions.append(predictions)

    return all_sampled_points,all_predictions



if __name__ == '__main__':
    root_path = 'RunData\\PerovskiteStability_max_initial_sample_percent0.05 85'

    for dataset_sampling in os.listdir(root_path):
        print(dataset_sampling)
        dataset_path = os.path.join(root_path, dataset_sampling)
        initial_sample_num = eval(dataset_sampling.split(' ')[1])
        dataset_name = dataset_sampling.split('_')[0]
        opt_direction = dataset_sampling.split('_')[1]

        # if dataset_name!='DiffusionActivationEnergies':
        #     continue

        # 创建存图根文件夹
        root_save_path = root_path+'_Behavior_save'
        if not os.path.exists(root_save_path):
            os.mkdir(root_save_path)

        # 创建存图文件夹
        dataset_save_path = root_save_path+'\\'+dataset_name+'Behavior'
        if not os.path.exists(dataset_save_path):
            os.mkdir(dataset_save_path)


        X_all, y_all = dataset_read(dataset_name)
        feature_num=X_all.shape[1]

        # 在峰值样本的附近邻域内生成一些样本用模型预测

        # 训练模型 标准化数据
        _X_StdScaler = StandardScaler()
        _y_StdScaler = StandardScaler()
        _X_train_nor = _X_StdScaler.fit_transform(X_all)
        _y_train_nor = _y_StdScaler.fit_transform(y_all.reshape(-1, 1))  # 要求_y_train的shape必须是(-1, 1)
        # print(_X_train_nor)
        # print(_y_train_nor)
        # print(_X_train_nor.shape)
        # print(_y_train_nor.shape)
        # 定义SVR模型作为基模型
        dataset_ = getattr(material_dataset, dataset_name)()
        parameters = dataset_.parameters
        model = SVR()
        clf = GridSearchCV(model, parameters, cv=5, scoring='neg_mean_absolute_error')
        clf.fit(_X_train_nor, _y_train_nor.ravel())  # 要求_y_train_nor的shape必须是(-1,)
        # print(clf.best_params_)
        y_pre = clf.predict(_X_train_nor)
        y_pre = _y_StdScaler.inverse_transform(y_pre.reshape(-1, 1))  # 要求y_pre的shape必须是(-1, 1)
        # SVR R2
        r2_train = r2_score(y_all, y_pre)
        print(r2_train)


        peaks_index=find_peaks(X_all,y_all,without_boundary=True)
        print(peaks_index)
        peaks_X = X_all[peaks_index]
        peaks_y = y_all[peaks_index]

        # peaks_index=find_localmin_peaks(X_all,y_all)
        # peaks_neighbor_index=find_neighbor_points(peaks_index ,X_all, y_all,2)  #在数据集中寻找邻居
        # peaks_neighbor_X = X_all[peaks_neighbor_index]
        # peaks_neighbor_y = y_all[peaks_neighbor_index]

        radius = 1.0  # 邻域半径
        n_samples = 50  # 每个点采样随机点
        if dataset_name=='CurieTemperature':
            radius = 10  # 邻域半径
            n_samples = 100
        # 调用采样和预测函数
        peaks_neighbor_X,peaks_neighbor_y = sample_and_predict(clf,_X_StdScaler,_y_StdScaler, peaks_X, feature_num, radius, n_samples)



        for repeat_num in os.listdir(dataset_path):
            repeat_num_path = os.path.join(dataset_path, repeat_num)
            # sample_data.pkl
            bg_ei_sample_index_list, gp_ei_sample_index_list, bg_ei_sample_value_list, gp_ei_sample_value_list, bg_ei_r2_list, gp_ei_r2_list = joblib.load(repeat_num_path + '\\sample_data.pkl')

            # print(repeat_num_path)
            # 本次重复的初始采样序列
            initial_sample_index = bg_ei_sample_index_list[:initial_sample_num]
            print(repeat_num_path)

            initial_sample_X=X_all[initial_sample_index]

            all_bg_sample_X=X_all[bg_ei_sample_index_list]
            bg_ei_sample_neighbor_index=find_neighbor_points(bg_ei_sample_index_list ,X_all, y_all,1)
            all_bg_sample_neighbor_X = X_all[bg_ei_sample_neighbor_index]

            all_gp_sample_X = X_all[gp_ei_sample_index_list]
            gp_ei_sample_neighbor_index = find_neighbor_points(gp_ei_sample_index_list, X_all, y_all, 1)
            all_gp_sample_neighbor_X=X_all[gp_ei_sample_neighbor_index]



            # 将采样结果转换为NumPy数组
            # all_sample_X包括initial_sample_X
            # all_points = np.concatenate((all_bg_sample_X, all_gp_sample_X, peaks_neighbor_X),axis=0)
            all_points = np.concatenate((all_bg_sample_X, all_bg_sample_neighbor_X, all_gp_sample_X, all_gp_sample_neighbor_X, peaks_neighbor_X),axis=0)

            # 使用t-SNE进行降维
            if dataset_name=='PerovskiteStability':
                pp = 400
            elif dataset_name=='DiffusionActivationEnergies':
                pp = 30
            elif dataset_name=='EnergyStorageDensity':
                pp = 10
            elif dataset_name=='PerovskiteStabilityReduced':
                pp = 100
            elif dataset_name=='CurieTemperature':
                pp = 300

            tsne = TSNE(n_components=2, n_iter=300, perplexity=pp)  # 修改perplexity的值
            # tsne = TSNE(n_components=2,random_state=33,n_iter=300,perplexity=20)  # 修改perplexity的值
            embeddings = tsne.fit_transform(all_points)

            # 创建画布和子图
            # fig,ax=plt.subplots(figsize=(7, 5))
            fig,ax=plt.subplots(figsize=(15, 7)) # 扁
            # fig,ax=plt.subplots(figsize=(14, 8))

            sc=plt.scatter(embeddings[-peaks_neighbor_X.shape[0]:, 0], embeddings[-peaks_neighbor_X.shape[0]:, 1], s=50,
                        c=peaks_neighbor_y, cmap='rainbow', alpha=0.8)


            # 各取前20次采样画箭头
            arrow_num=20

            # 绘制BGEI散点图
            ti=initial_sample_X.shape[0]
            t0=ti+arrow_num
            # 初始样本
            plt.scatter(embeddings[:ti, 0],
                        embeddings[:ti, 1], s=50,marker='^', c='black',
                        alpha=0.7)
            # 前。。次采样样本
            plt.scatter(embeddings[ti:t0+1, 0],
                        embeddings[ti:t0+1, 1], s=50, c='grey',
                        alpha=0.5)

            # 绘制箭头连接相邻的两个采样点,不包括初始样本
            for i in range(ti, t0):
                arrow = FancyArrowPatch(embeddings[i], embeddings[i + 1], mutation_scale=10, arrowstyle='->',
                                        linestyle='--', connectionstyle='arc3,rad=0.1', linewidth=4, alpha=0.8,color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765))
                plt.gca().add_patch(arrow)


            # 绘制GPEI散点图
            t1=all_bg_sample_X.shape[0]+all_bg_sample_neighbor_X.shape[0]+initial_sample_X.shape[0]
            # t1=all_bg_sample_X.shape[0]+initial_sample_X.shape[0]
            t2=t1+arrow_num
            plt.scatter(embeddings[t1:t2+1, 0],
                        embeddings[t1:t2+1, 1], s=50, c='grey',
                        alpha=0.5, label='Samples')

            # 绘制箭头连接相邻的两个采样点,不包括初始样本
            for i in range(t1, t2):
                arrow = FancyArrowPatch(embeddings[i], embeddings[i + 1], mutation_scale=15, arrowstyle='->',
                                        linestyle='--', connectionstyle='arc3,rad=0.1', linewidth=4, alpha=0.8,
                                        color=(1.0, 0.4980392156862745, 0.054901960784313725))
                plt.gca().add_patch(arrow)

            # 编辑图例
            from matplotlib.lines import Line2D
            # 创建一个虚拟的Line2D对象用于图例
            # legend_scatter = Line2D([0], [0], marker='o',linestyle='',markersize=8, c='grey',alpha=0.5)
            legend_scatter = Line2D([0], [0], marker='^',linestyle='',markersize=8, c='black',alpha=0.7)
            legend_arrow1 = Line2D([0], [0], lw=3, linestyle='--',
                                   markersize=10, alpha=0.8,
                                   color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765))
            legend_arrow2 = Line2D([0], [0],  lw=3, linestyle='--',
                                  markersize=10, alpha=0.8,color=(1.0, 0.4980392156862745, 0.054901960784313725))

            # 添加图例
            # ax.legend([legend_scatter,legend_arrow1, legend_arrow2], ['Initial Samples','BGEI Sampling Trajectory','GPEI Sampling Trajectory'],fontsize=15, bbox_to_anchor=(1,1.02), loc="lower right", borderaxespad=0, ncol=3)
            ax.legend([legend_scatter,legend_arrow1, legend_arrow2], ['Initial Samples','BGEI Sampling Trajectory','BGEI-U$_{GP}$ Sampling Trajectory'],fontsize=15, bbox_to_anchor=(1,1.02), loc="lower right", borderaxespad=0, ncol=3)
            # plt.legend(fontsize=10, loc='upper right')

            # 添加颜色条
            cbar = plt.colorbar(sc, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=15)  # 设置色标刻度字体大小。
            # 设置颜色条的标签
            # cbar.set_label('Function Value', rotation=90, fontsize=20)
            cbar.set_label('Function Value (Maximize)', labelpad=-75, rotation=90, fontsize=15)

            # 设置刻度标签的字体大小
            plt.tick_params(axis='both', which='major', labelsize=15)

            plt.xlabel('t-SNE1', fontsize=20)
            plt.ylabel('t-SNE2', fontsize=20)
            # plt.title('BGEI vs GPEI', fontsize=15)

            plt.rcParams['font.sans-serif'] = ['Calibri']

            # plt.savefig(dataset_save_path+'\\'+str(initial_sample_num)+repeat_num+'2.png', dpi=250, bbox_inches='tight')
            plt.show()
            # plt.close()



