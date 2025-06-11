import numpy as np
from model_code import base_model_training,bagging_fit_predict
import os
from scipy.spatial import Delaunay, ConvexHull
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import r2_score
from data_generation import testfunc_initial_sample, testfunc_grid_generation,testfunc_value
from data_analysis import residual_influences_gamma
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



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
                peaks_index.append(p_index)
    else:
        # 寻找采样中的峰值（极值），边界限制
        for p_index in range(len(points)):
            if p_index not in boundary_indices:
                start_index = vertex_neighbors[0][p_index]
                end_index = vertex_neighbors[0][p_index + 1]
                neighbors_indices = vertex_neighbors[1][start_index:end_index]  # 该点邻居索引
                neighbors_value = y_values[neighbors_indices]  # 该点邻居值
                if y_values[p_index] > np.max(neighbors_value) or y_values[p_index] < np.min(neighbors_value):
                    peaks_index.append(p_index)

    return peaks_index

def errorbar_plot(mean_list, std_list, sample_num_list, statistics_name_list, image_path):
    color_list = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                  (1.0, 0.4980392156862745, 0.054901960784313725),
                  (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
                  (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                  (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                  (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
                  (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
                  (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
                  (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
                  (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]
    plt.figure(figsize=(12, 8))
    for i in range(len(statistics_name_list)):
        plt.errorbar(sample_num_list, mean_list[i], yerr=std_list[i], fmt='-o', lw=2, color=color_list[i], elinewidth=3,
                     ms=9, capsize=6, capthick=3, label=statistics_name_list[i], alpha=1, zorder=2)
    plt.xlabel('sample_num')
    plt.ylabel('Correlation')
    plt.legend(fontsize=12, ncol=2, loc='lower left')
    plt.title('Errorbar Plot')
    plt.ylim(-1, 1)
    plt.xlim(plt.xlim()[0], plt.xlim()[1])
    plt.axhline(y=0.5, linestyle='--', color='red', zorder=1)
    plt.axhline(y=0, linestyle='--', color='gray', zorder=1)
    plt.axhline(y=-0.5, linestyle='--', color='blue', zorder=1)
    plt.fill_betweenx([0.5, plt.ylim()[1]], plt.xlim()[0], plt.xlim()[1], color='red', alpha=0.1)  # 在虚线上方绘制阴影区域
    plt.fill_betweenx([plt.ylim()[0], -0.5], plt.xlim()[0], plt.xlim()[1], color='blue', alpha=0.1)  # 在虚线下方绘制阴影区域
    plt.savefig(image_path + '\\errorbar_allinfluence.jpg', dpi=200, bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    import datetime

    testfunc_name_list = ['PeaksFunction','SinglePeakFunction','SumSquaresFunction','SixHumpCamelFunction','Hartmann3Function','ShubertFunction']
    base_model_list = ['SVR']
    statistics_name_list = ['gp_std_cor', 'gpstd_kde_cor', 'peaks_residual_influences_cor', 'all_residual_influences_cor', 'kernel_density_estimation_cor', 'p_combine_influences_cor', 'a_combine_influences_cor']

    bagging_save_path = 'RunData\\testfunc_bagging_model'
    if not os.path.exists(bagging_save_path):
        os.mkdir(bagging_save_path)

    errorbar_save_path = 'DataAnalysis\\testfunc_correlation_errorbar_plot'
    if not os.path.exists(errorbar_save_path):
        os.mkdir(errorbar_save_path)

    start = datetime.datetime.now()
    for base_model in base_model_list:
        for testfunc_name in testfunc_name_list:
            X_all = testfunc_grid_generation(testfunc_name)
            y_all = testfunc_value(X_all, testfunc_name)
            testfunc_path = bagging_save_path + '\\' + testfunc_name + '_basemodel_' + base_model
            if not os.path.exists(testfunc_path):
                os.mkdir(testfunc_path)
            e_testfunc_path = errorbar_save_path + '\\' + testfunc_name + '_basemodel_' + base_model
            if not os.path.exists(e_testfunc_path):
                os.mkdir(e_testfunc_path)

            if testfunc_name == 'SinglePeakFunction':
                sample_nums = [25, 75, 125, 175,225,275]
            elif testfunc_name == 'SumSquaresFunction':
                sample_nums = [5,10,25,50,75,100]
            elif testfunc_name == 'PeaksFunction':
                sample_nums = [10, 25, 50, 75, 100, 125, 150, 175, 200]
            elif testfunc_name == 'SixHumpCamelFunction':
                sample_nums = [5,10,25,50,75,100]
            elif testfunc_name == 'ShubertFunction':
                sample_nums = [25,75,125,175,225,275,325,400]
            elif testfunc_name == 'Hartmann3Function':
                sample_nums = [5,10,25,50,75,100]

            mean_list = []
            std_list = []
            data_list = []  # 储存所有相关性数据，后续绘制小提琴图用
            for sample_num in sample_nums:
                samplenum_path = testfunc_path + '\\' + str(sample_num) + '_sample'
                if not os.path.exists(samplenum_path):
                    os.mkdir(samplenum_path)

                sample_num_all_data = []

                for i in range(50):
                    print(base_model,testfunc_name,sample_num,i)
                    X_train, y_train, sample_index = testfunc_initial_sample(X_all, testfunc_name, sample_num)

                    # BG模型训练
                    bg_mean, bg_std, r2_train_bagging, r2_all_bagging = bagging_fit_predict(X_train, y_train, X_all,y_all, sample_index, 30, testfunc_name, base_model, samplenum_path + '\\bagging' + str(i) + '.pkl')

                    # GP模型训练
                    _index = [i for i in range(sample_num)]  # 全样本训练
                    gp_mean, gp_std,_index, _kernel, _r2_train = base_model_training(_index, X_train, y_train, X_all, sample_index, testfunc_name, 'GP')
                    r2_train_gp = r2_score(y_all[sample_index], gp_mean[sample_index])
                    r2_all_gp = r2_score(y_all, gp_mean)
                    # 储存gp数据
                    joblib.dump((sample_index, gp_mean, gp_std, r2_train_gp, r2_all_gp), samplenum_path + '\\gp' + str(i) + '.pkl')


                    print('bg train r2:', r2_train_bagging)
                    print('gp train r2:', r2_train_gp)
                    print('bg all r2:', r2_all_bagging)
                    print('gp all r2:', r2_all_gp)


                    # 寻找统计量的最优参数
                    bw_method_list = [0.01, 0.1, 0.3, 0.5, 0.8, 1, 1.5]
                    peaks_gamma_list = [0.01, 0.1, 1, 10, 100, 1000]
                    p_c_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
                    all_gamma_list = [0.01, 0.1, 1, 10, 100, 1000]
                    a_c_list = [0, 0.2, 0.4, 0.6, 0.8, 1]

                    gp_std_cor = np.corrcoef(gp_std, bg_std)[0][1]

                    #寻找极值
                    peaks_index = find_peaks(X_train, y_all[sample_index])  # 寻找极值索引

                    kernel_density_estimation_cor_list = []
                    for bw_method in bw_method_list:
                        # 数据矩阵中存在线性相关的列，需要PCA降维才行
                        kde = gaussian_kde(X_train.T)  # 得把数据转化成[[x1...],[x2...]]这样才行，把训练样本点转置一下
                        kde.set_bandwidth(bw_method=bw_method)  # 设置峰宽
                        kernel_density_estimation = kde.evaluate(X_all.T)
                        gpstd_kde_cor = np.corrcoef(kernel_density_estimation, gp_std)[0][1]
                        kernel_density_estimation_cor_list.append(gpstd_kde_cor)
                    gpstd_kde_cor = min(kernel_density_estimation_cor_list)  # 负相关故取最小值
                    best_gpstd_bw_method = bw_method_list[np.argmin(np.array(kernel_density_estimation_cor_list))]
                    print('best_gpstd_bw_method:', best_gpstd_bw_method)

                    peaks_residual_influences_cor_list = []
                    for peaks_gamma in peaks_gamma_list:
                        peaks_residual_influences = residual_influences_gamma(X_train[peaks_index],
                                                                              y_all[sample_index][peaks_index],
                                                                              X_all,
                                                                              peaks_gamma)
                        peaks_residual_influences_cor = np.corrcoef(peaks_residual_influences, bg_std)[0][1]
                        peaks_residual_influences_cor_list.append(peaks_residual_influences_cor)
                    peaks_residual_influences_cor = max(peaks_residual_influences_cor_list)
                    best_peaks_gamma = peaks_gamma_list[np.argmax(np.array(peaks_residual_influences_cor_list))]
                    print('best_peaks_gamma:', best_peaks_gamma)

                    all_residual_influences_cor_list = []
                    for all_gamma in all_gamma_list:
                        all_residual_influences = residual_influences_gamma(X_train, y_all[sample_index],
                                                                            X_all, all_gamma)
                        all_residual_influences_cor = np.corrcoef(all_residual_influences, bg_std)[0][1]
                        all_residual_influences_cor_list.append(all_residual_influences_cor)
                    all_residual_influences_cor = max(all_residual_influences_cor_list)
                    best_all_gamma = all_gamma_list[np.argmax(np.array(all_residual_influences_cor_list))]
                    print('best_all_gamma:', best_all_gamma)

                    kernel_density_estimation_cor_list = []
                    for bw_method in bw_method_list:
                        # 数据矩阵中存在线性相关的列，需要PCA降维才行
                        kde = gaussian_kde(X_train.T)  # 得把数据转化成[[x1...],[x2...]]这样才行，把训练样本点转置一下
                        kde.set_bandwidth(bw_method=bw_method)  # 设置峰宽
                        kernel_density_estimation = kde.evaluate(X_all.T)
                        # print(kernel_density_estimation)
                        # print(bg_std)
                        kernel_density_estimation_cor = np.corrcoef(kernel_density_estimation, bg_std)[0][1]
                        kernel_density_estimation_cor_list.append(kernel_density_estimation_cor)
                    kernel_density_estimation_cor = min(kernel_density_estimation_cor_list)  # 负相关故取最小值
                    best_bw_method = bw_method_list[np.argmin(np.array(kernel_density_estimation_cor_list))]
                    print('best_bw_method:', best_bw_method)

                    p_combine_influences_cor_list = []
                    #  数量级相差太大，需要分别归一化再系数相加！！！
                    for p_c in p_c_list:
                        peaks_residual_influences = residual_influences_gamma(X_train[peaks_index],
                                                                              y_all[sample_index][peaks_index],
                                                                              X_all,
                                                                              best_peaks_gamma)
                        r_transfer = MinMaxScaler()
                        peaks_residual_influences = r_transfer.fit_transform(peaks_residual_influences.reshape(-1, 1))
                        # print(np.mean(peaks_residual_influences))

                        kde = gaussian_kde(X_train.T)  # 得把数据转化成[[x1...],[x2...]]这样才行，把训练样本点转置一下
                        kde.set_bandwidth(bw_method=best_bw_method)  # 设置峰宽
                        kernel_density_estimation = kde.evaluate(X_all.T)
                        k_transfer = MinMaxScaler()
                        kernel_density_estimation = k_transfer.fit_transform(kernel_density_estimation.reshape(-1, 1))
                        # print(np.mean(kernel_density_estimation))

                        p_combine_influences = (1 - p_c) * peaks_residual_influences - p_c * kernel_density_estimation
                        # print(np.mean(p_combine_influences))
                        p_combine_influences_cor = np.corrcoef(p_combine_influences.flatten(), bg_std)[0][1]
                        p_combine_influences_cor_list.append(p_combine_influences_cor)
                    p_combine_influences_cor = max(p_combine_influences_cor_list)
                    best_p_c = p_c_list[np.argmax(np.array(p_combine_influences_cor_list))]
                    print('best_p_c:', best_p_c)

                    a_combine_influences_cor_list = []
                    #  数量级相差太大，需要分别归一化再系数相加！！！
                    for a_c in a_c_list:
                        all_residual_influences = residual_influences_gamma(X_train,
                                                                            y_all[sample_index], X_all,
                                                                            best_all_gamma)
                        r_transfer = MinMaxScaler()
                        all_residual_influences = r_transfer.fit_transform(all_residual_influences.reshape(-1, 1))
                        # print(np.mean(all_residual_influences))

                        kde = gaussian_kde(X_train.T)  # 得把数据转化成[[x1...],[x2...]]这样才行，把训练样本点转置一下
                        kde.set_bandwidth(bw_method=best_bw_method)  # 设置峰宽
                        kernel_density_estimation = kde.evaluate(X_all.T)
                        k_transfer = MinMaxScaler()
                        kernel_density_estimation = k_transfer.fit_transform(kernel_density_estimation.reshape(-1, 1))
                        # print(np.mean(kernel_density_estimation))

                        a_combine_influences = (1 - a_c) * all_residual_influences - a_c * kernel_density_estimation
                        # print(np.mean(a_combine_influences))
                        a_combine_influences_cor = np.corrcoef(a_combine_influences.flatten(), bg_std)[0][1]
                        a_combine_influences_cor_list.append(a_combine_influences_cor)
                    a_combine_influences_cor = max(a_combine_influences_cor_list)
                    best_a_c = a_c_list[np.argmax(np.array(a_combine_influences_cor_list))]
                    print('best_a_c:', best_a_c)

                    joblib.dump(
                        (best_gpstd_bw_method, best_peaks_gamma, best_all_gamma, best_bw_method, best_p_c, best_a_c),
                        samplenum_path + '\\all_statistics_parameter' + str(i) + '.pkl')

                    sample_num_all_data.append(
                        [gp_std_cor, gpstd_kde_cor, peaks_residual_influences_cor,
                         all_residual_influences_cor, kernel_density_estimation_cor, p_combine_influences_cor,
                         a_combine_influences_cor])
                    print()
                # print(sample_num_all_data)
                sample_num_all_data = np.array(sample_num_all_data)
                data_list.append(sample_num_all_data)
                # print(all_y_predict_data.shape)
                sample_num_all_data_mean = np.mean(sample_num_all_data, axis=0)
                sample_num_all_data_std = np.std(sample_num_all_data, axis=0)
                # print(sample_num_all_data_mean)
                # print(sample_num_all_data_std)
                mean_list.append(sample_num_all_data_mean)
                std_list.append(sample_num_all_data_std)
            # print(mean_list)
            # print(std_list)
            mean_list = np.array(mean_list).T
            std_list = np.array(std_list).T
            errorbar_plot(mean_list, std_list, sample_nums, statistics_name_list, e_testfunc_path)
            joblib.dump((data_list, mean_list, std_list), e_testfunc_path + '\\all_correlation_data.pkl')



    end = datetime.datetime.now()
    print('totally time is ', end - start)
