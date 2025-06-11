import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
import seaborn as sns
from data_generation import dataset_read
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from codes.data_generation import testfunc_grid_generation, testfunc_initial_sample, testfunc_value
from codes.data_analysis import find_localmin_peaks, find_localmax_peaks
from codes.data_analysis import find_peaks, residual_influences, calculate_cooks_distance
from sklearn.preprocessing import StandardScaler


def scatter_guideline_dict_plot(all_guideline_dict, all_delta_oc_dict,all_guideline,all_difference, save_path):
    scatter_marker=['o','^','v','s']
    dataset_name=['DAE Dataset','ESD Dataset','PSR Dataset','PS Dataset']

    scatter_type = ['bg' if d > 0 else 'gp' for d in all_difference]
    bg_points = [all_guideline[i] for i in range(len(scatter_type)) if scatter_type[i] == 'bg']
    bg_points_value = [abs(all_difference[i]) for i in range(len(scatter_type)) if scatter_type[i] == 'bg']
    gp_points = [all_guideline[i] for i in range(len(scatter_type)) if scatter_type[i] == 'gp']
    gp_points_value = [abs(all_difference[i]) for i in range(len(scatter_type)) if scatter_type[i] == 'gp']

    fig, axs = plt.subplots(2, 1, height_ratios=[1, 2], sharex=True, figsize=(8, 8))

    fig.subplots_adjust(hspace=0)
    ax1 = axs[0]
    ax2 = axs[1]

    # 使用Seaborn绘制概率密度图
    sns.kdeplot(np.array(bg_points), fill=True, alpha=0.3, label='bg', common_norm=True, ax=ax1)
    sns.kdeplot(np.array(gp_points), fill=True, alpha=0.3, label='gp', common_norm=True, ax=ax1)


    #不同数据集使用不同形状
    for _key,_marker,_name in zip(all_guideline_dict.keys(),scatter_marker,dataset_name):
        all_delta_oc_dict[_key] = [all_delta_oc_dict[_key][i] for i, d in enumerate(all_delta_oc_dict[_key]) if d != 0]
        all_guideline_dict[_key] = [all_guideline_dict[_key][i] for i, d in enumerate(all_delta_oc_dict[_key]) if d != 0]

        scatter_type = ['bg' if d > 0 else 'gp' for d in all_delta_oc_dict[_key]]
        bg_points = [all_guideline_dict[_key][i] for i in range(len(scatter_type)) if scatter_type[i] == 'bg']
        bg_points_value = [abs(all_delta_oc_dict[_key][i]) for i in range(len(scatter_type)) if scatter_type[i] == 'bg']
        gp_points = [all_guideline_dict[_key][i] for i in range(len(scatter_type)) if scatter_type[i] == 'gp']
        gp_points_value = [abs(all_delta_oc_dict[_key][i]) for i in range(len(scatter_type)) if scatter_type[i] == 'gp']

        ax2.scatter(bg_points, bg_points_value, color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),marker=_marker , alpha=0.6, s=60, edgecolors='grey',label=_name+' (BGEI Better)')
        ax2.scatter(gp_points, gp_points_value, color=(1.0, 0.4980392156862745, 0.054901960784313725),marker=_marker ,alpha=0.6, s=60, edgecolors='grey',label=_name+' (GPEI Better)')

    ax1.set_ylabel('Density',fontsize=15)
    ax2.set_ylabel('OC Difference',fontsize=15)
    ax2.set_ylim(0, 0.13)
    # ax2.set_xlabel('Feasibility of Exploration',fontsize=15)
    ax2.set_xlabel('${FE}$',fontsize=15)

    # 自动对齐y轴标签
    fig.align_labels()

    # 设置刻度标签的字体大小
    # plt.tick_params(axis='both', which='major', labelsize=10)


    # ax1.axvline(x=0.5, color='k', linestyle='dashed', alpha=0.3, linewidth=2)
    # ax2.axvline(x=0.5, color='k', linestyle='dashed', alpha=0.3, linewidth=2)
    # ax1.axvline(x=0.3, color='k', linestyle='dashed', alpha=0.3, linewidth=2)
    # ax2.axvline(x=0.3, color='k', linestyle='dashed', alpha=0.3, linewidth=2)
    ax1.axvline(x=0.4, color='k', linestyle='dashed', alpha=0.3, linewidth=2)
    ax2.axvline(x=0.4, color='k', linestyle='dashed', alpha=0.3, linewidth=2)
    plt.xlim(0, 3)


    # 添加图例到图表
    plt.legend(loc='upper right', ncol=1,fontsize=12)
    plt.savefig(save_path + '\\all_datasetnew.jpg', dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close()

if __name__ == '__main__':
    root_path = 'RunData\\dataset_iteration_sample_OC_Curve_percent_controlBGSVR'
    root_save_path = 'DataAnalysis\\dataset_iteration_sample_OC_Curve_percent_controlBGSVR_guideline'

    if not os.path.exists(root_save_path):
        os.mkdir(root_save_path)
    # print(os.listdir(root_path))

    iteration_budget_num_list=[10,20,30,40,50,60,70,80,90]

    # 指标数据
    all_guideline_list = []
    all_guideline_dict = {}

    all_bg_budget_oc_list = []
    all_gp_budget_oc_list = []

    all_bg_budget_oc_dict = {}
    all_gp_budget_oc_dict = {}




    # for dataset_opt_direction in os.listdir(root_path):
    #     # dataset_save_path = root_save_path + '\\' + dataset_opt_direction
    #     # if not os.path.exists(dataset_save_path):
    #     #     os.mkdir(dataset_save_path)
    #     # 提取初始样本数
    #     initial_sample_num = eval(dataset_opt_direction.split(' ')[1])
    #     dataset_opt_direction_path = os.path.join(root_path, dataset_opt_direction)
    #     dataset_name = dataset_opt_direction.split('_')[0]
    #     opt_direction = dataset_opt_direction.split('_')[1]
    #
    #     X_all, y_all = dataset_read(dataset_name)
    #
    #     if opt_direction == 'max':
    #         global_opt = max(y_all)
    #         global_worst = min(y_all)
    #     else:
    #         global_opt = min(y_all)
    #         global_worst = max(y_all)
    #
    #     print(initial_sample_num)
    #     print(dataset_name)
    #     print(opt_direction)
    #
    #     # 指标数据
    #     guideline_list=[]
    #
    #
    #     bg_budget_oc_list=[]
    #     gp_budget_oc_list=[]
    #
    #     # 误差棒数据
    #     bg_ei_all_data = []
    #     gp_ei_all_data = []
    #     for repeat_num in os.listdir(dataset_opt_direction_path):
    #         repeat_num_path = os.path.join(dataset_opt_direction_path, repeat_num)
    #
    #         bg_ei_sample_index_list, gp_ei_sample_index_list, bg_ei_sample_value_list, gp_ei_sample_value_list, bg_ei_r2_list, gp_ei_r2_list = joblib.load(repeat_num_path + '\\sample_data.pkl')
    #         print(repeat_num_path)
    #
    #         initial_sample_num, initial_sample_index, gp_ei_std=joblib.load(repeat_num_path + '\\initial_sample_info.pkl')
    #         # predictable_sample_num = len([g for g in gp_ei_std if g-min(gp_ei_std) < 0.2*(max(gp_ei_std)-min(gp_ei_std))])
    #         # predictable_sample_num = len([g for g in gp_ei_std if g-min(gp_ei_std) < 0.2*(max(gp_ei_std)-min(gp_ei_std))])-initial_sample_num #未知样本中可以被预测的
    #         # 不确定性小于均值的认为是可被预测的样本
    #         predictable_sample_num = len([g for g in gp_ei_std if g < np.mean(gp_ei_std)])-initial_sample_num #未知样本中可以被预测的
    #
    #         print(predictable_sample_num)
    #         print(max(gp_ei_std))
    #         print(min(gp_ei_std))
    #         print(np.mean(gp_ei_std))
    #
    #         unknown_sample_num=len(gp_ei_std)-initial_sample_num  #目前还剩的未知的样本数量
    #
    #         # 计算指标 多次采样统计
    #         g_list=[]
    #         for iteration_budget_num in iteration_budget_num_list:
    #             guideline = (iteration_budget_num * predictable_sample_num) / (initial_sample_num * unknown_sample_num)
    #             g_list.append(guideline)
    #         guideline_list.append(g_list)
    #
    #
    #         # 本次重复的初始采样序列
    #         initial_sample_index = bg_ei_sample_index_list[:initial_sample_num]
    #         # print(initial_sample_index)
    #         # print(bg_sample_index_list)
    #         # print(gp_sample_index_list)
    #
    #         bg_ei_current_opt_list = []
    #         gp_ei_current_opt_list = []
    #         for i in range(initial_sample_num, len(bg_ei_sample_index_list)):
    #             bg_ei_current_sample_value = y_all[bg_ei_sample_index_list[:i]]
    #             gp_ei_current_sample_value = y_all[gp_ei_sample_index_list[:i]]
    #             if opt_direction == 'max':
    #                 bg_ei_current_opt = max(bg_ei_current_sample_value)
    #                 gp_ei_current_opt = max(gp_ei_current_sample_value)
    #                 bg_ei_current_opt_list.append(bg_ei_current_opt)
    #                 gp_ei_current_opt_list.append(gp_ei_current_opt)
    #             else:
    #                 bg_ei_current_opt = min(bg_ei_current_sample_value)
    #                 gp_ei_current_opt = min(gp_ei_current_sample_value)
    #                 bg_ei_current_opt_list.append(bg_ei_current_opt)
    #                 gp_ei_current_opt_list.append(gp_ei_current_opt)
    #         # print(bg_current_opt_list)
    #         # print(gp_current_opt_list)
    #         # print(gp_mean_current_opt_list)
    #
    #         iteration_num = [i for i in range(len(bg_ei_current_opt_list))]
    #         print(opt_direction, global_opt)
    #
    #         # 处理，计算OC值
    #         if opt_direction == 'max':
    #             bg_ei_oc_list = abs((bg_ei_current_opt_list - global_opt) / (global_opt - global_worst))
    #             gp_ei_oc_list = abs((gp_ei_current_opt_list - global_opt) / (global_opt - global_worst))
    #         else:
    #             bg_ei_oc_list = abs((bg_ei_current_opt_list - global_opt) / (global_opt - global_worst))
    #             gp_ei_oc_list = abs((gp_ei_current_opt_list - global_opt) / (global_opt - global_worst))
    #
    #         # 误差棒数据
    #         bg_ei_all_data.append(bg_ei_oc_list)
    #         gp_ei_all_data.append(gp_ei_oc_list)
    #
    #     bg_ei_all_data = np.array(bg_ei_all_data)
    #     bg_ei_mean_data = np.mean(bg_ei_all_data, axis=0)
    #     bg_ei_std_data = np.std(bg_ei_all_data, axis=0, ddof=1) / np.sqrt(bg_ei_all_data.shape[0])
    #
    #     gp_ei_all_data = np.array(gp_ei_all_data)
    #     gp_ei_mean_data = np.mean(gp_ei_all_data, axis=0)
    #     gp_ei_std_data = np.std(gp_ei_all_data, axis=0, ddof=1) / np.sqrt(gp_ei_all_data.shape[0])
    #
    #
    #     guideline_list = np.array(guideline_list)
    #     guideline_mean_data = np.mean(guideline_list, axis=0)
    #     for gm in guideline_mean_data:
    #         all_guideline_list.append(gm)
    #
    #         _guideline_list = all_guideline_dict.get(dataset_name, [])
    #         _guideline_list.append(gm)
    #         all_guideline_dict[dataset_name] = _guideline_list
    #
    #
    #     print(guideline_mean_data.shape)
    #
    #
    #
    #     for iteration_budget_num in iteration_budget_num_list:
    #         _bg_dict_list=all_bg_budget_oc_dict.get(dataset_name,[])
    #         _bg_dict_list.append(bg_ei_mean_data[iteration_budget_num])
    #         all_bg_budget_oc_dict[dataset_name]=_bg_dict_list
    #
    #         _gp_dict_list = all_gp_budget_oc_dict.get(dataset_name, [])
    #         _gp_dict_list.append(gp_ei_mean_data[iteration_budget_num])
    #         all_gp_budget_oc_dict[dataset_name] = _gp_dict_list
    #
    #         bg_budget_oc_list.append(bg_ei_mean_data[iteration_budget_num])
    #         all_bg_budget_oc_list.append(bg_ei_mean_data[iteration_budget_num])
    #         gp_budget_oc_list.append(gp_ei_mean_data[iteration_budget_num])
    #         all_gp_budget_oc_list.append(gp_ei_mean_data[iteration_budget_num])
    #
    #     delta_oc_list=[gp_budget_oc_list[i]-bg_budget_oc_list[i] for i in range(len(bg_budget_oc_list))]
    #     # print([d for d in delta_oc_list if d != 0])
    #     bg_better = len([d for d in delta_oc_list if d < 0])
    #     print('GP better:', bg_better)
    #     gp_better = len([d for d in delta_oc_list if d > 0])
    #     print('BG better:', gp_better)
    #     no_better = len([d for d in delta_oc_list if d == 0])
    #     print('No better:', no_better)
    #     print()
    #
    #     # delta_oc_list = [d for d in delta_oc_list if d != 0]
    #
    #     # print(guideline_list)
    #     # kde_guideline_plot([guideline_mean_data[i] for i, d in enumerate(delta_oc_list) if d != 0],
    #     #                    [delta_oc_list[i] for i, d in enumerate(delta_oc_list) if d != 0],
    #     #                    dataset_opt_direction +  ' Guideline feasibility_of_exploration',
    #     #                    root_save_path)
    #
    #
    #     # scatter_guideline_plot([guideline_mean_data[i] for i, d in enumerate(delta_oc_list) if d != 0],
    #     #                        [delta_oc_list[i] for i, d in enumerate(delta_oc_list) if d != 0],
    #     #                        dataset_opt_direction +  ' Guideline feasibility_of_exploration', root_save_path)
    #
    #
    #
    #
    # all_delta_oc_list = [all_gp_budget_oc_list[i] - all_bg_budget_oc_list[i] for i in range(len(all_bg_budget_oc_list))]
    # # print([d for d in all_delta_oc_list if d != 0])
    # bg_better = len([d for d in all_delta_oc_list if d < 0])
    # print('GP better:', bg_better)
    # gp_better = len([d for d in all_delta_oc_list if d > 0])
    # print('BG better:', gp_better)
    # no_better = len([d for d in all_delta_oc_list if d == 0])
    # print('No better:', no_better)
    # print()
    #
    # all_delta_oc_dict={}
    # for key_ in all_gp_budget_oc_dict.keys():
    #     all_delta_oc_dict[key_]=[all_gp_budget_oc_dict[key_][i] - all_bg_budget_oc_dict[key_][i] for i in range(len(all_bg_budget_oc_dict[key_]))]
    #
    # for _key in all_guideline_dict.keys():
    #     print(len(all_guideline_dict[_key]))
    #     print(len(all_delta_oc_dict[_key]))
    # print(all_guideline_dict)
    # print(all_delta_oc_dict)
    #
    # # # 相差大于归一化OC的5%才计入 不太科学，给小数据集搞没了
    # # all_delta_oc_list=[d for d in all_delta_oc_list if abs(d) > 0.05]
    # # all_delta_oc_list=[d for d in all_delta_oc_list if d!=0]
    #
    #
    # # scatter_guideline_plot([all_guideline_list[i] for i, d in enumerate(all_delta_oc_list) if d != 0],
    # #                    [all_delta_oc_list[i] for i, d in enumerate(all_delta_oc_list) if d != 0],
    # #                    'All Guideline feasibility_of_exploration', root_save_path)
    #
    # all_guideline=[all_guideline_list[i] for i, d in enumerate(all_delta_oc_list) if d != 0]
    # all_difference=[all_delta_oc_list[i] for i, d in enumerate(all_delta_oc_list) if d != 0]
    #
    # for _key in all_guideline_dict.keys():
    #     all_delta_oc_dict[_key]=[all_delta_oc_dict[_key][i] for i, d in enumerate(all_delta_oc_dict[_key]) if d != 0]
    #     all_guideline_dict[_key]=[all_guideline_dict[_key][i] for i, d in enumerate(all_delta_oc_dict[_key]) if d != 0]
    #
    # for _key in all_guideline_dict.keys():
    #     print(len(all_guideline_dict[_key]))
    #     print(len(all_delta_oc_dict[_key]))
    # # print(all_guideline_dict)
    # # print(all_delta_oc_dict)

    # joblib.dump((all_guideline_dict, all_delta_oc_dict,all_guideline,all_difference),'guideline_different_dataset0701 mean.pkl')

    all_guideline_dict, all_delta_oc_dict, all_guideline, all_difference=joblib.load('guideline_different_dataset0701 mean.pkl')

    scatter_guideline_dict_plot(all_guideline_dict, all_delta_oc_dict,all_guideline,all_difference, root_save_path)


