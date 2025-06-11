from data_generation import dataset_read, dataset_initial_sample
from sampling_strategy import ego_without_repeat
import copy
import numpy as np
import os
import joblib
from model_code import base_model_training, bagging_fit_predict
from sklearn.metrics import r2_score
import material_dataset
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import datetime
    import random
    random.seed(1)

    dataset_opt_direction_list = ['DiffusionActivationEnergies_max', 'EnergyStorageDensity_max', 'PerovskiteStability_max']
    # dataset_opt_direction_list = ['PerovskiteStabilityReduced_max']
    initial_sample_percent_list = [0.05, 0.1, 0.15]
    root_path = 'RunData\\dataset_iteration_sample_OC_Curve_percent_controlBGSVR'
    repeat_num = 20

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    start = datetime.datetime.now()
    for initial_sample_percent in initial_sample_percent_list:
        for dataset_opt_direction in dataset_opt_direction_list:
            dataset_name, opt_direction = dataset_opt_direction.split('_')
            X_all, y_all = dataset_read(dataset_name)
            initial_sample_num = int(initial_sample_percent * len(y_all))
            dataset_path = root_path + '\\' + dataset_name + '_' + opt_direction + '_initial_sample_percent' + str(
                initial_sample_percent) + ' ' + str(initial_sample_num)
            if not os.path.exists(dataset_path):
                os.mkdir(dataset_path)
            for rn in range(repeat_num):
                repeat_path = dataset_path + '\\repeat_num' + str(rn)
                if not os.path.exists(repeat_path):
                    os.mkdir(repeat_path)
                _X_train, _y_train, initial_sample_index = dataset_initial_sample(X_all, y_all, initial_sample_num)  # 取百分比的样本数量
                bg_ei_sample_index_list = copy.deepcopy(initial_sample_index)
                gp_ei_sample_index_list = copy.deepcopy(initial_sample_index)

                bg_ei_sample_value_list = []  # 装推荐样本的[预测值，不确定性，EI值]
                gp_ei_sample_value_list = []  # 装推荐样本的[预测值，不确定性，EI值]

                # 观察模型的精度随迭代过程的变化  1 与真实测试函数的差距 2 与训练集的拟合精度
                bg_ei_r2_list = []  # 装[bg_ei_real_r2,bg_ei_train_r2,svrs_r2_mean,svr_r2_std] svr误差为与bagging所有样本间的误差，而非与svr抽样样本间的误差
                gp_ei_r2_list = []  # 装[gp_ei_real_r2,gp_ei_train_r2]

                for i in range(100):
                    print(dataset_name, initial_sample_percent, initial_sample_num, opt_direction)
                    print(rn, i)

                    bg_ei_X_train = X_all[bg_ei_sample_index_list]
                    bg_ei_y_train = y_all[bg_ei_sample_index_list]

                    gp_ei_X_train = X_all[gp_ei_sample_index_list]
                    gp_ei_y_train = y_all[gp_ei_sample_index_list]

                    if opt_direction == 'max':
                        bg_ei_cur_opt_value = np.max(bg_ei_y_train)
                        gp_ei_cur_opt_value = np.max(gp_ei_y_train)
                    elif opt_direction == 'min':
                        bg_ei_cur_opt_value = np.min(bg_ei_y_train)
                        gp_ei_cur_opt_value = np.min(gp_ei_y_train)

                    # BGEI:U BGsvr + P BGsvr
                    bg_ei_mean, bg_ei_std, bg_ei_train_r2, bg_ei_real_r2 = bagging_fit_predict(bg_ei_X_train,
                                                                                               bg_ei_y_train, X_all,
                                                                                               y_all,
                                                                                               bg_ei_sample_index_list,
                                                                                               30, dataset_name,
                                                                                               'SVR')  # 用SVR作为BG基模型
                    bg_ei_recommend_index, bg_ei_value = ego_without_repeat(bg_ei_mean, bg_ei_std,
                                                                            bg_ei_cur_opt_value,
                                                                            opt_direction,
                                                                            bg_ei_sample_index_list)  # 控制变量实验，都用GP单模型，仅不确定性不同
                    bg_ei_sample_index_list.append(bg_ei_recommend_index)
                    bg_ei_sample_value_list.append(
                        [bg_ei_mean[bg_ei_recommend_index], bg_ei_std[bg_ei_recommend_index],
                         bg_ei_value[bg_ei_recommend_index]])
                    bg_ei_r2_list.append([bg_ei_real_r2, bg_ei_train_r2])

                    # BGEI-UGP:U GP + P BGsvr
                    bg_ei_mean, bg_ei_std, bg_ei_train_r2, bg_ei_real_r2 = bagging_fit_predict(gp_ei_X_train,
                                                                                               gp_ei_y_train, X_all,
                                                                                               y_all,
                                                                                               gp_ei_sample_index_list,
                                                                                               30, dataset_name,
                                                                                               'SVR')  # 用SVR作为BG基模型

                    _index = [i for i in range(len(gp_ei_sample_index_list))]  # 全样本训练
                    gp_ei_mean, gp_ei_std, _index, _kernel, _r2_train = base_model_training(_index, gp_ei_X_train,
                                                                                            gp_ei_y_train, X_all,
                                                                                            gp_ei_sample_index_list,
                                                                                            dataset_name, 'GP')
                    gp_ei_recommend_index, gp_ei_value = ego_without_repeat(bg_ei_mean, gp_ei_std,
                                                                            gp_ei_cur_opt_value,
                                                                            opt_direction, gp_ei_sample_index_list)

                    if i==0:
                        joblib.dump((initial_sample_num, initial_sample_index, gp_ei_std),
                                    repeat_path + '\\initial_sample_info.pkl')

                    gp_ei_sample_index_list.append(gp_ei_recommend_index)
                    gp_ei_sample_value_list.append( [bg_ei_mean[gp_ei_recommend_index], gp_ei_std[gp_ei_recommend_index], gp_ei_value[gp_ei_recommend_index]])
                    gp_ei_r2_list.append([bg_ei_real_r2, bg_ei_train_r2])

                    print(bg_ei_sample_index_list)
                    print(gp_ei_sample_index_list)

                    # print(bg_ei_sample_index_list)
                    # print(bg_mean_sample_index_list)
                    # print(gp_ei_sample_index_list)
                    # print(gp_mean_sample_index_list)
                    print()

                # 储存运行数据
                joblib.dump((bg_ei_sample_index_list, gp_ei_sample_index_list, bg_ei_sample_value_list, gp_ei_sample_value_list, bg_ei_r2_list, gp_ei_r2_list), repeat_path + '\\sample_data.pkl')

        end = datetime.datetime.now()
        print('totally time is ', end - start)
