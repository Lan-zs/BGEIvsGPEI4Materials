import random
import numpy as np
from multiprocessing import Pool
from functools import partial
from sklearn.svm import SVR
import joblib
import test_function
import material_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from data_generation import testfunc_grid_generation, testfunc_initial_sample,testfunc_value
from sklearn.gaussian_process import GaussianProcessRegressor
import warnings
warnings.filterwarnings('ignore')


# 并行bagging的基模型
def base_model_training(_index, X_train, y_train, X_all, sample_index, dataset_or_testfunc_name, base_model):
    _X_train = X_train[list(_index)]
    _y_train = y_train[list(_index)]
    if dataset_or_testfunc_name in ['CurieTemperature', 'DiffusionActivationEnergies', 'EnergyStorageDensity','PerovskiteStability','PerovskiteStabilityReduced']:
        dataset_or_testfunc = getattr(material_dataset, dataset_or_testfunc_name)()  # 取出测试函数类并实例化
    else:
        dataset_or_testfunc = getattr(test_function, dataset_or_testfunc_name)()  # 取出测试函数类并实例化

    if base_model == 'SVR':
        # 训练模型 标准化数据
        _X_StdScaler = StandardScaler()
        _y_StdScaler = StandardScaler()
        _X_train_nor = _X_StdScaler.fit_transform(_X_train)
        _y_train_nor = _y_StdScaler.fit_transform(_y_train.reshape(-1, 1))  # 要求_y_train的shape必须是(-1, 1)
        X_all_nor = _X_StdScaler.transform(X_all)
        # print(_X_train_nor)
        # print(_y_train_nor)
        # print(_X_train_nor.shape)
        # print(_y_train_nor.shape)
        # 定义SVR模型作为基模型
        parameters = dataset_or_testfunc.parameters
        model = SVR()
        clf = GridSearchCV(model, parameters, cv=5, scoring='neg_mean_absolute_error')
        clf.fit(_X_train_nor, _y_train_nor.ravel())  # 要求_y_train_nor的shape必须是(-1,)
        # print(clf.best_params_)
        # 预测网格点
        y_pre = clf.predict(X_all_nor)
        y_pre = _y_StdScaler.inverse_transform(y_pre.reshape(-1, 1))  # 要求y_pre的shape必须是(-1, 1)
        # SVR R2
        r2_train = r2_score(y_train, y_pre[sample_index])
        return y_pre.flatten(), _index, clf.best_params_,r2_train

    elif base_model == 'GP':
        gp_kernel = dataset_or_testfunc.gp_kernel
        model = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=5,normalize_y=True)
        # model = GaussianProcessRegressor()
        model.fit(_X_train, _y_train)
        # print(model.kernel_)
        y_pre, y_std = model.predict(X_all, return_std=True)
        r2_train = r2_score(y_train, y_pre[sample_index])
        return y_pre.flatten(), y_std.flatten(),_index, model.kernel_, r2_train


# 并行bagging模型训练，输出模型预测数据，控制输出预测与bootstrap方差,不固定参数
def bagging_fit_predict(X_train, y_train, X_all, y_all, sample_index, bagging_model_num, dataset_or_testfunc_name, base_model, file_path=None):
    # bootstrap：从中抽取给定数量的训练集数据索引
    data_partition = [random.sample(range(y_train.shape[0]), int(0.63 * y_train.shape[0])) for i in
                      range(bagging_model_num)]

    # p.map 传入多个参数
    func = partial(base_model_training, X_train=X_train, y_train=y_train, X_all=X_all, sample_index=sample_index, dataset_or_testfunc_name=dataset_or_testfunc_name, base_model=base_model)
    with Pool() as p:
        all_basemodel_data = p.map(func, data_partition)

    all_y_predict_data = []
    # r2_svrs = []
    # print(all_basemodel_data)
    for d in all_basemodel_data:
        all_y_predict_data.append(d[0])
        # r2_svrs.append(d[3])
    # print(all_y_predict_data)
    # print(r2_svrs)

    all_y_predict_data = np.array(all_y_predict_data).T
    # print(all_y_predict_data.shape)
    all_y_predict_data_mean = np.mean(all_y_predict_data, axis=1)
    # print(all_y_predict_data_mean.shape)
    all_y_predict_data_std = np.std(all_y_predict_data, axis=1)  # 标准差
    # print(all_y_predict_data_std.shape)

    # R2
    r2_train_bagging = r2_score(y_train, all_y_predict_data_mean[sample_index])
    # print('bagging R2: ' + str(r2_bagging))
    r2_all_bagging = r2_score(y_all, all_y_predict_data_mean)

    if file_path!=None:
        # 储存bagging数据
        # joblib.dump((sample_index, all_basemodel_data, all_y_predict_data_mean, all_y_predict_data_std, r2_train_bagging,r2_all_bagging), file_path)
        joblib.dump((sample_index, all_y_predict_data_mean, all_y_predict_data_std, r2_train_bagging, r2_all_bagging), file_path)

    # 返回计算数据、R2数据
    return all_y_predict_data_mean, all_y_predict_data_std, r2_train_bagging, r2_all_bagging


if __name__ == '__main__':
    import datetime
    from visualization_code import model_surface_plot
    testfunction_name = 'SumSquaresFunction'
    # testfunction_name = 'PeaksFunction'
    test_func = getattr(test_function, testfunction_name)()  # 取出测试函数类并实例化

    grid_points = testfunc_grid_generation(testfunction_name)

    ranges = [np.arange(-2, 2 +0.1, step=0.1) for n in range(test_func.num_variables)]  # SumSquaresFunction
    grid = np.meshgrid(*ranges)  # 生成网格
    grid = np.stack(grid, axis=-1)
    grid_points_L = grid.reshape(-1, grid.shape[-1]) #将网格展开成数据条

    # r2=0
    for i in range(10):
        sample_points, y, sample_index = testfunc_initial_sample(grid_points, testfunction_name, 5)
        start1 = datetime.datetime.now()
        # bg_mean1, bg_std1, r2_train_bagging, r2_all_bagging = bagging_fit_predict(sample_points, y, grid_points,y_all, sample_index,  30, testfunction_name,'GP')

        _index = [i for i in range(len(sample_index))]
        bg_mean1, bg_std1,_index, _kernel, r2_train=base_model_training(_index, sample_points, y, grid_points_L, sample_index, testfunction_name, 'GP')
        # bg_mean1, _index, best_params_,r2_train=base_model_training(_index, sample_points, y, grid_points_L, sample_index, testfunction_name, 'SVR')
        # print(r2_train_bagging, r2_all_bagging)
        print(np.mean(bg_mean1))
        # print(np.mean(bg_std1))
        end1 = datetime.datetime.now()
        print('totally 1time is ', end1 - start1)
        model_surface_plot(sample_points, y, bg_mean1, testfunction_name, 0.1)
        model_surface_plot(sample_points, y, bg_std1, testfunction_name, 0.1)
    # print(r2/30)
