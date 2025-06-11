import numpy as np
from scipy import stats


'''
准备工作
测试函数定义;网格生成;初始样本采样
'''


def ei(miu, sigma, cur_opt_value, opt_direction):
    if opt_direction == 'max':
        z = (miu - cur_opt_value) / sigma
        return (miu - cur_opt_value) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
    elif opt_direction == 'min':
        z = (cur_opt_value - miu) / sigma
        return (cur_opt_value - miu) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)

def ego(mean,std,cur_opt_value,opt_direction,sample_num=1):
    ei_value=ei(mean, std, cur_opt_value, opt_direction)
    recommend_index = np.argmax(ei_value)  # 返回的是最大值的索引
    return recommend_index, ei_value

def ego_without_repeat(mean,std,cur_opt_value,opt_direction,sample_index_list,sample_num=1):
    ei_value=ei(mean, std, cur_opt_value, opt_direction)
    ei_value_removed=[_ei if _index not in sample_index_list else -1 for _index,_ei in enumerate(ei_value) ] #索引顺序不变，将已有的元素变为-1
    recommend_index = np.argmax(ei_value_removed)

    return recommend_index, ei_value

def prediction(mean,opt_direction):
    if opt_direction=='max':
        recommend_index = np.argmax(mean)  # 返回的是最大值的索引
    else:
        recommend_index = np.argmin(mean)  # 返回的是最小值的索引
    return recommend_index

def prediction_without_repeat(mean,opt_direction,sample_index_list):
    if opt_direction=='max':
        mean_value_removed = [_m if _index not in sample_index_list else min(mean) for _index, _m in enumerate(mean)]  # 索引顺序不变，将已有的元素变为最小值
        recommend_index = np.argmax(mean_value_removed)  # 返回的是最大值的索引
    else:
        mean_value_removed = [_m if _index not in sample_index_list else max(mean) for _index, _m in enumerate(mean)]  # 索引顺序不变，将已有的元素变为最大值
        recommend_index = np.argmin(mean_value_removed)  # 返回的是最小值的索引
    return recommend_index

