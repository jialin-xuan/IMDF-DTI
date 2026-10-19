'''
Description: Prepare Data for main process
CopyRight 2022 by MrZQAQ. All rights reserved.
'''

import numpy as np

def get_kfold_data(i, datasets, k=5):

    fold_size = len(datasets) // k

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]
        trainset = datasets[0:val_start]

    return trainset, validset

def shuffle_dataset(dataset, seed):
    '''
    Description: 随机打乱数据集，确保不同随机种子下的实验结果可复现
    Args:
        dataset: 待打乱的数据集（列表或数组）
        seed: 随机种子（整数）
    Returns:
        dataset: 打乱后的数据集（列表或数组）
    '''
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
