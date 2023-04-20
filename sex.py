import random

from sklearn import preprocessing
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from w2v import *


# 性别对准确度
def sex_accuracy(w2v_model, judge_he, judge_she):
    he = []
    she = []
    with open("sex_he.txt", encoding='utf-8') as f_h:
        for i in f_h:
            i = i.replace('\n', '')
            he.append(i)

    with open("sex_she.txt", encoding='utf-8') as f_sh:
        for i in f_sh:
            i = i.replace('\n', '')
            she.append(i)

    number = 0
    not_judge = []

    for i in tqdm(he):
        similar_he = calculate_words_similar(w2v_model, i, judge_he)
        similar_she = calculate_words_similar(w2v_model, i, judge_she)
        if similar_he > similar_she:
            number += 1
        else:
            not_judge.append(i)

    for i in tqdm(she):
        similar_he = calculate_words_similar(w2v_model, i, judge_he)
        similar_she = calculate_words_similar(w2v_model, i, judge_she)
        if similar_she > similar_he:
            number += 1
        else:
            not_judge.append(i)

    print(judge_he + "/" + judge_she)
    print("判断正确的数量为:" + str(number))
    print("百分比为:" + str(number / (len(he) + len(she))))
    print("判断错误的词包括:")
    print(not_judge)


def PCA_(w2v_model):
    he = ["男子", "男", "父亲", "男性", "儿子", "丈夫", "男士", "男人", "爸爸", "男孩"]
    she = ["女子", "女", "母亲", "女性", "女儿", "妻子", "女士", "女人", "妈妈", "女孩"]
    x = []
    for i in range(10):
        x.append(w2v_model.wv[he[i]] - w2v_model.wv[she[i]])
    X = np.array(x)

    pca = PCA(n_components=2)
    print(pca)
    # 应用于训练集数据进行PCA降维
    pca.fit(X)
    # 用X来训练PCA模型，同时返回降维后的数据
    # newX = pca.fit_transform(X)
    # print(newX)
    # 将降维后的数据转换成原始数据，
    # pca_new = pca.transform(X)
    # print(pca_new.shape)
    # 输出具有最大方差的成分
    print(pca.components_)
    # 输出所保留的n个成分各自的方差百分比
    print(pca.explained_variance_ratio_)
    # 输出所保留的n个成分各自的方差
    print(pca.explained_variance_)
    # 输出未处理的特征维数
    print(pca.n_features_)
    # 输出训练集的样本数量
    print(pca.n_samples_)
    # 输出协方差矩阵
    print(pca.noise_variance_)
    # 每个特征的奇异值
    print(pca.singular_values_)
    # 用生成模型计算数据精度矩阵
    print(pca.get_precision())

    # 计算生成特征系数矩阵
    covX = np.around(np.corrcoef(X.T), decimals=3)
    # 输出特征系数矩阵
    print(covX)
    # 求解协方差矩阵的特征值和特征向量
    featValue, featVec = np.linalg.eig(covX)
    # 将特征进行降序排序
    featValue = sorted(featValue)[::-1]

    sum_featValue = np.sum(featValue)
    for i in range(len(featValue)):
        featValue[i] = featValue[i] / sum_featValue
    print(featValue)

    # 图像绘制
    # 同样的数据绘制散点图和折线图
    plt.scatter(range(1, X.shape[1] + 1), featValue)
    plt.plot(range(1, X.shape[1] + 1), featValue)

    # 显示图的标题
    plt.title("Test Plot")
    # xy轴的名字
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")
    # 显示网格
    plt.grid()
    # 显示图形
    plt.show()


def PCA_average(w2v_model):
    percentage = []
    for i in range(300):
        percentage.append(0)
    X_ = []
    for i in tqdm(range(1000)):
        a = random.sample(list(w2v_model.wv.key_to_index), 100)
        a_1 = a[:50]
        a_2 = a[50:]

        x = []
        for j in range(50):
            x.append(w2v_model.wv[a_1[j]] - w2v_model.wv[a_2[j]])
        X = np.array(x)
        pca = PCA(n_components=2)
        pca.fit(X)
        covX = np.around(np.corrcoef(X.T), decimals=3)
        featValue, featVec = np.linalg.eig(covX)
        featValue = sorted(featValue)[::-1]
        sum_featValue = np.sum(featValue)
        # print(featValue)
        for j in range(len(featValue)):
            b = featValue[j] / sum_featValue
            percentage[j] += b
        # print(percentage)
        X_ = X

    for i in range(300):
        percentage[i] /= 1000

    print(percentage)

    # 图像绘制
    # 同样的数据绘制散点图和折线图
    plt.scatter(range(1, X_.shape[1] + 1), percentage)
    plt.plot(range(1, X_.shape[1] + 1), percentage)

    # 显示图的标题
    plt.title("Test Plot")
    # xy轴的名字
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")
    # 显示网格
    plt.grid()
    # 显示图形
    plt.show()
