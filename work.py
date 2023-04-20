from w2v import *
from glove import *
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import scipy.stats as stats


def calculate_spearman_correlation(X, Y):
    return stats.spearmanr(X, Y)[0]
def calculate_spearman_correlation_p(X, Y):
    return stats.spearmanr(X, Y)[1]


def delete_repeat(str):
    s = []
    for i in str:
        if i not in s:
            s.append(i)
    return s

def format_work(txt):
    a=[]
    with open(txt,encoding='utf-8') as f:
        for i in f:
            a=i.split(" ")
    a = delete_repeat(a)
    for j in a:
        print(j)

def work(w2v_model,glove_model,sex_vector):
    a = []
    with open("country.txt", encoding='utf-8') as f:
        for i in f:
            a = i.split("、")
        # print(len(a))
    a = delete_repeat(a)
    # print(len(a))

    sex_vector_g=vector_between_2_words_g(glove_model,"男子","女子")

    x=[]
    y=[]
    for i in a:
        try:
            print(i+":",end='')
            print(projection(w2v_model,i,sex_vector),end=' ')
            print(projection_g(glove_model,i,sex_vector_g))
            x.append(projection(w2v_model,i,sex_vector))
            y.append(projection_g(glove_model,i,sex_vector_g))
        except:
            print(i + "不存在")


    print("数量："+str(len(x)))
    print("系数1："+str(calculate_spearman_correlation(x,y)))
    print("系数2：" + str(calculate_spearman_correlation_p(x, y)))

    # plt.scatter(x,y)
    #
    # x_a =np.array( [-0.4,0,0.4])
    # y_a = x_a
    # plt.plot(x_a, y_a,c="black",ls="--")
    # plt.axhline(y=0.0, c="black", ls="--", lw=2)
    # plt.axvline(x=0.0, c="black", ls="--", lw=2)

    # plt.bar(x,)

    plt.show()
