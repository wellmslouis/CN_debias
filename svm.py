# -*- coding: utf-8 -*-
import csv
import pickle
import time

import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split, cross_val_score
from tqdm import tqdm
from sklearn import svm

from w2v import projection


def build_svm(model):
    x = []
    y = []
    with open("sex.txt", encoding='utf-8') as sex_file:
        for i in sex_file:
            str = i.replace('\n', '')
            x.append(model.wv[str])
            y.append(1)

    with open("no_sex.txt", encoding='utf-8') as no_sex_file:
        with open("not_involve.txt", "w", encoding='utf-8') as not_inv_file:
            for i in no_sex_file:
                try:
                    str = i.replace('\n', '')
                    x.append(model.wv[str])
                    y.append(0)
                except:
                    print(str + "不存在")
                    not_inv_file.write(str + "\n")
    x = np.array(x)
    y = np.array(y)

    svm_model = svm.SVC(C=1, kernel='linear')
    svm_model.fit(x, y)

    s = pickle.dumps(svm_model)
    f = open('svm.model', "wb+")
    f.write(s)
    f.close()

def build_svm_with_10(model):
    x = []
    y = []
    with open("sex.txt", encoding='utf-8') as sex_file:
        for i in sex_file:
            str_sex = i.replace('\n', '')
            x.append(model.wv[str_sex])
            y.append(1)

    with open("no_sex.txt", encoding='utf-8') as no_sex_file:
        with open("not_involve.txt", "w", encoding='utf-8') as not_inv_file:
            for i in no_sex_file:
                try:
                    str_sex = i.replace('\n', '')
                    x.append(model.wv[str_sex])
                    y.append(0)
                except:
                    print(str_sex + "不存在")
                    not_inv_file.write(str_sex + "\n")
    x = np.array(x)
    y = np.array(y)
    cv_scores=[]
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=111, stratify=y)
    svm_model = svm.SVC(C=1, kernel='linear', class_weight="balanced")
    scores = cross_val_score(svm_model, x, y, scoring='accuracy', cv=10)
    cv_scores.append(scores.mean())
    print("accuracy:")
    print(scores)
    std=scipy.stats.sem(scores)
    min=scores.mean()-1.96*std
    max=scores.mean()+1.96*std
    print("置信区间在95%置信度上为["+str(min)+','+str(max)+']')
    # svm_model.fit(x, y)

    # scores = cross_val_score(svm_model, x, y, scoring='recall', cv=10)
    # cv_scores.append(scores.mean())
    # print("recall:")
    # print(scores)
    # std=scipy.stats.sem(scores)
    # min=scores.mean()-1.96*std
    # max=scores.mean()+1.96*std
    # print("置信区间在95%置信度上为["+str(min)+','+str(max)+']')
    #
    #
    # scores = cross_val_score(svm_model, x, y, scoring='f1', cv=10)
    # cv_scores.append(scores.mean())
    # print("f1:")
    # print(scores)
    # std=scipy.stats.sem(scores)
    # min=scores.mean()-1.96*std
    # max=scores.mean()+1.96*std
    # print("置信区间在95%置信度上为["+str(min)+','+str(max)+']')

    s = pickle.dumps(svm_model)
    f = open('svm.model', "wb+")
    f.write(s)
    f.close()


def delete_word(all_file, delete_file, out_file):
    all = []
    with open(all_file, encoding='utf-8') as f_a:
        for i in f_a:
            all.append(i)

    delete_ = []
    with open(delete_file, encoding='utf-8') as f_b:
        for i in f_b:
            delete_.append(i)

    for i in delete_:
        if i in all:
            all.remove(i)

    with open(out_file, "w", encoding='utf-8') as f_c:
        for i in all:
            f_c.write(i)


def load_svm(file_name):
    f2 = open(file_name, 'rb')
    s2 = f2.read()
    return pickle.loads(s2)


def predict_svm(w2v_model):
    f2 = open("svm.model", 'rb')
    s2 = f2.read()
    svm_model=pickle.loads(s2)
    a = w2v_model.wv["女"]
    testx=[a]
    a_pre = svm_model.predict(testx)
    print("a_pre:", a_pre)

    # words=w2v_model.wv.key_to_index
    # with open("sex_final.txt", "w", encoding='utf-8') as sex_f:
    #     with open("no_sex_final.txt", "w", encoding='utf-8') as n_sex_f:
    #         for word in tqdm(words):
    #             testx=[w2v_model.wv[word]]
    #             testx_pre=svm_model.predict(testx)
    #             if testx_pre==1:
    #                 sex_f.write(word+"\n")
    #             elif testx_pre==0:
    #                 n_sex_f.write(word+"\n")


#画figure7使用
def length_svm(w2v_model,sex_vector):
    f2 = open("svm.model", 'rb')
    s2 = f2.read()
    svm_model = pickle.loads(s2)
    x=[]
    sex_axis=[]
    word=[]
    with open("figure7words.txt", encoding='utf-8') as sex_file:
        for i in sex_file:
            str = i.replace('\n', '')
            word.append(str)
            x.append(w2v_model.wv[str])
            sex_axis.append(projection(w2v_model,str,sex_vector))
    svm_axis=svm_model.decision_function(x)

    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc")
    plt.scatter(sex_axis, svm_axis,c='w')
    for i in range(len(x)):
        plt.text(sex_axis[i],svm_axis[i],word[i],fontproperties=font,fontsize=14)
    plt.axhline(y=0.0, c="r", ls="--", lw=2)
    plt.axvline(x=0.0, c="r", ls="--", lw=2)
    plt.show()
    # with open("figure7words_sex.csv", "w", encoding='utf-8') as sex_w_file:
    #     writer = csv.writer(sex_w_file)
    #     for p in person:
    #         writer.writerow(p)

