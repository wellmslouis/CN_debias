from math import sqrt

import numpy as np

from w2v import *
from bias import *

def identify_gender_subspace(w2v_model):
    B=vector_between_2_unit_words(w2v_model,"男子","女子")
    num=1
    with open("sex_team.txt", encoding='utf-8') as f:
        for i in f:
            if num>1:
                a = i.split(" ")
                for j in range(len(a)):
                    a[j]=a[j].replace("\n","")
                B+=vector_between_2_unit_words(w2v_model,a[0],a[1])
            else:
                num+=1
    B/=num
    return unit_word(B)
    # print(B)
    # print(num)

#硬去偏_中和
#把性别中性词中和消偏
def hard_debias_neutralize(w2v_model):
    B=identify_gender_subspace(w2v_model)#求性别子空间（差平均）
    # all_db=0
    # all_db_n=0
    num=0
    # sex=equalize(w2v_model)
    with open("no_sex_final.txt", encoding='utf-8') as f:
        with open("w2v_vector_neutralize.txt", "w", encoding='utf-8') as f_n:
        # for m in f:
        #     a=m.split("、")
        # a = delete_repeat(a)
        #
        # for word in a:
            for word in f:
                word=word.replace("\n","")
                try:
                    embedding_n=unit_word(w_vertical(w2v_model.wv[word],B))

                    f_n.write(word)
                    for i in embedding_n:
                        f_n.write(" ")
                        f_n.write(str(i))
                    f_n.write("\n")
                    # db_n = np.dot(embedding_n, np.array(gender)) / (np.linalg.norm(gender) * np.linalg.norm(embedding_n))
                    #
                    # db=np.dot(embedding_n,np.array(sex))/(np.linalg.norm(sex)*np.linalg.norm(embedding_n))
                    # print(word+":"+str(db))
                    # all_db+=abs(db)
                    # all_db_n+=abs(db_n)
                    num+=1
                    if num%10000==0:
                        print(num)
                except:
                    print(word)
    # all_db/=num
    # print("均衡:"+str(all_db))
    #
    # print("中和:"+str(all_db_n/num))

    # direct_bias(w2v_model,gender)

#硬去偏_均衡
#处理性别对
def equalize(w2v_model):
    B = identify_gender_subspace(w2v_model)
    with open("sex_team.txt", encoding='utf-8') as f:
        with open("w2v_vector_equalize.txt", "w", encoding='utf-8') as f_e:
            for team in f:
                words=team.replace("\n","").split(" ")
                a=unit_word(w2v_model.wv[words[0]])
                b=unit_word(w2v_model.wv[words[1]])

                u=(a+b)/2
                v=w_vertical(u,B)
                a_e=v+unit_word(B*cos(a,B)-B*cos(u,B))*sqrt(1-(np.linalg.norm(v))*(np.linalg.norm(v)))
                b_e=v+unit_word(B*cos(b,B)-B*cos(u,B))*sqrt(1-(np.linalg.norm(v))*(np.linalg.norm(v)))
                f_e.write(words[0])
                for i in a_e:
                    f_e.write(" ")
                    f_e.write(str(i))
                f_e.write("\n")
                f_e.write(words[1])
                for i in b_e:
                    f_e.write(" ")
                    f_e.write(str(i))
                f_e.write("\n")
    # print(np.linalg.norm(a_e))
    # print(np.linalg.norm(b_e))
    # return a_e-b_e

#硬去偏文件合成
#中和+均衡+其它
def merge_dehardbias_txt(w2v_model):
    sex_team=[]
    with open("w2v_vector_hard.txt","w", encoding='utf-8') as f:
        with open("w2v_vector_neutralize.txt", encoding='utf-8') as f1:
            for i in f1:
                f.write(i)
        with open("w2v_vector_equalize.txt", encoding='utf-8') as f2:
            for i in f2:
                f.write(i)
        with open("sex_final.txt",encoding='utf-8')as f3:
            #把性别对文件中的性别对变成一个列表
            with open("sex_team.txt",encoding='utf-8')as f4:
                for i in f4:
                    words=i.replace("\n","").split(" ")
                    sex_team.append(words[0])
                    sex_team.append(words[1])
            for i in f3:
                word=i.replace("\n","")
                if word not in sex_team:
                    f.write(word)
                    word_embedding=w2v_model.wv[word]
                    for j in word_embedding:
                        f.write(" ")
                        f.write(str(j))
                    f.write("\n")

