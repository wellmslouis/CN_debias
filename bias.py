from tqdm import tqdm

from w2v import *
from work import *


def direct_bias(w2v_model,sex_vector):
    # with open("no_sex_final.txt", encoding='utf-8') as no_sex_file:
    #     db=0
    #     num=0
    #     for word in tqdm(no_sex_file):
    #         try:
    #             word = word.replace('\n', '')
    #             db+=abs(projection(w2v_model,word,sex_vector))
    #             num+=1
    #         except:
    #             print(word+"不存在")
    #     db/=num
    #     print("直接偏见为："+str(db))
    a = []
    with open("work.txt", encoding='utf-8') as f:
        for i in f:
            a = i.split(" ")
        # print(len(a))
    a = delete_repeat(a)

    db=0
    num=0
    for word in a:
        try:
            db+=abs(projection(w2v_model,word,sex_vector))
            num+=1
        except:
            print(word+"不存在")
    db /= num
    print("直接偏见为："+str(db))

def five_most_extreme(w2v_model,axis1,axis2):
    vector=vector_between_2_words(w2v_model,axis1,axis2)
    a = []#职业列表
    with open("work.txt", encoding='utf-8') as f:
        for i in f:
            a = i.split(" ")
        # print(len(a))
    a = delete_repeat(a)
    # print(len(a))

    work_dic={}#职业字典 职业：投影
    for i in a:
        work_dic[i]=projection(w2v_model,i,vector)

    work_dic_order=sorted(work_dic.items(), key=lambda x: x[1], reverse=False)

    for i in work_dic_order:
        print(i)

def cos(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

#w是词向量，g是性别子空间
#w_g=(w·g)g，其中w和g都是单位向量
# def w_g(w,g):
#     return unit_word(g)*cos(w,g)

#w是词向量，g是性别子空间
#w垂直=w-w_g
def w_vertical(w,g):
    return unit_word(w)-unit_word(g)*cos(w,g)

#计算β(w,v)，w,v为词，g为性别子空间
def indirect_bias_beta(w2v_model,w,v,g):
    w_em=unit_word(np.array(w2v_model.wv[w]))
    v_em=unit_word(np.array(w2v_model.wv[v]))
    w_ver=w_vertical(w_em,g)#w垂直
    v_ver=w_vertical(v_em,g)#v垂直
    # print("cos垂直="+str(cos(w_ver,v_ver)))
    # print("cos="+str(cos(w_em,v_em)))
    # print("cos差："+str(cos(w_ver,v_ver)-cos(w_em,v_em)))
    # return 1-np.dot(w_ver,v_ver)/(np.linalg.norm(w_ver)*np.linalg.norm(v_ver)*np.dot(unit_word(w_em),unit_word(v_em)))
    return 1-cos(w_ver,v_ver)/cos(w_em,v_em)