# -*- coding: utf-8 -*-

import numpy as np
import logging
import sys
import gensim.models as word2vec
from gensim.models.word2vec import LineSentence, logger


def train_word2vec(dataset_path, out_vector):
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # 把语料变成句子集合
    sentences = LineSentence(dataset_path)
    # 训练word2vec模型（size为向量维度，window为词向量上下文最大距离，min_count需要计算词向量的最小词频）
    model = word2vec.Word2Vec(sentences, vector_size=50, sg=1, window=5, min_count=200, workers=4, epochs=5)
    # (iter随机梯度下降法中迭代的最大次数，sg为1是Skip-Gram模型)
    # 保存word2vec模型（创建临时文件以便以后增量训练）
    model.save("word2vec_50.model")
    model.wv.save_word2vec_format(out_vector, binary=False)


# 加载模型
def load_word2vec_model(w2v_path):
    model = word2vec.Word2Vec.load(w2v_path)
    return model


# 计算词语的相似词
def calculate_most_similar(model, word):
    similar_words = model.wv.most_similar(word)
    print(word)
    for term in similar_words:
        print(term[0], term[1])


# 计算两个词相似度
def calculate_words_similar(model, word1, word2):
    return model.wv.similarity(word1, word2)


# 找出不合群的词
def find_word_dismatch(model, list):
    print(model.wv.doesnt_match(list))

def vector_between_2_words(w2v_model,word1,word2):
    return w2v_model.wv[word1]-w2v_model.wv[word2]

def vector_between_2_unit_words(w2v_model,word1,word2):
    return unit_word(w2v_model.wv[word1])-unit_word(w2v_model.wv[word2])

#word在vector方向上的单位投影
def projection(w2v_model,word,vector):
    # print("模为：" + str(np.linalg.norm(w2v_model.wv[word])))
    return np.dot(np.array(w2v_model.wv[word]),np.array(vector))/(np.linalg.norm(vector)*np.linalg.norm(w2v_model.wv[word]))

#单位化向量
#word_embedding是向量->np.array类型
def unit_word(word_embedding):
    return word_embedding/np.linalg.norm(word_embedding)