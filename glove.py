import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

def glove_to_w2v(path,save_name):#path是源文件路径，save_name是模型名字
    glove_file = datapath(path)
    tmp_file = get_tmpfile("w.txt")
    _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    # 将词向量保存，方便下次直接导入
    model.save(save_name)

    # 加载词向量模型
    # word_embeddings = KeyedVectors.load("glove_embedding")
    # vector = word_embeddings["水管"]

def calculate_most_similar_g(model, word):
    similar_words = model.most_similar(word)
    print(word)
    for term in similar_words:
        print(term[0], term[1])

def calculate_words_similar_g(model, word1, word2):
    return model.similarity(word1, word2)

def vector_between_2_words_g(glove_model,word1,word2):
    return glove_model[word1]-glove_model[word2]

#word在vector方向上的投影
def projection_g(glove_model,word,vector):
    # print("模为：" + str(np.linalg.norm(glove_model[word])))
    return np.dot(np.array(glove_model[word]),np.array(vector))/(np.linalg.norm(vector)*np.linalg.norm(glove_model[word]))