# -*- coding: utf-8 -*-
from bias import *
from glove import *
from w2v import *
from svm import *
from sex import *
from work import *
# from delete_bias import *
from sdp import *

if __name__ == '__main__':
    # delete_word("no_sex.txt","not_involve.txt","no_sex_2.txt")

    # dataset_path = "corpusFinal.txt"
    # out_vector = 'w2v_50.vector'
    # train_word2vec(dataset_path, out_vector)

    # w2v_model = load_word2vec_model("word2vec.model")  # 加载w2v模型
    # print("加载w2v模型成功！")
    # glove_model = KeyedVectors.load("glove_embedding")  # 加载glove模型
    # print("加载w2v模型成功！")
    #
    # sex_vector = vector_between_2_words(w2v_model, "男子", "女子")

    # calculate_most_similar_g(glove_model, "娘亲") #最相似的词
    # calculate_words_similar(model, "女", "护士")  # 两个词相似度

    # list = ["早饭", "吃饭", "恰饭", "嘻哈"]
    # find_word_dismatch(model,list)
    # model.wv.

    # similarity = model.wv.n_similarity(['男', '程序员'], ['女', '程序员'])
    # print(f"{similarity:.4f}")

    # print(model.wv['不对'])

    # print(len(model.wv['男']-model.wv['女']))
    # print(miao_similarity(model.wv["男"], model.wv["女"],model.wv["医生"], model.wv["医生"] ))

    # build_svm(w2v_model)
    # build_svm_with_10(w2v_model)

    # predict_svm(w2v_model)
    # a=w2v_model.wv.key_to_index
    # print(a)

    # sex_accuracy(w2v_model, "王子", "公主")
    # sex_accuracy(w2v_model, "男", "女")
    # sex_accuracy(w2v_model, "父亲", "母亲")
    # sex_accuracy(w2v_model, "男性", "女性")
    # sex_accuracy(w2v_model, "儿子", "女儿")
    # sex_accuracy(w2v_model, "丈夫", "妻子")
    # sex_accuracy(w2v_model, "男士", "女士")
    # sex_accuracy(w2v_model, "男人", "女人")
    # sex_accuracy(w2v_model, "爸爸", "妈妈")
    # sex_accuracy(w2v_model, "男孩", "女孩")

    # PCA_(w2v_model)
    # PCA_average(w2v_model)

    # format_work("adj.txt")

    # work(w2v_model, glove_model,sex_vector)

    # direct_bias(w2v_model,sex_vector)
    # length_svm(w2v_model,sex_vector)

    # five_most_extreme(w2v_model,"教师","司机")
    # a = ["模特","化妆师","园艺师","采购员","裁缝"]
    # a=["警察","司机","律师","工人","科学家"]
    # a=["校长","副教授","教师","秘书","船员"]
    #
    # # a=[]
    # # with open("work.txt", encoding='utf-8') as f:
    # #     for i in f:
    # #         a = i.split(" ")
    # #     # print(len(a))
    # # a = delete_repeat(a)
    # # # print(len(a))
    # # b={}
    # for j in a:
    #     print(j)
    #     print(indirect_bias_beta(w2v_model,"摩托车",j,sex_vector))
    #     b[j]=abs(indirect_bias_beta(w2v_model,"摩托车",j,sex_vector))
    # work_dic_order = sorted(b.items(), key=lambda x: x[1], reverse=False)
    # for i in work_dic_order:
    #     print(i)

    # print(indirect_bias_beta(w2v_model,"护理","护士",sex_vector))
    # print(indirect_bias_beta(w2v_model, "护理", "医生", sex_vector))
    # print(indirect_bias_beta(w2v_model, "护理", "牙医", sex_vector))
    # print(indirect_bias_beta(w2v_model, "粉色", "采购员", sex_vector))
    # print(indirect_bias_beta(w2v_model, "粉色", "裁缝", sex_vector))

    # identify_gender_subspace(w2v_model)
    # hard_debias_neutralize(w2v_model)
    # equalize(w2v_model)
    # merge_dehardbias_txt(w2v_model)

    # glove_to_w2v("E:\\Projects\\w2v\\w2v_vector_hard.txt","word2vec_embedding_hard_debias")

    w2v_model_50 = load_word2vec_model("word2vec_50.model")  # 加载w2v模型
    w2v_sdp(w2v_model_50)
    # sdp_test()