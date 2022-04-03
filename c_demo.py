# -*- coding: utf-8 -*-
"""
功 能：基于同义词挖掘的SKipGram词向量模型的使用示例
版权信息：技术有限公司，版本所有(C) 2010-2022
修改记录：2022-3-17 12:00 Li Wenbo l00634667 创建
"""

import torch
import os
import w2v_code
import warnings
warnings.filterwarnings('ignore')
from rich.progress import track
from pathlib import Path



if __name__ == "__main__":
    root_path = Path(__file__).parent  # 获取根目录
    # 模型存储的文件名
    save_title = "_test_"
    # 使用的CPU数
    workers = 12
    # 上下文窗口大小
    C = 5
    # 每次训练负采样个数   
    num_sampled = 64  
    threhold = 0.8
    # 批处理大小
    batch_size = 64
    # 词向量维度（100或200）
    embedding_size = 100
    # 每次训练同义词、非同义词采用个数
    syn_ant_sample_size = 8 
    # 学习率
    rate = 4e-5
    # 上下文损失权重参数
    eta0 = 1
    # 同义词、非同义词损失起始阈值参数
    upsilon = 0.6
    # 同义词、非同义词损失权重参数
    eta = 1000
    # 训练迭代次数
    num_epochs = 1
    # 指定按词频排在前列的词的范围
    top_k_words = 50000
    # 设备对象
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    # 输入的数据集文件
    path = "data_for_w2v801.txt"
    f = open(path,'r', encoding = "utf-8")
    data = f.read()
    training_data = eval(data)
    f.close()
    
    # 读取标签下关键词们
    f = open(str(root_path)+"/w_anchors",'r', encoding = "utf-8")
    dic = f.read()
    data = eval(dic)[:]
    f.close()
    anchors_words = data
    f = open(str(root_path)+"/w_anchor_dist",'r', encoding = "utf-8")
    dic = f.read()
    data = eval(dic)[:]
    f.close()
    anchors_words_vecs = data
    
    
    # 读取训练集中的词们
    f = open(str(root_path)+"/w_vocs",'r', encoding = "utf-8")
    dic = f.read()
    data = eval(dic)[:]
    f.close()
    vocbs = data
    f = open(str(root_path)+"/w_voc_dist",'r', encoding = "utf-8")
    dic = f.read()
    data = eval(dic)[:]
    f.close()
    vocbs_vecs = data
    

    # 输入的同义词集字典
    syn_dict= dict()
    # 输入的非同义词集字典
    ant_dict= dict()
    # 输入的同义词集字典文件
    f = open('syn_dict.txt','r', encoding = "utf-8")
    dic = f.read()
    syn_dict = eval(dic)
    f.close()
    # 输入的非同义词集字典文件
    f = open('ant_dict.txt','r', encoding = "utf-8")
    dic = f.read()
    ant_dict = eval(dic)
    f.close()
    
    # wordcore文件
    word_core_file = Path.joinpath(root_path, "words_core.txt")
    
    pre_vecs_path = Path.joinpath(root_path, "tencent-ailab-embedding-zh-d100-v0.2.0.txt")
    
    stop_words_file = Path.joinpath(root_path, "cn_stopwords.txt")
    
    model = w2v_code.SynSkipGram(training_data, 
                                 syn_dict, 
                                 ant_dict,
                                 device,
                                 save_title,
                                 workers,
                                 C,
                                 num_sampled,
                                 batch_size,
                                 num_epochs,
                                 embedding_size,
                                 syn_ant_sample_size,
                                 top_k_words,
                                 rate,
                                 eta,
                                 eta0,
                                 upsilon,
                                 str(pre_vecs_path),
                                 str(word_core_file),
                                 str(stop_words_file))
    r_model = model.fit()    

    # anchors = list(syn_dict.keys())
    # anchors_idxs = [model.dictionary[w] for w in anchors if w in model.dictionary.keys()]
    
    # vob = [w for w in model.dictionary.keys() if w not in anchors]
    # vob_idxs = [model.dictionary[w] for w in vob]
    
    # res_syn_dict = model.expand_syns(anchors_idxs, vob_idxs, 100, threhold)

    # # 写之前，先检验文件是否存在，存在就删掉
    # if os.path.exists("syn_candidates" + save_title + "[E3].txt"):
    #     os.remove("syn_candidates"+save_title+"[E3].txt")
    # # 以写的方式打开文件，如果文件不存在，就会自动创建
    # file_write_obj = open("syn_candidates" + save_title + "[E3].txt", 'w')
    # count = 0
    # for item in res_syn_dict.keys():
    #     file_write_obj.writelines('\''+str(item)+' \'同义词候选集: ' \
    #                                   +str(res_syn_dict[item]))
    #     file_write_obj.write('\n')
    # file_write_obj.close()   
   
    
    # hyb_sims = model.obtain_topical_word_sims(anchors_words, 
    #                                                 vocbs,  
    #                                                 anchors_words_vecs, 
    #                                                 vocbs_vecs,
    #                                                 r_threshold=0.64,
    #                                                 weight=0.3)
    
    # print(hyb_sims[list(hyb_sims.keys())[10]])
    
    # f = open(str(root_path)+"/w_"+"hyb_sims.txt",'w', encoding = "utf-8")
    # f.write(str(hyb_sims))
    # f.close()   
    
    f = open(str(root_path)+ '/w_cls_word_topic_dists','r', encoding = "utf-8")
    dic = f.read()
    w_cls_word_topic_dists = eval(dic)
    f.close()
    f = open(str(root_path)+ '/w_cls_id2word','r', encoding = "utf-8")
    dic = f.read()
    w_cls_id2word = eval(dic)
    f.close()
    f = open(str(root_path)+ '/w_cls_words','r', encoding = "utf-8")
    dic = f.read()
    w_cls_words = eval(dic)
    f.close()
    
    clusters = model.obtain_topical_clusters(w_cls_words, w_cls_id2word, w_cls_word_topic_dists)
       
    f = open(str(root_path)+"/w_"+"clusters.txt",'w', encoding = "utf-8")
    f.write(str(clusters))
    f.close()      
            
    # 读取已存储的模型
    # saved_model = torch.load(save_title + ".pkl", map_location=device)

    

