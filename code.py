# -*- coding: utf-8 -*-
"""
功 能：基于同义词挖掘的SKipGram词向量训练模型
版权信息：技术有限公司，版本所有(C) 2010-2022
修改记录：2015-3-17 12:00 Li Wenbo l00634667 创建
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import collections
import numpy as np
import random
from gensim.models import KeyedVectors
import jieba
from string import punctuation as pun_eng
from zhon.hanzi import punctuation as pun_chn
import re
import scipy.stats
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
from rich.progress import track



class SkipGramDataset(Dataset):
    """
    功能描述：继承自torch.utils.data.Dataset，它是代表这一数据的抽象类。通过继承和重写此
    类，实现模型的小批量优化，即每次都会从原数据集中取出一小批量进行训练，完成一次权重更新后
    ，再从原数据集中取下一个小批量数据，然后再训练再更新。
    接口：（1）__init__()：初始化参数；
         （2）__len__()：提供数据的大小；
         （3）__getitem__()：通过给定索引获取数据与标签。
    修改记录：
    """
    def __init__(self, training_label, 
                 word_to_idx, 
                 idx_to_word, 
                 word_freqs, 
                 syn_ant_sample_size, 
                 num_sampled, 
                 window_size,
                 syn_dict,
                 nonsyn_dict):
        """
        功能描述：初始化参数。
        参数：（1）training_label：目标词的索引；
             （2）word_to_idx：单词-索引的字典；
             （3）idx_to_word：索引-单词的字典；
             （4）word_freqs：词频系数；
             （5）syn_ant_sample_size：每次采样目标词同义词样本和非同义词样本的数量；
             （6）num_sampled：每次负采样的数量；
             （7）window_size：上下文窗口的尺寸；
             （8）syn_dict：保存同义词集的字典；
             （）nonsyn_dict：保存非同义词集的字典。
        返回值：无
        修改记录：
        """
        super(SkipGramDataset, self).__init__()
        self.text_encoded = torch.Tensor(training_label).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.syn_ant_sample_size = syn_ant_sample_size
        self.num_sampled = num_sampled
        self.window_size = window_size
        self.words_size = len(word_to_idx.keys())
        self.new_syn_dict = syn_dict
        self.new_ant_dict = nonsyn_dict
        
    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        """
        功能描述：通过给定索引获取数据（目标词）与标签（上下文单词），其中还包括目标词所对
        应的预定义同义词集与非同义词集的各自的索引。
        参数：（1）idx：目标词的索引
        返回值：（1）center_word：目标词；
               （2）pos_words：上下文窗口内的单词的索引列表；
               （3）neg_words：通过负采样得到的负例单词的索引列表；
               （4）syn_word_idxs：目标词所对应的预定义同义词的索引列表；
               （5）ant_word_idxs：目标词所对应的预定义的上下文相似的非同义词列表；
               （6）valid_syn：目标词所对应的预定义的同义词的个数；
               （7）valid_ant：目标词所对应的预定义的非同义词的个数。
        修改记录：
        """
        # 防止越界
        idx = min( max(idx, self.window_size),
                  len(self.text_encoded)-2- self.window_size)
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx- self.window_size, idx)) \
                        + list(range(idx+1, idx+1+ self.window_size))
        pos_words = self.text_encoded[pos_indices] 
        # 多项式分布采样，取出指定个数的高频词
        neg_words = torch.multinomial(self.word_freqs, 
                                      self.num_sampled+2* self.window_size, 
                                      False)
        neg_words = torch.Tensor(np.setdiff1d(neg_words.numpy(),
                                              pos_words.numpy())\
                                 [:self.num_sampled]).long()
        
        
        index = center_word.item()
        valid_syn = 0
        valid_ant = 0
        #同义词 非同义词 batch生成
        t_syn_word_idxs = (self.words_size)*np.ones([self.syn_ant_sample_size])
        t_ant_word_idxs = (self.words_size)*np.ones([self.syn_ant_sample_size])
        if(index not in self.new_syn_dict.keys()):
            syn_word_idxs = t_syn_word_idxs
        else:
            syn_word_idxs = self.new_syn_dict[index]
            if(self.syn_ant_sample_size > len(syn_word_idxs)):
                t_syn_word_idxs[:len(syn_word_idxs)] = syn_word_idxs
                syn_word_idxs = t_syn_word_idxs
            else:
                t_syn_word_idxs = random.sample(syn_word_idxs, 
                                                self.syn_ant_sample_size)
                syn_word_idxs = t_syn_word_idxs
        if(index not in self.new_ant_dict.keys()):
            ant_word_idxs = t_ant_word_idxs
        else:
            ant_word_idxs = self.new_ant_dict[index]
            if(self.syn_ant_sample_size > len(ant_word_idxs)):
                t_ant_word_idxs[:len(ant_word_idxs)] = ant_word_idxs
                ant_word_idxs = t_ant_word_idxs
            else:
                t_ant_word_idxs = random.sample(ant_word_idxs, 
                                                self.syn_ant_sample_size)
                ant_word_idxs = t_ant_word_idxs
        valid_syn = 1.0/max(1, self.syn_ant_sample_size \
                                - list(syn_word_idxs).count(self.words_size))
        valid_ant = 1.0/max(1, self.syn_ant_sample_size \
                                - list(ant_word_idxs).count(self.words_size))
        return center_word, pos_words, neg_words, \
                    torch.LongTensor(syn_word_idxs), \
                        torch.LongTensor(ant_word_idxs), \
                            valid_syn, valid_ant

class Model(nn.Module):
    """
    功能描述：继承自nn.Module,包含网络各层的定义及forward方法。通过继承和重写此类定义自已
    的网络。只要在nn.Module的子类中定义了forward函数，backward
    函数就会被自动实现。
    接口：（1）__init__()：初始化参数；
         （2）get_syn_ant_loss()：计算目标词向量与它所对应的同义词以及非同义词的损失；
         （3）forward()：前向传播，计算总体损失。
    修改记录：
    """
    def __init__(self, vocab_size, embed_size, pre_vecs, device):
        """
        功能描述：初始化参数。
        参数：（1）vocab_size：输入数据集的字典大小；
             （2）embed_size：词向量的维度（100或者200）；
             （3）pre_vecs：输入字典所对应的预定义的词向量；
             （4）device：设备对象（GPU 或者 CPU）。
        返回值：无
        修改记录：
        """
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.in_embed = nn.Embedding(self.vocab_size + 1, 
                                     self.embed_size, 
                                     padding_idx = self.vocab_size,
                                     sparse=False)
        self.in_embed = nn.Embedding.from_pretrained(torch.tensor(pre_vecs, 
                                                                dtype = float),
                                                      freeze=False)
        self.device = device

    def get_syn_ant_loss(self, 
                         input_labels, 
                         syn_word_idxs, 
                         ant_word_idxs, 
                         valid_syns, 
                         valid_ants, 
                         upsilon, 
                         eta): 
        """
        功能描述：通过给定索引获取目标词、（a）目标词所对应的预定义同义词集与(b)非同义词集
        的各自的索引，计算目标词向量与(a)、(b)所对应词向量的损失。
        参数：（1）input_labels：目标词的索引；
             （2）syn_word_idxs：目标词所对应的同义词集的索引；
             （3）ant_word_idxs：目标词所对应的非同义词集的索引；
             （4）valid_syn：目标词所对应的预定义的同义词的个数；
             （5）valid_ant：目标词所对应的预定义的非同义词的个数；
             （6）upsilon：计算目标词与同义词损失的起始有效相似度的阈值参数；
             （7）eta：目标词与（a）、(b)所产生的损失占在总损失的权重参数。
        返回值：（1）loss：目标词向量与(a)、(b)所对应词向量的损失。
        修改记录：
        """
        sim_total = 0
        syn_sim_w = 0
        ant_sim_w = 0
        w_embeddings = self.in_embed(input_labels)
        
        syn_embeddings = \
            self.in_embed(syn_word_idxs.to(self.device)).sum(1).to(self.device)
        ant_embeddings = \
            self.in_embed(ant_word_idxs.to(self.device)).sum(1).to(self.device)
        syn_valid = \
            syn_embeddings.multiply(w_embeddings).sum(1)\
                .multiply(valid_syns.to(self.device))
        syn_sim_w += upsilon - syn_valid.mean(0)
        ant_valid = \
            ant_embeddings.multiply(w_embeddings).sum(1)\
                .multiply(valid_ants.to(self.device))
        ant_sim_w += ant_valid.mean(0)
        sim_total = max(0, syn_sim_w+ ant_sim_w)
        return -eta*sim_total
       
    def forward(self, 
                input_labels, 
                pos_labels, 
                neg_labels, 
                syn_word_idxs, 
                ant_word_idxs, 
                valid_syn, 
                valid_ant, 
                upsilon, 
                eta0, 
                eta):
        """
        功能描述：定义前向传播网络。只要在nn.Module的子类中定义了forward函数，backward
        函数就会被自动实现。
        参数：（1）input_labels：目标词的索引；
             （2）pos_labels：目标词所对应的上下文词的索引；
             （3）neg_labels：目标词所负样本词的索引；
             （4）syn_word_idxs：目标词所对应的同义词集的索引
             （5）ant_word_idxs：目标词所对应的非同义词集的索引；
             （6）valid_syn：目标词所对应的预定义的同义词的个数；
             （7）valid_ant：目标词所对应的预定义的非同义词的个数；
             （8）upsilon：计算目标词与同义词损失的起始有效相似度的阈值参数；
             （9）eta0：上下文损失占总体损失的权重参数；
             （10）eta：目标词与（a）、(b)所产生的损失占在总损失的权重参数。
        返回值：（1）loss：总体损失值。
        修改记录：
        """
        input_embedding = self.in_embed(input_labels)
        pos_embedding = self.in_embed(pos_labels)
        neg_embedding = self.in_embed(neg_labels)
        pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()
        neg = torch.bmm(neg_embedding, input_embedding.unsqueeze(2)).squeeze()
        log_pos = F.logsigmoid((pos)**2).sum(1)
        log_neg = F.logsigmoid(-(neg)**2).sum(1)
        sim_total = self.get_syn_ant_loss(input_labels, 
                                          syn_word_idxs, 
                                          ant_word_idxs, 
                                          valid_syn, 
                                          valid_ant,
                                          upsilon, 
                                          eta)
        loss = eta0*(log_pos.mean(0) + log_neg.mean(0)) + sim_total
        return -loss

class SynSkipGram():
    """
    功能描述：对数据输入、模型训练、结果输出的封装，具体包括：<1> 通过输入领域数据集、预定
    义的同义词集与非同义词集，基于预训练的词向量在输入数据集上进行微调，输出领域特定的词向量
    ，以及提供接口返回给定单词（集）的topk的单词。<2> 输入给定的单词的列表，直接返回其基于
    预训练词向量的topk单词的列表。
    接口：（1）__init__()：初始化参数；
         （2）compute_ngrams()：计算并返回输入词或短语的ngram组合列表；
         （3）word_to_vec()：基于预训练的词向量，返回给定词的词向量；
         （4）build_dataset()：计算并返回模型训练所需的数据准备;
         （5）fit()：模型训练；
         （6）get_similar_tokens()：计算并返回给定词的topk相似词列表；
         （7）get_similar_tokens_total()：计算并返回给定词的topk并满足阈值条件的相似
         词列表，其中所返回的词集不局限与输入数据集中；
    修改记录：
    """
    def __init__(self,
                 data,
                 syn_dict,
                 nonsyn_dict,
                 device = torch.device("cpu"),
                 save_title="",
                 workers=8,
                 window_size=2,
                 num_sampled=64,
                 batch_size=64,
                 num_epochs=10,
                 embed_size=100,
                 syn_ant_sample_size = 8,
                 topk_ws=-1,
                 lr=1e-5,
                 eta=1000,
                 eta0=1e-9,
                 upsilon=0.6,
                 pre_vecs_file="",
                 words_core_file="",
                 stop_words_file=""):
        """
        功能描述：初始化参数。
        参数：（1）data：输入数据集（格式：["从来", "都", "没有", "什么", ...]）；
             （2）syn_dict：同义词的字典（{"没有":["无"]}）；
             （3）nonsyn_dict：非同义词字典（{"没有":["有"]}）；
             （4）device：设备对象；
             （5）workers：CPU工作数量；
             （6）save_title：保存训练模型的名字；
             （7）window_size：上下文窗口尺度；
             （8）num_sampled：每次负采样的数量；
             （9）batch_size：batch的大小；
             （10）num_epochs：迭代次数；
             （11）embed_size：词向量维度（100或者200）；
             （12）syn_ant_sample_size：每次采样目标词同义词样本和非同义词样本的数量；
             （13）topk_ws：指定输入数据集中基于词频的前topk_ws个词；
             （14）lr：学习率；
             （15）eta：目标词与（a）、(b)所产生的损失占在总损失的权重参数。
             （16）eta0：目标词与上下文词所产生的损失占在总损失的权重参数。
             （17）upsilon：计算目标词与同义词损失的起始有效相似度的阈值参数；
    
        返回值：无
        修改记录：
        """
        if(embed_size not in [100, 200]):
            print("词向量维度只限可选：100或200.")
            return
        self.training_data = data
        self.syn_dict = syn_dict
        self.nonsyn_dict = nonsyn_dict
        self.device = device
        self.save_title = save_title
        self.workers = workers
        self.window_size = window_size
        self.num_sampled = num_sampled
        self.bt_size = batch_size
        self.num_epochs = num_epochs
        self.embed_size = embed_size
        self.syn_ant_sample_size = syn_ant_sample_size
        if(topk_ws == -1):
            self.topk_ws = len(set(data))
        else:
            self.topk_ws = topk_ws
        self.rate = lr
        self.eta = eta
        self.eta0 = eta0
        self.upsilon = upsilon
        self.dictionary = dict()
        self.reversed_dictionary = dict()
        if(pre_vecs_file == "" and self.embed_size == 100):
            self.pre_vecs_file="tencent-ailab-embedding-zh-d100-v0.2.0.txt"
        elif(pre_vecs_file == "" and self.embed_size == 200):
            self.pre_vecs_file="tencent-ailab-embedding-zh-d200-v0.2.0.txt"
        else:
            self.pre_vecs_file = pre_vecs_file
        file = self.pre_vecs_file
        wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False)
        wv_from_text.init_sims(replace=True)
        self.wv_from_text = wv_from_text
        self.model = None
        self.words_core_file = words_core_file
        self.stop_words_file = stop_words_file

    def compute_ngrams(self, word, min_n, max_n):
        """
        功能描述：当出现OOV的情况时，使用ngram估计OOV的单词向量。返回输入单词的ngram组合
        列表。
        参数：（1）word：目标词；
             （2）min_n：最小的ngram窗口大小；
             （3）max_n：最大的ngram窗口大小；
        返回值：（1）输入词word所对应的ngram字词的列表。
        修改记录：
        """
        extended_word =  word
        ngrams = []
        for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
            for i in range(0, len(extended_word) - ngram_length + 1):
                ngrams.append(extended_word[i:i + ngram_length])
        return list(set(ngrams))

    def word_to_vec(self, word, wv_from_text, min_n=1, max_n=3):
        """
        功能描述：计算并返回基给定词的基于预训练词向量库的词向量。
        参数：（1）word：目标词；
             （2）wv_from_text：预训练词向量库；
             （3）min_n：最小的ngram窗口大小；
             （4）max_n：最大的ngram窗口大小；
        返回值：（1）输入词word所对应的预训练词向量。
        修改记录：
        """
        word_size = wv_from_text.vectors.shape[1]   
        ngrams = self.compute_ngrams(word, min_n=min_n, max_n=max_n)
        if word in wv_from_text.key_to_index.keys():
            return wv_from_text[word]
        else:  
            word_vec = np.zeros(word_size, dtype=np.float32)
            ngrams_found = 0
            ngrams_single = [ng for ng in ngrams if len(ng) == 1]
            ngrams_more = [ng for ng in ngrams if len(ng) > 1]
            for ngram in ngrams_more:
                if ngram in wv_from_text.key_to_index.keys():
                    word_vec += wv_from_text[ngram]
                    ngrams_found += 1
            if ngrams_found == 0:
                for ngram in ngrams_single:
                    if(ngram not in wv_from_text.key_to_index.keys()):
                        continue
                    word_vec += wv_from_text[ngram]
                    ngrams_found += 1
            if word_vec.any():
                return word_vec / max(1, ngrams_found)
            else:
                return -1

    def build_dataset(self, data):   
        """
        功能描述：计算模型训练的字典、反向索引字典、同义词以及非同义词的基于索引的字典，返
        回输入词库对应的预训练词向量集、输入数据集按词频排序的topk词列表。
        参数：  （1）data：输入数据集。
        返回值：（1）count: 输入数据集按词频排序的词列表（[("中国", 100), ("技术", 90)
                   ,...]）；
               （2）target_vecs：输入词库对应的预训练词向量集。
        修改记录：
        
        """    
        count = [['UNK', -1]]
        count.extend(collections.Counter(data).most_common(self.topk_ws - 1))
        for word, _ in count:
            self.dictionary[word] = len(self.dictionary)
        unk_count = 0
        count[0][1] = unk_count
        self.reversed_dictionary = dict(zip(self.dictionary.values(), \
                                            self.dictionary.keys()))
        word_size = self.wv_from_text.vectors.shape[1]  
        
        new_syn_dict = dict()
        for word in self.syn_dict:
            if(word in self.dictionary):
                new_syn_dict[self.dictionary[word]]\
                    = [self.dictionary[syn] for syn in self.syn_dict[word] \
                       if syn in self.dictionary]
            else:
                continue
        self.syn_dict = new_syn_dict
        new_nonsyn_dict = dict()
        for word in self.nonsyn_dict:
            if(word in self.dictionary):
                new_nonsyn_dict[self.dictionary[word]] \
                    = [self.dictionary[syn] \
                       for syn in self.nonsyn_dict[word] \
                           if syn in self.dictionary]
            else:
                continue
        self.nonsyn_dict = new_nonsyn_dict
        
        target_vecs = []
        for word in self.dictionary.keys():
            vec = self.word_to_vec(word, 
                                   self.wv_from_text,
                                   min_n = 1, 
                                   max_n = 3)
            if(isinstance(vec,int) and vec == -1):
                target_vecs.append(list(np.random.uniform(-0.5/word_size, 
                                                          0.5/word_size, 
                                                          (word_size))))
            else:
                target_vecs.append(list(vec))
        return count, target_vecs
    
    def fit(self):
        """
        功能描述：训练模型。
        参数：无。
        返回值：model：训练后的模型。
        修改记录：
        
        """   
        count, pretrained_vecs = self.build_dataset(self.training_data)
        pretrained_vecs.append([0 for i in range(self.embed_size)])
        words_size = len(self.dictionary.keys())
        self.model = Model(words_size, 
                      self.embed_size, 
                      pretrained_vecs,
                      self.device).to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.rate)
        training_label = []
        for word in self.training_data:
            if(word in  self.dictionary):
                training_label.append(self.dictionary[word])
            else:
                training_label.append(self.dictionary["UNK"])
        # 计算词频
        word_count = np.array([freq for _, freq in count], 
                              dtype=np.float32)
        # 计算每个词的词频
        word_freq = word_count / np.sum(word_count)
        # 词频变换
        word_freq = word_freq ** (3. / 4.)
        train_dataset = SkipGramDataset(training_label, 
                                        self.dictionary, 
                                        self.training_data, 
                                        word_freq, 
                                        self.syn_ant_sample_size,
                                        self.num_sampled,
                                        self.window_size,
                                        self.syn_dict,
                                        self.nonsyn_dict)
     
        for e in range(self.num_epochs):
            dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                     batch_size=self.bt_size, 
                                                     drop_last=True, 
                                                     shuffle=True, 
                                                     num_workers=self.workers)
            
            sample = iter(dataloader)				
            # 从迭代器中取出一批次样本
            center_word,   \
            pos_words,     \
            neg_words,     \
            syn_word_idxs, \
            ant_word_idxs, \
            valid_syn,     \
            valid_ant  = sample.next()				
            for ei, (input_labels,  \
                     pos_labels,    \
                     neg_labels,    \
                     syn_word_idxs, \
                     ant_word_idxs, \
                     valid_syn,     \
                     valid_ant) in enumerate(dataloader):
                input_labels = input_labels.to(self.device)
                pos_labels = pos_labels.to(self.device)
                neg_labels = neg_labels.to(self.device)
                optimizer.zero_grad()
                loss = self.model(input_labels, 
                             pos_labels, 
                             neg_labels, 
                             syn_word_idxs, 
                             ant_word_idxs, 
                             valid_syn, 
                             valid_ant, 
                             self.upsilon, 
                             self.eta0, 
                             self.eta)
                loss.backward()
                optimizer.step()
                if(ei % 500 == 0):
                    print("epoch: {}, iter: {}, loss: {}"\
                          .format(e, ei, loss.item()))
        if(self.save_title != ""):
            torch.save(self.model, self.save_title + ".pkl") 
        return self.model
    
    # def self_attention(sentence):
        
    #     torch.matmul(W, W.T)
        
    
    def most_similar_local(self, item, k):
        """
        功能描述：计算给定的词的topk个基于余弦相似度的语义相似词。
        参数：（1）item：输入的词（"中国"）；
             （2）k：指定topk的值。
        返回值：（1）results：输入词对应的topk的语义相似词列表（[("小米", 0.8),...]）。
        修改记录：
        
        """   
        if(item not in self.dictionary):
            return []
        results = []
        W = self.model.in_embed.weight.data
        x = W[self.dictionary[item]]
        # 添加的 1e-9 是为了数值稳定性，防止除法出现 0
        cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) \
                                    * torch.sum(x * x) + 1e-9).sqrt()
        _, topk = torch.topk(cos, k = k+1)
        for i in topk[1:]:
            if(int(i)>=len(self.dictionary.keys())):
                continue
            results.append((self.reversed_dictionary[int(i)], cos[i].item()))
        return results
    
    def most_similar_global(self, item, k, min_n=1, max_n=3):
        """
        功能描述：计算并返回给定词的topk并满足阈值条件的单词的相似词列表，其中所返回的词集
        不局限与输入数据集中。
        参数：（1）item：输入的词（"得到"）；
             （2）k：指定topk的值；
             （3）min_n：最小的ngram窗口大小；
             （4）max_n：最大的ngram窗口大小；
        返回值：（1）results：输入词对应的topk的语义相似词列表（[("苹果", 0.8),...]）;
               （2）pre_syn_words: 输入的同义词集
        修改记录：
        
        """  
        if(item not in self.dictionary):
            return []
        sim_words = []
        W = self.model.in_embed.weight.data
        x = W[self.dictionary[item]]
        # 添加的 1e-9 是为了数值稳定性，防止除法出现0
        cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1)\
                                   * torch.sum(x * x) + 1e-9).sqrt()
        # 数据集词库中的近义词
        _, topk = torch.topk(cos, k = k+1)
        for i in topk[1:]:
            if(int(i)>=len(self.dictionary.keys())):
                continue
            sim_words.append((self.reversed_dictionary[int(i)], cos[i].item()))
        
        # 数据集词库外的近义词
        vec = self.word_to_vec(item, 
                               self.wv_from_text, 
                               min_n = min_n, 
                               max_n = max_n)
        if(not isinstance(vec,int)):
            syns = self.wv_from_text.most_similar(positive=[vec], topn=k)
        else:
            syns = []
        syns = [item for item in syns if item[1] \
                not in [item[1] for item in sim_words]]
            
        if(item in self.dictionary and self.dictionary[item] in self.syn_dict):
            pre_syn_words = [self.reversed_dictionary[it] \
                          for it in self.syn_dict[self.dictionary[item]]]
        else:
            pre_syn_words = []
    
        sim_words = sorted(sim_words + syns, key=lambda x: x[0], reverse=True)    
        results = []
        for w in sim_words:
            if(w[0] not in pre_syn_words):
                results.append(w)
        return results, pre_syn_words

    def most_similars(self, item, k, min_n=1, max_n=3):
        """
        功能描述：在不微调的情况下计算并返回给定词的topk并满足阈值条件的单词的相似词列表，其中所
        返回的词集不局限与输入数据集中。
        参数：（1）item：输入的词（"AA"）；
             （2）k：指定topk的值；
             （3）min_n：最小的ngram窗口大小；
             （4）max_n：最大的ngram窗口大小；
        返回值：（1）results：输入词对应的topk的语义相似词列表（[("苹果", 0.8),...]）。
        修改记录：

        """
        # 数据集词库外的近义词
        vec = self.word_to_vec(item,
                               self.wv_from_text,
                               min_n=min_n,
                               max_n=max_n)
        if (not isinstance(vec, int)):
            syns = self.wv_from_text.most_similar(positive=[vec], topn=k)
        else:
            syns = []
        return syns

    def most_similars_2(self, item, k, pos, ant, min_n=1, max_n=3):
        """
        功能描述：在不微调的情况下计算并返回给定词的topk并满足阈值条件的单词的相似词列表，其中所
        返回的词集不局限与输入数据集中。
        参数：（1）item：输入的词（"华啊"）；
             （2）k：指定topk的值；
             （3）min_n：最小的ngram窗口大小；
             （4）max_n：最大的ngram窗口大小；
        返回值：（1）results：输入词对应的topk的语义相似词列表（[("苹果", 0.8),...]）。
        修改记录：

        """
        pos_vec = np.zeros(self.embed_size, dtype=np.float32)
        for word in pos:
            wvec = self.word_to_vec(word,
                                   self.wv_from_text,
                                   min_n=min_n,
                                   max_n=max_n)
            if(not isinstance(wvec,int)):
                pos_vec += wvec
        if(len(pos) > 1):
            pos_vec = pos_vec/len(pos)
        
        ant_vec = np.zeros(self.embed_size, dtype=np.float32)
        for word in ant:
            wvec = self.word_to_vec(word,
                                   self.wv_from_text,
                                   min_n=min_n,
                                   max_n=max_n)
            if(not isinstance(wvec,int)):
                ant_vec += wvec
        if(len(ant) > 1):
            ant_vec = ant_vec/len(ant)
        
        # 数据集词库外的近义词
        vec = np.zeros(self.embed_size, dtype=np.float32)
        if (not isinstance(vec, int)):
            vec += self.word_to_vec(item,
                                   self.wv_from_text,
                                   min_n=min_n,
                                   max_n=max_n)       
        vec = vec + pos_vec - ant_vec

        if (vec.sum() == 0 and vec.std() == 0):
            syns = []
        else:
            syns = self.wv_from_text.most_similar(positive=[vec], topn=k)
        return syns
    
    def most_similars_3(self, items, k, min_n=1, max_n=3):
        """
        功能描述：在不微调的情况下计算并返回给定词的topk并满足阈值条件的单词的相似词列表，其中所
        返回的词集不局限与输入数据集中。
        参数：（1）item：输入的词（"问问"）；
             （2）k：指定topk的值；
             （3）min_n：最小的ngram窗口大小；
             （4）max_n：最大的ngram窗口大小；
        返回值：（1）results：输入词对应的topk的语义相似词列表（[("苹果", 0.8),...]）。
        修改记录：

        """
        vec = np.zeros(self.embed_size, dtype=np.float32)
        for word in items:
            wvec = self.word_to_vec(word,
                                   self.wv_from_text,
                                   min_n=min_n,
                                   max_n=max_n)
            if(not isinstance(wvec,int)):
                vec += wvec
        if(len(items) > 1):
            vec = vec/len(items)
        
        if (vec.sum() == 0 and vec.std() == 0):
            syns = []
        else:
            syns = self.wv_from_text.most_similar(positive=[vec], topn=k)
        return syns
    
    def expand_syns(self, target_idxs, vob_idx, topk=100, threshold=0.8, min=1, max=3):
    
        W = self.model.in_embed.weight.data
        
        target_vecs = W[target_idxs]
        vob_vecs = W[vob_idx]
        
        result = dict()
        sim = torch.mm(target_vecs, vob_vecs.T)
        for i in range(len(target_vecs)):
            result[self.reversed_dictionary[target_idxs[i]]] = []
            candidates = (-sim[i, :]).argsort()[0:topk]
            nearest = [j for j in candidates if sim[i, j] > threshold]
            for index in nearest:
                 result[self.reversed_dictionary[target_idxs[i]]].append(self.reversed_dictionary[vob_idx[index.item()]])
    
        return result

    def most_similars_4(self, item, k, sentence, min_n=1, max_n=3):
        """
        功能描述：在不微调的情况下计算并返回给定词的topk并满足阈值条件的单词的相似词列表，其中所
        返回的词集不局限与输入数据集中。
        参数：（1）item：输入的词（"方法"）；
             （2）k：指定topk的值；
             （3）min_n：最小的ngram窗口大小；
             （4）max_n：最大的ngram窗口大小；
        返回值：（1）results：输入词对应的topk的语义相似词列表（[("苹果", 0.8),...]）。
        修改记录：

        """
        vec = np.zeros(self.embed_size, dtype=np.float32)
        if (not isinstance(vec, int)):
            vec += self.word_to_vec(item,
                                   self.wv_from_text,
                                   min_n=min_n,
                                   max_n=max_n) 
        
        jieba.load_userdict(self.words_core_file)
        with open(self.stop_words_file,'r', encoding='utf-8') as f:
            stop_words_chn=f.readlines()
            stop_words_chn = [word.strip() for word in stop_words_chn]
        
        sentence = re.sub("["+pun_eng+"=]+|["+pun_chn+"]+", " ", sentence)
        words = jieba.cut(sentence)
        words = list(words)
        sen_items = [x for x in words if x != '' and x != ' ' and x != item and x not in stop_words_chn]
        print(sen_items)
        svec = np.zeros(self.embed_size, dtype=np.float32)
        for word in sen_items:
            wvec = self.word_to_vec(word,
                                   self.wv_from_text,
                                   min_n=min_n,
                                   max_n=max_n)
            if(not isinstance(wvec,int)):
                svec += wvec
        if(len(sen_items) > 1):
            svec = svec/len(sen_items)
            
        vec = vec + svec
        if (vec.sum() == 0 and vec.std() == 0):
            syns = []
        else:
            syns = self.wv_from_text.most_similar(positive=[vec], topn=k)
        return syns
    
    def obtain_terms_vecs(self, terms, min_n=1, max_n=3):
        term_vecs = []
        new_segs = []
        for word in terms:
            vec = self.word_to_vec(word, self.wv_from_text, min_n=min_n, max_n=max_n)
            if isinstance(vec, int) and vec == -1:
                pass
            else:
                new_segs.append(word)
                term_vecs.append(list(vec))
        return term_vecs, new_segs

    def KL_divergence(self, p,q, upsilon=2e-9):
        p = [item+upsilon for item in p]
        q = [item+upsilon for item in q]
        return scipy.stats.entropy(p, q, base=2)

    def sym_KL_divergence(self, p, q, upsilon=2e-9):
        return (self.KL_divergence(p, q, upsilon) \
                + self.KL_divergence(q, p, upsilon) \
                    + upsilon) / 2

    def sigmoid(self, x):
      
        z = np.exp(-x)
        sig = 1 / (1 + z)

        return sig

    def sim(self, d1, d2, base=10):
        kl = self.sym_KL_divergence(d1, d2)
        sim = self.sigmoid(1.0/kl**base)
        return sim
    
    # def obtain_topical_vecs(self, target_words, total_vob_words, term_topic_distributions):
    #     term_topic_distributions = torch.tensor(lda.get_topics(), dtype=torch.float32)

    #     target_idxs = [self.dictionary[w] for w in anchors if w in total_vob_words]
    #     filtered_target_words = [w for w in target_words if w in total_vob_words]
    #     filtered_vob_words_idxs = [self.dictionary[w] for w in filtered_vob_words]

    #     target_words_distributions = term_topic_distributions[:, target_idxs].T
    #     new_words_distributions = term_topic_distributions[:, new_words_idxs].T
    
    def obtain_topical_word_sims(self, 
                                 target_words, 
                                 vob_words,  
                                 target_topic_vecs, 
                                 vob_topic_vecs,
                                 r_threshold,
                                 weight=0.5):
        
        target_vecs, filtered_target_words = self.obtain_terms_vecs(target_words)
        vob_vecs, filtered_vob_words = self.obtain_terms_vecs(vob_words)
        
        topical_score = dict()
        for i in range(len(target_topic_vecs)):
            if target_words[i] not in filtered_target_words:
                continue
            else:
                topical_score[target_words[i]] = dict()
            for j in range(len(vob_topic_vecs)):
                if vob_words[i] not in filtered_vob_words:
                    continue
                else:
                    term_sim = self.sim(target_topic_vecs[i], vob_topic_vecs[j])
                    topical_score[target_words[i]][vob_words[j]] = term_sim
        
        embedding_sim_dict = dict()
        similarity = torch.mm(torch.tensor(target_vecs, dtype=torch.double), 
                              torch.tensor(torch.tensor(vob_vecs, dtype=torch.double).T, dtype=torch.double))
        for i in range(len(filtered_target_words)):
            embedding_sim_dict[filtered_target_words[i]] = dict()
            for j in range(len(filtered_vob_words)):
                embedding_sim_dict[filtered_target_words[i]][filtered_vob_words[j]] = similarity[i,j]
        
        hybrid_sims = dict()
        for tw in filtered_target_words:
            hybrid_sims[tw] = []
            for vw in filtered_vob_words:
                hybrid_sims[tw].append((vw, weight*topical_score[tw][vw] \
                                        + (1-weight)*embedding_sim_dict[tw][vw]))
        
        related_words = dict()
        for tw in filtered_target_words:
            related_words[tw] = [(item[0], item[1].item()) for item in hybrid_sims[tw] if item[1] > r_threshold]
        
        return related_words
    
    def obtain_terms_vecs(self, terms, min_n=1, max_n=3):
        term_vecs = []
        new_segs = []
        for word in terms:
            vec = self.word_to_vec(word, self.wv_from_text, min_n=min_n, max_n=max_n)
            if isinstance(vec, int) and vec == -1:
                pass
            else:
                new_segs.append(word)
                term_vecs.append(list(vec))
        return term_vecs, new_segs
    
    def obtain_new_topic_wordset(self, words_list):
        
        output_topical_clusters = []
        for words in words_list:
            clusters = []
            words_vecs, f_words = self.obtain_terms_vecs(words)
            CHs = []
            for cn in range(2, len(f_words)/2):
                clf = KMeans(n_clusters=cn, random_state=9)
                y_pred = clf.fit_predict(words_vecs)
                CHs.append(sm.calinski_harabaz_score(words_vecs, y_pred))
            cn = np.argmax(CHs) + 2
            clf = KMeans(n_clusters=cn, random_state=9)
            y_pred = clf.fit_predict(words_vecs)
            for i in range(cn):
                clusters.append([])
            for i in range(len(f_words)):
                clusters[y_pred[i]].append(f_words[i])
            output_topical_clusters.append(clusters)
            
        return output_topical_clusters
    
    
    
    def filter_word_vecs(self, new_words, old_vecs, old_id2word):
        
        old_vecs = np.array(old_vecs)
        new_ids = []
        new_id2word = dict()
        for i in range(len(old_id2word.keys())):
            if old_id2word[i] in new_words:
                new_id2word[len(new_ids)] = old_id2word[i]
                new_ids.append(i)
               
        new_vecs = old_vecs[new_ids]
        
        return new_vecs, new_words, new_id2word
            
    
    def obtain_topical_clusters(self, words, id2word, wordid_topic_dists):
        
        def sim_metric(point1, point2):
            
            def dist(x,y):   
                return np.sqrt(np.sum((x-y)**2))
            
            def KL_divergence(p,q, upsilon=2e-9):
                p = [item+upsilon for item in p]
                q = [item+upsilon for item in q]
                return scipy.stats.entropy(p, q, base=2)

            def sym_KL_divergence(p, q, upsilon=2e-9):
                return (KL_divergence(p, q, upsilon) \
                        + KL_divergence(q, p, upsilon) \
                            + upsilon) / 2
            # print(len(point1),len(point2))
            dis_embed= dist(point1[:100],point2[:100])
            dis_dist = sym_KL_divergence(point1[100:],point2[100:])
            
            return 0.5*dis_embed + 0.5*dis_dist
        
        my_metric = distance_metric(type_metric.USER_DEFINED, func=sim_metric)
        clusters = []
        words_vecs, f_words = self.obtain_terms_vecs(words)
        new_dists, new_words, new_id2word = self.filter_word_vecs(f_words, wordid_topic_dists, id2word)
        
        term_vecs = np.hstack((np.array(words_vecs), np.array(new_dists)))
        
        CHs = []
        for cn in track(range(41, 50)):
            initial_centers = kmeans_plusplus_initializer(term_vecs, cn).initialize()
            kmeans_instance = kmeans(term_vecs, initial_centers, metric=my_metric)
            kmeans_instance.process()
            y_pred = kmeans_instance.predict(term_vecs)
            CHs.append(sm.calinski_harabasz_score(term_vecs, y_pred))
        cn = np.argmax(CHs) + 2
        initial_centers = kmeans_plusplus_initializer(term_vecs, cn).initialize()
        kmeans_instance = kmeans(term_vecs, initial_centers, metric=my_metric)
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
        centers = kmeans_instance.get_centers()
        
        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                clusters[i][j] = f_words[clusters[i][j]]
            
        return clusters, centers, f_words
                









