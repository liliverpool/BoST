# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:44:26 2020

@author: slab
"""

import numpy as np
import math
from pprint import pprint
from matplotlib import pyplot as plt
import time
import os


model_name = "HCT"
save_path = ""

# parameter settings and initializations
# the number of documnets, topics, words and other hyper paramters for dirichlet distribution
start = 9
end = 1
data_clip = 100
data = []
word_index = dict()
index_word = dict()
#create_dictionary(data) 
docs_num = 1
topic_num = 1
words_num = 1
alpha = 0.05
beta = 0.05
etas = [] #[1/3, 1/3, 1/3]
context_len = 10
iteration_num = 30
label_num = 4

# all used distributions
topic_word = 0*np.ones([1, 1])
# topic_word_list = 0*np.ones([1, 1, 1])
# word_topic_vectors = np.array(topic_word_list).T
doc_topic = 0*np.ones([1, 1])
#words_co_topic_list =  0*np.ones([1, 1, 1])
docs_list = []
doc_topic_distributions = []
topic_word_distributions = []
topic_word_distribution = []

perplexities = []
per_list = []
st= 0
ed= 0
total_time = 0
# init value
#eta = [1/3, 1/3, 1/3]

stop_file = open('stopwords2.txt', 'r')
readtext = stop_file.read()
stop_list = readtext.split('\n')



# create dictionary for training data
def create_dictionary_s(data, path):
    global word_index, index_word
    if not os.path.exists(path):
        os.makedirs(path)
    for doc in data:
        for w in doc:
            if w not in word_index:
                word_index[w] = len(word_index)
    index_word = dict(zip(word_index.values(), word_index.keys()))
    f = open(path + str(model_name)+'word_index.txt','w')
    f.write(str(word_index))
    f.close()
    f = open(path + str(model_name)+'index_word.txt','w')
    f.write(str(index_word))
    f.close()
# create dictionary for training data
def create_dictionary(data):
    global word_index, index_word
    for doc in data:
        for w in doc:
            if w not in word_index:
                word_index[w] = len(word_index)
    index_word = dict(zip(word_index.values(), word_index.keys()))


# compute the co-occurance matrix of context word pairs
#def compute_words_co_topic_list(c_len):
#    global words_co_topic_list
#    for d in docs_list:
#        for i in range(0, len(d)):
#            bottom = max(i - c_len, 0)
#            upper = min(i + c_len + 1, len(d))
#            for j in range(bottom, upper):
#                if(i != j):
#                    words_co_topic_list[d[j][1]][d[i][0]][d[j][0]] += 1

# topic assignment based on a topic distribution                    
def get_a_topic(doc_topic_distribution):
    topics = np.random.multinomial(1, doc_topic_distribution)
    topic = -1
    for i in range(0, len(topics)):
        if topics[i] > 0:
            topic = i
            break
    return topic

# initialization of all distributions
def initialize_distributions():
    global doc_topic_distributions, topic_word_distributions, topic_word_distribution, word_topic_vectors
    doc_topic_distributions.clear()
    topic_word_distributions.clear()
    topic_word_distribution.clear()
    for i in range(0, docs_num):
        doc_topic_distributions.append(1./topic_num*np.ones([topic_num]))
        topics_pdf = [] 
        for j in range(0, topic_num):
            topics_pdf.append(1./words_num*np.ones([words_num]))
        topic_word_distributions.append(topics_pdf)
    for i in range(0, topic_num):
        topic_word_distribution.append(1./words_num*np.ones([words_num]))
    # word_topic_vectors = np.array(topic_word_list).T
    return

# malloc the memories topic assignments of each word for each document
def initial_docs_list():
    global data, docs_list
    docs_list.clear()
    for doc in data:
         docs_list.append(np.ones([len(doc), 2], dtype = np.int64))
    return

# initialization of topic assignments for each word in each document
def initialize_values_docs_list():
    global docs_list
    for d in range(0, len(data)):
        for w in range(0, len(data[d])):
            docs_list[d][w] = [word_index[data[d][w]], get_a_topic(doc_topic_distributions[d])]
    return

# initialization of eta for each word in each document
def initialize_etas():
    global etas
    etas.clear()
    for doc in data:
         etas.append(1/2*np.ones([len(doc), 2], dtype = np.uint8))
    return

# compute topics for each document
def compute_doc_topic():
    global doc_topic
    doc_topic = np.array(doc_topic)
    doc_topic = 0*doc_topic
    for i in range(len(docs_list)):
        for j in range(0, len(docs_list[i])):
            doc_topic[i][docs_list[i][j][1]] += 1

# compute the topics for document d
def compute_doc_topic_doc(d):
    global doc_topic
    doc_topic[d] = np.array(doc_topic[d])
    doc_topic[d] = 0*doc_topic[d]
    for j in range(0, len(docs_list[d])):
        doc_topic[d][docs_list[d][j][1]] += 1

# compute topic-word distributions        
def compute_topic_word():
    global topic_word
    topic_word = np.array(topic_word)
    topic_word = 0*topic_word
    for i in range(len(docs_list)):
        for j in range(0, len(docs_list[i])):
            topic_word[docs_list[i][j][1]][docs_list[i][j][0]] += 1
    return

# computer topic word distribution of document d
# def compute_topic_word_list_doc(d):  
#     global docs_list
#     topic_word_list[d] = np.array(topic_word_list[d])
#     topic_word_list[d] = 0*topic_word_list[d]
#     for i in range(len(docs_list)):
#         for j in range(0, len(docs_list[i])):
#             topic_word_list[d][docs_list[i][j][1]][docs_list[i][j][0]] += 1
#     return

# get the number of word w assigned by topic k in document d
def get_n_d_k(d, w, k):
    n_d_k = 0
    for i in range(0, len(docs_list[d])):
        if(i != w and docs_list[d][i][1]- k == 0):
            n_d_k += 1
    return n_d_k

# get the number of word w assigned by topic k in all documents except the current one
def get_n_w_k(d, w, k):
    n_w_k = 0
    if(docs_list[d][w][1] - k == 0 and topic_word[k][docs_list[d][w][0]] > 0):
        n_w_k = topic_word[k][docs_list[d][w][0]] - 1
    else:
        n_w_k = topic_word[k][docs_list[d][w][0]]
    return n_w_k

# get the number of word w in corpus
def get_n_w(d, w):
    n_w = 0
    n_w = topic_word[:,docs_list[d][w][0]].sum()
    return n_w

# get the number context words of word w in corpus
def get_n_w_context(context):
    n_w = 0
    for w in context:
        n_w += topic_word[:,w].sum()
    return n_w

# get the number context words of word w in corpus
def get_n_k_w_context(context, k):
    n_w = 0
    for w in context:
        n_w += topic_word[k,w].sum()
    return n_w

# get the number of words assigned by topic k in document d 
def get_total_n_k(d, w, k):
    total_n_k = np.sum(topic_word[k])
    if(docs_list[d][w][1] - k == 0):
        total_n_k = total_n_k - 1
    return total_n_k

# get context words of given word w2 in a specific document
def get_context_num_w2(text, w1, w2, k, c_len):
    indexes = [x for x,a in enumerate(text) if a[0] == w1]
    w2_list = []
    for i in indexes:
        bottom = max(i - c_len, 0)
        upper = min(i + c_len + 1, len(text))
        for j in range(bottom, upper):
            if(text[j][0] == w2 and text[j][1] == k and j!= i and j not in w2_list):
                w2_list.append(j)
    return len(w2_list)

# get context words of given word w2 in all documents
def get_context_num_all(text, w1, k, c_len):
    indexes = [x for x,a in enumerate(text) if a[0] == w1]
    w_list = []
    for i in indexes:
        bottom = max(i - c_len, 0)
        upper = min(i + c_len + 1, len(text))
        for j in range(bottom, upper):
            if(text[j][1] == k and j!= i and j not in w_list):
                w_list.append(j)
    return len(w_list)

def get_context(d, w, c_len):
    bottom = max(w - c_len, 0)
    upper = min(w + c_len + 1, len(docs_list[d]))
    result = []
    for w in range(bottom, upper):
        if(docs_list[d][w][0] not in result):
            result.append(docs_list[d][w][0])
    return result
    

def compute_dominator(context_words, k, c_len):
    result = 0
    for doc in docs_list:
        for w1 in context_words:
            result += get_context_num_all(doc, w1, k, c_len)
    return result

def compute_numerator(context_words, w2, k, c_len):
    result = 0
    for doc in docs_list:
        for w1 in context_words:
            result += get_context_num_w2(doc, w1, w2, k, c_len)
    return result

def compute_dominator2(context_words, w2, k, c_len):
    res = np.zeros([topic_num])
    for w in context_words:
        res += words_co_topic_list[:,w,w2]
    return res.sum()

def compute_numerator2(context_words, w2, k, c_len):
    res = 0
    for w in context_words:
        res += words_co_topic_list[k][w][w2]
    return res

# recompute topic distribution for word w
def recompute_w_topic_distribution(d, w):
    new_topic_distribution = np.ones([topic_num])
    for topic in range(0, topic_num):
        n_d_k = get_n_d_k(d, w, topic)
        n_w_k = get_n_w_k(d, w, topic)
        total_n_k = get_total_n_k(d, w, topic)
        context_words = get_context(d, w, context_len)
#        n_w = get_n_w(d,w)
#        n_context = get_n_w_context(context_words)
        n_k_context = get_n_k_w_context(context_words, topic)
#        numerator = compute_numerator2(context_words, docs_list[d][w][0], topic, context_len)
        
#        baysian_dominator = (n_d_k + alpha)/(len(docs_list[d]) + topic_num*alpha) + (n_k_context + beta)/(n_context + topic_num*beta) + n_w_k/(n_w + topic_num*beta)
#        dominator = compute_dominator2(context_words, w, topic, context_len)
#        etas[d][w][0] = (n_d_k + alpha)/len(docs_list[d]) / baysian_dominator
#        etas[d][w][1] = (n_k_context + beta)/(n_context + topic_num*beta) / baysian_dominator
#        etas[d][w][2] = n_w_k/(n_w + topic_num*beta) / baysian_dominator
        p_d_w_k = (n_d_k + alpha)*( etas[d][w][0]*(n_k_context + beta)/(len(context_words)*(total_n_k + words_num*beta))+ etas[d][w][1]*(n_w_k + beta)/(total_n_k + words_num*beta))
        new_topic_distribution[topic] = p_d_w_k
    new_topic_distribution = new_topic_distribution/new_topic_distribution.sum()   
    return new_topic_distribution

# recompute etas for word w in document d
def recompute_etas(d, w, topic):
    global etas
    n_w_k = get_n_w_k(d, w, topic)
    total_n_k = get_total_n_k(d, w, topic)
    context_words = get_context(d, w, context_len)
    n_k_context = get_n_k_w_context(context_words, topic)
    baysian_dominator = (n_k_context + beta)/(len(context_words)*(total_n_k + words_num*beta)) + (n_w_k + beta)/(total_n_k + words_num*beta)
    etas[d][w][0] = (n_k_context + beta)/(len(context_words)*(total_n_k + words_num*beta)) / baysian_dominator
    etas[d][w][1] = (n_w_k + beta)/(total_n_k + words_num*beta) / baysian_dominator
    return


# gibbs_sampling iteration
def gibbs_sampling():
    global doc_topic_distributions, etas, st, ed, total_time
    st = 0
    ed = 0
    total_time = 0
    for d in range(0, len(docs_list)):        
        st = time.time()
        for w in range(0, len(docs_list[d])):
            new_pdf = recompute_w_topic_distribution(d, w)
#            print(new_pdf)
            new_topic = get_a_topic(new_pdf)
            docs_list[d][w][1] = new_topic
            recompute_etas(d, w, new_topic)
        ed = time.time()
        total_time += ed - st

# recompute all distribuiotns            
def recompute_distributions():
#    compute_words_co_topic_list(context_len)
    for d in range(0, len(doc_topic)):
        doc_topic_distributions[d] = (doc_topic[d] + alpha) / (np.sum(doc_topic[d]) + len(doc_topic[d]) * alpha)
    for topic in range(0, len(topic_word)):
        topic_word_distribution[topic] = (topic_word[topic] + beta) / (np.sum(topic_word[topic]) + len(topic_word[topic]) * beta)


def compute_perplexities():
    global doc_topic_distributions, docs_list, topic_word_distribution, words_co_topic_list, etas

    total = 0
    total_num = 0
    for d in range(0, len(docs_list)):
        for v in range(0, len(docs_list[d])):
            total_t = 0
            for k in range(0, len(topic_word_distribution)):
                w = docs_list[d][v][0]
                context_words = get_context(d, v, context_len)
                p_context = 0
                for cw in context_words:
                    p_context += topic_word_distribution[k][cw]
                p_d_w_k = etas[d][v][0]*p_context/len(context_words)+ etas[d][v][1]* topic_word_distribution[k][w]
                theta_d_k = doc_topic_distributions[d][k]
#                theta_d_k = doc_topic_distributions[d][k]
                total_t += theta_d_k*p_d_w_k
            total_num += 1.0
            total += (-1)*math.log(total_t)
    
    return math.exp(total / total_num) 
        
        
def parameter_estimation():
    global save_path
    per_list.clear()
    print(model_name)
#    per_list.append(compute_perplexities())
    for i in range(0, iteration_num):    
        gibbs_sampling()
        print(model_name + "_Iteration" , i, "----TIME----:", total_time)
        recompute_distributions()
        compute_doc_topic()
        compute_topic_word()     
        per_list.append(compute_perplexities())
        temp_save(save_path)
    return

def temp_save(path):
    if not os.path.exists(path):
        os.makedirs(path)
#    print(path)
    LDA_docs_list = np.array(docs_list) 
    LDA_doc_topic_distributions = np.array(doc_topic_distributions)
    LDA_topic_word_distribution = np.array(topic_word_distribution)
    LDA_topic_word = np.array(topic_word)
    np.save(path + str(model_name)+"topic_word"+str(topic_num)+".npy", LDA_topic_word)
    np.save(path + str(model_name)+"docs_list"+str(topic_num)+".npy", LDA_docs_list)
    np.save(path + str(model_name)+"doc_topic_distributions_"+str(topic_num)+".npy", LDA_doc_topic_distributions)
    np.save(path + str(model_name)+"topic_word_distribution_"+str(topic_num)+".npy", LDA_topic_word_distribution)
#    np.save(path + str(model_name)+"words_co_topic_list_"+str(topic_num)+".npy", words_co_topic_list)
    np.save(path + str(model_name)+"docs_list"+str(topic_num)+".npy", docs_list)
    np.save(path + str(model_name)+"eta_list_"+str(topic_num)+".npy", np.array(etas))
    LDA_per_list = np.array(per_list)
    np.save(path + str(model_name)+"per_list"+str(topic_num)+".npy", LDA_per_list)
    return 

 
def save_result(path):
    if not os.path.exists(path):
        os.makedirs(path)
#    print(path)
    LDA_docs_list = np.array(docs_list) 
    LDA_doc_topic_distributions = np.array(doc_topic_distributions)
    LDA_topic_word_distribution = np.array(topic_word_distribution)
    LDA_topic_word = np.array(topic_word)
    np.save(path + str(model_name)+"topic_word"+str(topic_num)+".npy", LDA_topic_word)
    np.save(path + str(model_name)+"docs_list"+str(topic_num)+".npy", LDA_docs_list)
    np.save(path + str(model_name)+"doc_topic_distributions_"+str(topic_num)+".npy", LDA_doc_topic_distributions)
    np.save(path + str(model_name)+"topic_word_distribution_"+str(topic_num)+".npy", LDA_topic_word_distribution)
#    np.save(path + str(model_name)+"words_co_topic_list_"+str(topic_num)+".npy", words_co_topic_list)
    np.save(path + str(model_name)+"docs_list"+str(topic_num)+".npy", docs_list)
    np.save(path + str(model_name)+"eta_list_"+str(topic_num)+".npy", np.array(etas))
    LDA_per_list = np.array(per_list)
    np.save(path + str(model_name)+"per_list"+str(topic_num)+".npy", LDA_per_list)
    f = open('word_index.txt','w')
    f.write(str(word_index))
    f.close()
    f = open('index_word.txt','w')
    f.write(str(index_word))
    f.close()
    return 

def initialize():
    global topic_word, doc_topic, words_co_topic_list, etas
    print("initializing...")
    topic_word = 0*np.ones([topic_num, words_num])
    doc_topic = 0*np.ones([docs_num, topic_num])
    # topic_word_list = 0*np.ones([docs_num, topic_num, words_num], dtype = np.uint16)
#    words_co_topic_list =  0*np.ones([topic_num, words_num, words_num])
    initialize_distributions()
    initial_docs_list()
    initialize_values_docs_list()
    initialize_etas()
    compute_doc_topic()
    compute_topic_word()
    # for i in range(0, docs_num):
    #     print("compute_topic_word_list_doc", str(i))
    #     compute_topic_word_list_doc(i)
    print("initialization finished")
    return

# get topic vector of word w in document d, where w refers to the index of wrod in dictionary and wv refers to its index in document
def get_word_vector_in_document(d, w, wv, d_list, c_len, doc_topic, topic_word, etas):
    global docs_list
    docs_list = d_list
    context_vector = np.zeros(len(topic_word[:,0]))
    w_vector = topic_word[:,w]
    context_words = get_context(d, wv, c_len)
    for v in context_words:
        context_vector += topic_word[:,v]
    context_vector = context_vector / len(context_words)
    d_w_vector_1 = etas[d][wv][1]*context_vector + etas[d][wv][0]*w_vector
    d_w_vector_2 = w_vector
    return d_w_vector_1, d_w_vector_2, context_words

# compute vectors of given word text w
def get_vectors_of_word_text(data, dict_w_index, w_text, d_list, c_len, doc_topic, topic_word, etas):
    vectors_w_1 = []
    vectors_w_2 = []
#    vectors_w_3 = []
    context_words_list = []
    w_index = dict_w_index[w_text]
    all_labels = np.load("SE_DATA_LABLE.npy", allow_pickle= True)
    labels = []
    for d in range(0, len(data)):
        doc = np.array(data[d])
        d_w_indexes = np.argwhere(doc == w_text)
        for i in d_w_indexes:
            labels.append(all_labels[d][1:])
            v_1, v_2, context_words = get_word_vector_in_document(d, w_index, i[0], d_list, c_len, doc_topic, topic_word, etas)
            vectors_w_1.append([d, v_1])
            vectors_w_2.append([d, v_2])
            context_words_list.append(context_words)
    return vectors_w_1, vectors_w_2, context_words_list, labels
    
# get etas of given word text w
def get_etas_of_word(data, w_text, etas):
    etas_w = []
    for d in range(0, len(data)):
        doc = np.array(data[d])
        d_w_indexes = np.argwhere(doc == w_text)
        for i in d_w_indexes:
            etas_w.append([d, etas[d][i[0]]])
    return etas_w

def run(t_data, start, end_iter, iterations, save_p, clip, c_len, palpha, pbeta):
    global save_path, topic_num, iteration_num, data_clip, data, docs_num, topic_num, words_num, etas, context_len, alpha, beta, etas
    data=t_data
    alpha = palpha
    beta = pbeta 
    context_len = c_len
    save_path = save_p
    data_clip = clip
    topic_num = start
    iteration_num = iterations
    
    create_dictionary_s(data,save_p)
    docs_num = len(data)
    topic_num = start
    words_num = len(word_index)
    for i in range(0, end_iter):
        initialize()
        parameter_estimation()
        save_result(save_path)
        topic_num += 2
        np.save("LDA_runtime_"+str(data_clip)+".npy", total_time)
    return 
