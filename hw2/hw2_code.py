###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################

import os
from parsivar import Normalizer
from parsivar import Tokenizer
from parsivar import FindStems
from gensim.models import Word2Vec
import numpy as np
import json
from gensim.models import TfidfModel
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from sklearn.cluster import KMeans
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.decomposition import TruncatedSVD

from google.colab import drive
drive.mount('/content/drive')

# change directory
os.chdir("/content/drive/My Drive/NLP")

# load dataset
# 
count = 0
data = []
with open("Hamshahri.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        count += 1
        data.append(line.strip())
print(len(data))

# list of stop words
stop_words = ['که','از','به','در','برای','زیرا','همچنین','آن','این','و','شد','اس','کرد','است','هست','را','با','نیست','ای','الا','اما','اگر','می','خود','ای','نیز','وی','هم','ما','نمی','پیش','همه','بی','من'
,'چه','هیچ','ولی','حتی','توسط','شما','تو','او','ایشان','هنوز','البته','فقط','شاید','شان','روی','مانند','کجا','کی','چطور','چگونه','مگر','چندین','کدام','چیزی','چیز','دیگر','دیگری','مثل','بلی','همین']

# preprocessing
# 
x = []
y = []
for i in range(len(data)):
    s = data[i].split("@@@@@@@@@@")
    y.append(s[0])
    # n = Normalizer().normalize(s[1])
    # words = Tokenizer().tokenize_words(n.replace('\u200c', ''))
    # tokens = []
    # for word in words:
    #     w = FindStems().convert_to_stem(word)
    #     w1 = w.split("&")[0]
    #     if not w1 in stop_words :
    #         tokens.append(w1)
    # x.append(tokens)
# saving preprocessed data in file
# 
# with open("x_json.txt", "w") as f:
#     json.dump(x, f)

#  word2vec model training and saving it
# 
model = Word2Vec(x,min_count=1,size=300,workers=4, window=3,sg = 1)
model.save("skip_gram_x.model")

# laoding preprocessed data 
# 
with open('x_json.txt') as f:
    lst = json.load(f)

# loading trained word2vec model
skip_gram = Word2Vec.load("skip_gram_x.model")

# part 2_a
# 
docs_vec = np.zeros([8599,300])
for i in range(8599):
    count = 0 
    for w in lst[i]:
        docs_vec[i] += skip_gram.wv[w]
        count += 1 
    docs_vec[i] /= count 
    
# saving docs_vec
np.savetxt("docs_vec_a.txt",docs_vec)

# part 2_b
# 

def Convert(tup, di):
    for a, b in tup:
        di.setdefault(a, []).append(b)
    return di


# fit dictionary
dct = Dictionary(lst)  
cor = [dct.doc2bow(line) for line in lst]
tf_idf_model = TfidfModel(cor)


dictionary_docs = {}
for i in range(8599):
    dictionary = {}
    dictionary_docs.update({i: Convert(tf_idf_model[cor[i]], dictionary)})

# tf_idf representation of docs
docs_vec = np.zeros([8599,300])
for i in range(8599):
    count = 0 
    for w in lst[i]:
        tf_idf = dictionary_docs.get(i).get(dct.token2id[w])[0]
        docs_vec[i] += tf_idf * skip_gram.wv[w]
        count += tf_idf
    docs_vec[i] /= count
    
# saving docs_vec
np.savetxt("docs_vec_b.txt",docs_vec)

# part 2_c 
# 
with open('hamshahri.fa.text.300.vec') as f:
    lines = f.readlines()

dict_vec_rep = {}
for i in range(len(lines)):
    s = lines[i].split(" ")
    dict_vec_rep.update({s[0]:s[1:301]})

docs_vec = np.zeros([8599,300])
for i in range(8599):
    count = 0 
    for w in lst[i]:
        if dict_vec_rep.get(w) is not None:
            vec = list(map(float, dict_vec_rep.get(w)))
            docs_vec[i] = docs_vec[i] + np.array(vec)
            count += 1 
    docs_vec[i] /= count 

# saving docs_vec
np.savetxt("docs_vec_c.txt",docs_vec)

# part 2_d
# 

def Convert(tup, di):
    for a, b in tup:
        di.setdefault(a, []).append(b)
    return di

with open('hamshahri.fa.text.300.vec') as f:
    lines = f.readlines()

dict_vec_rep = {}
for i in range(len(lines)):
    s = lines[i].split(" ")
    dict_vec_rep.update({s[0]:s[1:301]})

dct = Dictionary(lst)  # fit dictionary
cor = [dct.doc2bow(line) for line in lst]
tf_idf_model = TfidfModel(cor)
dictionary_docs = {}
for i in range(8599):
    dictt = {}
    dictionary_docs.update({i: Convert(tf_idf_model[cor[i]], dictt)})
docs_vec = np.zeros([8599,300])
for i in range(8599):
    count = 0 
    for w in lst[i]:
        if dict_vec_rep.get(w) is not None:
            tf_idf = dictionary_docs.get(i).get(dct.token2id[w])[0]
            vec = list(map(float, dict_vec_rep.get(w)))
            docs_vec[i] += tf_idf * np.array(vec)
            count += tf_idf
    docs_vec[i] /= count

# saving docs_vec
np.savetxt("docs_vec_d.txt",docs_vec)

# part 2_e
# 
dct = Dictionary(lst)
word_doc_M = np.zeros([len(dct),8599])
i = 0
for line in lst: 
    list_tf = dct.doc2bow(line)
    for tf in list_tf:
        word_doc_M[tf[0]][i] = tf[1]
    i += 1
print(word_doc_M.shape)
svd = TruncatedSVD(n_components=300, n_iter=7, random_state=42)
svd.fit(word_doc_M)
docs_vec = np.transpose(svd.components_)
np.savetxt("docs_vec_e.txt",docs_vec)

# part 3 --> clustering 
# 
docs_vec = np.loadtxt("docs_vec_e.txt")
kmeans = KMeans(n_clusters=5, random_state=1).fit(docs_vec)
pred_y = kmeans.labels_

# confusion matrix
name_of_clusters = confusion_matrix(y,pred_y)
print(name_of_clusters)
# convert number of clusters to their appropriate name
y_pred = []
for i in range(8599):
    y_pred.append(name_of_clusters.get(pred_y[i]))

# metrics
print("accuracy: %f " %accuracy_score(y,y_pred))
print("f1_score: %f " %f1_score(y,y_pred,average= 'macro'))
print("NMI: %f" %normalized_mutual_info_score(y,y_pred))

# part d_1
# 
# fit dictionary
dct = Dictionary(lst)  
cor = [dct.doc2bow(line) for line in lst]
lda = LdaModel(cor, num_topics=5)

# part d_2
#
docs_per_topic = np.zeros([5,8599])
for doc_id, doc_bow in enumerate(cor):
    # ...get its topics...
    doc_topics = lda.get_document_topics(doc_bow)
    # ...& for each of its topics...
    
    for topic_id, score in doc_topics:
        # ...add the doc_id & its score to the topic's doc list
        docs_per_topic[topic_id,doc_id] = score
pred_y = []
for i in range(8599):
    pred_y.append(np.argmax(docs_per_topic[:,i]))

# confusion matrix
name_of_clusters = confusion_matrix(y,pred_y)
print(name_of_clusters)
# convert number of clusters to their appropriate name
y_pred = []
for i in range(8599):
    y_pred.append(name_of_clusters.get(pred_y[i]))

# metrics
print("accuracy: %f " %accuracy_score(y,y_pred))
print("f1_score: %f " %f1_score(y,y_pred,average= 'macro'))
print("NMI: %f" %normalized_mutual_info_score(y,y_pred))

def confusion_matrix(y_true,y_pred):
    dict_labels = {'اجتماعی':0,'ورزش':1,'ادب و هنر':2,'اقتصاد':3,'سیاسی':4}
    # names = list(dict_labels.keys())
    num2name = {}
    # name2num = {}
    for i in range(5):
        t_label = np.zeros([5,])
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                t_label[dict_labels.get(y_true[j])] += 1
        class_name = list(dict_labels.keys())[np.argmax(t_label)]
        # if  name2num.get(class_name) is None:
        num2name.update({i: class_name})
        # name2num.update({class_name: i})
        # names.remove(class_name)
        # else: 
        #     bad_cluster = i
    # num2name.update({bad_cluster: names[-1]})
    # name2num.update({names[-1]: bad_cluster})
    # confusion_M = np.zeros([5,5])
    # for label,y in zip(y_pred,y_true):
    #     j = name2num.get(y)
    #     confusion_M[label,j] += 1
    # print(confusion_M)
    # total = list(map(sum, confusion_M))
    # new_M = np.empty([5, 5])
    # for i in range(5):
    #     for j in range(5):
    #         new_M[i, j] = confusion_M[i, j] / total[i] * 100
    # ax = plt.axes()
    # df_cm = pd.DataFrame(new_M, list(name2num.keys()),list(name2num.keys()))
    # sn.set(font_scale=1)
    # sn.heatmap(df_cm, cmap="RdYlGn", annot=True, annot_kws={"size": 10},ax=ax,fmt='g')
    # ax.set_xlabel('predict')
    # ax.set_ylabel('true')
    # ax.set_title('Normalized Confusion Matrix (%)')
    # plt.show()

    return num2name