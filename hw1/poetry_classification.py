###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn import metrics
from parsivar import FindStems
import codecs
import seaborn as sn
from parsivar import Normalizer




unigram_train = {}
bigram_train = {}
N = 0
color = {'moulavi': 'red', 'amir': 'green', 'bahar': 'pink', 'ghaani': 'blue', 'sanaee': 'orange', 'khosro': 'm'}
N_poet = {'moulavi': 0, 'amir': 0, 'bahar': 0, 'ghaani': 0, 'sanaee': 0, 'khosro': 0}
prob = {'moulavi': 0, 'amir': 0, 'bahar': 0, 'ghaani': 0, 'sanaee': 0, 'khosro': 0}


def tokenizer_a(filename,mode):
    # data = np.loadtxt(filename, encoding='utf8', dtype=str, delimiter='\n')
    data = []
    my_normalizer = Normalizer()
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(my_normalizer.normalize(line.strip()))
    unigram = {}
    bigram = {}
    counter = 0
    my_stemmer = FindStems()
    for i in range(int(len(data) / 3)):
        # token_list = str.split(data[3 * i + 1]) + str.split(data[3 * i + 2])
        if mode == 1 :
            token_list = data[3 * i + 1].split() + data[3 * i + 2].split()
        elif mode == 2 :
            token_list = list(set(data[3 * i + 1].split() + data[3 * i + 2].split()))
        for t in token_list:
            # t = my_stemmer.convert_to_stem(t)
            counter += 1
            if unigram.get(t) is None:
                unigram.update({t: 1})
            else:
                value = unigram.get(t) + 1
                unigram.update({t: value})
    for i in range(int(len(data) / 3)):
        # token_list = str.split(data[3 * i + 1])
        if mode == 1:
            token_list = data[3 * i + 1].split()
        elif mode == 2 :
            token_list = list(set(data[3 * i + 1].split()))
        for j in range(len(token_list) - 1):
            # b = my_stemmer.convert_to_stem(token_list[j]) + ' ' + my_stemmer.convert_to_stem(token_list[j + 1])
            b = token_list[j] + ' ' + token_list[j + 1]
            if bigram.get(b) is None:
                bigram.update({b: 1})
            else:
                value = bigram.get(b) + 1
                bigram.update({b: value})
        # token_list = str.split(data[3 * i + 2])
        if mode == 1:
            token_list = data[3 * i + 2].split()
        elif mode == 2:
            token_list = list(set(data[3 * i + 2].split()))
        for k in range(len(token_list) - 1):
            # b = my_stemmer.convert_to_stem(token_list[k]) + ' ' + my_stemmer.convert_to_stem(token_list[k + 1])
            b = token_list[k] + ' ' + token_list[k+1]
            if bigram.get(b) is None:
                bigram.update({b: 1})
            else:
                value = bigram.get(b) + 1
                bigram.update({b: value})
    # f = open('dict.txt', 'w', encoding='utf8')
    # f.write(str(unigram))
    # f.close()
    return unigram, bigram,counter


def tokenizer_b(filename,mode):
    # data = np.loadtxt(filename, encoding='utf8', dtype=str, delimiter='\n')
    data = []
    my_normalizer = Normalizer()
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(my_normalizer.normalize(line.strip()))
    unigram = {}
    bigram = {}
    my_stemmer = FindStems()
    for i in range(int(len(data) / 3)):
        poet = data[3 * i]
        temp = prob.get(poet) + 1
        prob.update({poet:temp})
        counter = 0
        if unigram.get(poet) is None:
            # token_list = str.split(data[3 * i + 1]) + str.split(data[3 * i + 2])
            if mode == 1:
                token_list = data[3 * i + 1].split() + data[3 * i + 2].split()
            elif mode == 2:
                token_list = list(set(data[3 * i + 1].split() + data[3 * i + 2].split()))
            token_dict = {}
            for t in token_list:
                # t = my_stemmer.convert_to_stem(t)
                counter += 1
                if token_dict.get(t) is None:
                    token_dict.update({t: 1})
                else:
                    value = token_dict.get(t) + 1
                    token_dict.update({t: value})
            unigram.update({poet: token_dict})
        else:
            # token_list = str.split(data[3 * i + 1]) + str.split(data[3 * i + 2])
            if mode == 1:
                token_list = data[3 * i + 1].split() + data[3 * i + 2].split()
            elif mode == 2:
                token_list = list(set(data[3 * i + 1].split() + data[3 * i + 2].split()))
            token_dict = unigram.get(poet)
            for t in token_list:
                counter += 1
                # t = my_stemmer.convert_to_stem(t)
                if token_dict.get(t) is None:
                    token_dict.update({t: 1})
                else:
                    value = token_dict.get(t) + 1
                    token_dict.update({t: value})
            unigram.update({poet: token_dict})
        n = N_poet.get(poet) + counter
        N_poet.update({poet:n})
    for i in range(int(len(data) / 3)):
        poet = data[3 * i]
        if bigram.get(poet) is None:
            # token_list = str.split(data[3 * i + 1])
            if mode == 1:
                token_list = data[3 * i + 1].split()
            elif mode == 2:
                token_list = list(set(data[3 * i + 1].split()))
            token_dict = {}
            for j in range(len(token_list) - 1):
                # b = my_stemmer.convert_to_stem(token_list[j]) + ' ' + my_stemmer.convert_to_stem(token_list[j + 1])
                b = token_list[j] + ' ' + token_list[j + 1]
                if token_dict.get(b) is None:
                    token_dict.update({b: 1})
                else:
                    value = token_dict.get(b) + 1
                    token_dict.update({b: value})
                bigram.update({poet: token_dict})
            # token_list = str.split(data[3 * i + 2])
            if mode == 1:
                token_list = data[3 * i + 2].split()
            elif mode == 2:
                token_list = list(set(data[3 * i + 2].split()))
            for k in range(len(token_list) - 1):
                # b = my_stemmer.convert_to_stem(token_list[k]) + ' ' + my_stemmer.convert_to_stem(token_list[k + 1])
                b = token_list[k] + ' ' + token_list[k + 1]
                if token_dict.get(b) is None:
                    token_dict.update({b: 1})
                else:
                    value = token_dict.get(b) + 1
                    token_dict.update({b: value})
                bigram.update({poet: token_dict})
        else:
            # token_list = str.split(data[3 * i + 1])
            if mode == 1:
                token_list = data[3 * i + 1].split()
            elif mode == 2:
                token_list = list(set(data[3 * i + 1].split()))
            token_dict = bigram.get(poet)
            for s in range(len(token_list) - 1):
                # b = my_stemmer.convert_to_stem(token_list[s]) + ' ' + my_stemmer.convert_to_stem(token_list[s + 1 ])
                b = token_list[s] + ' ' + token_list[s + 1]
                if token_dict.get(b) is None:
                    token_dict.update({b: 1})
                else:
                    value = token_dict.get(b) + 1
                    token_dict.update({b: value})
                bigram.update({poet: token_dict})
            # token_list = str.split(data[3 * i + 2])
            if mode == 1:
                token_list = data[3 * i + 2].split()
            elif mode == 2:
                token_list = list(set(data[3 * i + 2].split()))
            for t in range(len(token_list) - 1):
                # b = my_stemmer.convert_to_stem(token_list[t]) + ' ' + my_stemmer.convert_to_stem(token_list[t + 1])
                b = token_list[t] + ' ' + token_list[t + 1]
                if token_dict.get(b) is None:
                    token_dict.update({b: 1})
                else:
                    value = token_dict.get(b) + 1
                    token_dict.update({b: value})
                bigram.update({poet: token_dict})
    # f = open('dict.txt', 'w', encoding='utf8')
    # f.write(str(bigram))
    # f.close()
    # print(prob)
    # for p in prob.keys():
    #     value = prob.get(poet) / 94505
    #     prob.update({p:value})
    return unigram, bigram

def probability(poet, w, delta, n_gram, B, model):
    if model == 1:
        if n_gram == 1:
            Pbg = 1 / len(unigram_train)
            if unigram_train.get(w) is None:
                return delta * B * Pbg / N
            else:
                return (max((unigram_train.get(w) - delta), 0) / N) + (
                        delta * B * Pbg / N)
        elif n_gram == 2:
            w2, w1 = str.split(w)
            if unigram_train.get(w2) is None:
                return probability("", w2, delta, 1, B, model)
            else:
                Pbg = unigram_train.get(w2) / N
            if unigram_train.get(w1) is None:
                return probability("",w2,delta,1,B,model)
            else:
                q = unigram_train.get(w1)
                if bigram_train.get(w) is None:
                    return delta * B * Pbg / q
                else:
                    return (max((bigram_train.get(w) - delta), 0) / q) + (
                            delta * B * Pbg / q)

    elif model == 2:
        if n_gram == 1:
            Pbg = 1 / len(unigram_train.get(poet))
            if unigram_train.get(poet).get(w) is None:
                return delta * B * Pbg / N_poet.get(poet)
            else:
                return (max((unigram_train.get(poet).get(w) - delta), 0) /N_poet.get(poet)) + (
                        delta * B * Pbg / N_poet.get(poet))
        elif n_gram == 2:
            w2, w1 = str.split(w)
            if unigram_train.get(poet).get(w2) is None:
                return probability(poet, w2, delta, 1, B, model)
            else:
                Pbg = unigram_train.get(poet).get(w2) / N_poet.get(poet)
            if unigram_train.get(w1) is None:
                return probability(poet,w2,delta,1,B,model)
            else:
                q = unigram_train.get(poet).get(w1)
            if bigram_train.get(w) is None:
                return delta * B * Pbg / q
            else:
                return (max((bigram_train.get(poet).get(w) - delta), 0) / q) + (
                        delta * B * Pbg / q)



def perplexity(filename, model, n_gram):
    data = []
    my_normalizer = Normalizer()
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(my_normalizer.normalize(line.strip()))
    # sigm = np.linspace(0.1, 1, 10)
    sigm = [1]
    if model == 1:
        perplexity_array = np.ones([int(len(data)/3),len(sigm)])
        for i in range(int(len(data) / 3)):
            if n_gram == 1:
                token_list = data[3 * i + 1].split() + data[3 * i + 2].split()
                B = len(unigram_train)
            elif n_gram == 2:
                B = len(bigram_train)
                temp = data[3 * i + 1].split()
                token_list = []
                for s in range(len(temp) - 1) :
                    token_list.append(temp[s]+' ' + temp[s+1])
                temp = data[3 * i + 2].split()
                for v in range(len(temp) - 1):
                    token_list.append(temp[v] + ' '  + temp[v + 1])
            for e in token_list:
                for j in range(len(sigm)):
                    perplexity_array[i][j] *= (1 / probability("",e , sigm[j], n_gram, B, model)) ** (1 / int(len(token_list)))
        y = []
        for k in range(len(sigm)):
            y.append(np.mean(perplexity_array[:,k]))
        print(y)
        plt.plot(sigm, y,'o--', color='orange')
        plt.xlabel("delta")
        plt.ylabel("perplexity")
        plt.title(str(n_gram ) + ' gram ')
        plt.show()
    elif model == 2:
        perplexity_array = np.ones([int(len(data) / 3), len(sigm),6])
        for i in range(int(len(data) / 3)):
            p = list(color.keys()).index(data[3*i])
            if n_gram == 1:
                token_list = data[3 * i + 1].split() + data[3 * i + 2].split()
                B = len(unigram_train.get(data[3*i]))
            elif n_gram == 2:
                temp = data[3 * i + 1].split()
                token_list = []
                B = len(bigram_train.get(data[3*i]))
                for s in range(len(temp) - 1):
                    token_list.append(temp[s] + ' ' + temp[s + 1])
                temp = data[3 * i + 2].split()
                for v in range(len(temp) - 1):
                    token_list.append(temp[v] + ' ' + temp[v + 1])
            for e in token_list:
                for j in range(len(sigm)):
                    perplexity_array[i][j][p] *= (1 / probability(data[3*i], e, sigm[j], n_gram, B, model))** (1 / int(len(token_list)))
        y = np.empty([6,len(sigm)])
        for r in range(6):
            for k in range(len(sigm)):
                y[r,k] = np.mean(perplexity_array[:, k,r])
            plt.plot(sigm, y[r,:],'o--' ,color=color.get(list(color.keys())[r]), label=list(color.keys())[r])
        print(y)
        plt.xlabel("delta")
        plt.ylabel("perplexity")
        plt.title(str(n_gram) + ' gram ')
        plt.legend(loc='upper right')
        plt.show()

def information_gain():
    uni,bi,N = tokenizer_a("train.txt",2)
    uni_class,bi_class = tokenizer_b("train.txt",2)
    gain_dict = {}
    poets_quote = {}
    constant = 0
    for p in color.keys():
        constant += (prob.get(p)/94505)*math.log2((prob.get(p)/94505))
    for t in uni.keys():
        Nw = uni.get(t)
        Nnw = 94505 - Nw
        max_prob = -10
        max_poet = ""
        ig = 0
        ig2 = 0
        # print(t)
        for poet in color.keys():
            # print(poet)
            poet_quote = 0 - (prob.get(poet)/94505)*math.log((prob.get(poet)/94505),2)
            # print(poet_quote)
            if uni_class.get(poet).get(t) is None:
                Niw = 0
                Ninw = prob.get(poet)
            else:
                Niw = uni_class.get(poet).get(t)
                Ninw = prob.get(poet) - Niw
            if Niw != 0:
                ig += (Niw / Nw)*math.log((Niw/Nw),2)
                poet_quote += (Niw/Nw) * math.log((Niw/Nw),2) * (Nw / 94505)
            else:
                poet_quote = 0
            if Ninw != 0:
                ig2 += (Ninw/Nnw)*math.log((Ninw/Nnw),2)
                poet_quote += (Ninw / Nnw)*math.log((Ninw / Nnw),2) * (Nnw/94505)
            else:
                poet_quote = 0
            if poet_quote > max_prob:
                max_poet = poet
                max_prob = poet_quote
        poets_quote.update({t:max_poet})
        ig *= (Nw/94505)
        ig2 *= 1 - (Nw/94505)
        gain_dict.update({t:ig + ig2 - constant})
    gain_dict_sorted = {k: v for k, v in sorted(gain_dict.items(), key=lambda item: item[1])}
    keys = list(gain_dict_sorted.keys())
    output = {}
    for i in range(200):
        output.update({i+1:  {'word':keys[len(keys) - i - 1],'information gain':gain_dict_sorted.get(keys[len(keys) - i - 1]),'poet': poets_quote.get(keys[len(keys) - i - 1])}})
    columns = ('poet','information gain','word')
    df = pd.DataFrame(data = output,index=columns).T
    pd.options.display.max_rows = 200
    print(df)
    return (df.loc[:, 'word']).to_numpy()


def x_square():
    uni, bi, N = tokenizer_a("train.txt",mode=2)
    uni_class, bi_class = tokenizer_b("train.txt",mode=2)
    x2_dict = {}
    poets_quote = {}
    for w in uni.keys():
        x2 = 0
        max_prob = -100
        for poet in color.keys():
            if uni_class.get(poet).get(w) is None:
                Niw = 0
            else:
                Niw = uni_class.get(poet).get(w)
            Nninw = 94505 - prob.get(poet) - uni.get(w) + Niw
            Nniw =  uni.get(w) - Niw
            Ninw = prob.get(poet) -Niw
            poet_quote = (94505 * math.pow((Niw*Nninw - Ninw*Nniw),2))/((Niw+Ninw)*(Nniw+Nninw)*(Niw+Nniw)*(Ninw+Nninw)) * (prob.get(poet)/94505)
            x2 += poet_quote
            if poet_quote > max_prob:
                max_poet = poet
                max_prob = poet_quote
        poets_quote.update({w: max_poet})
        x2_dict.update({w:x2})
    x2_dict_sorted = {k: v for k, v in sorted(x2_dict.items(), key=lambda item: item[1])}
    keys = list(x2_dict_sorted.keys())
    output = {}
    for i in range(200):
        output.update({i: {'word': keys[len(keys) - i - 1],
                           'X_square': x2_dict_sorted.get(keys[len(keys) - i - 1]),
                           'poet': poets_quote.get(keys[len(keys) - i - 1])}})
    columns = ('poet', 'X_square', 'word')
    df = pd.DataFrame(data=output, index=columns).T
    pd.options.display.max_rows = 200
    print(df)
    # f = open('x2_dict.txt', 'w', encoding='utf8')
    # f.write(str(x2_dict_sorted))
    # f.close()
    return (df.loc[:,'word']).to_numpy()


def CP(w,poet):
    if unigram_train.get(poet).get(w) is None:
        return 0
    else:
        return unigram_train.get(poet).get(w) / (prob.get(poet)/94505)



def classification_without_feature_selection():
    data = []
    my_normalizer = Normalizer()
    with codecs.open('test.txt', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(my_normalizer.normalize(line.strip()))
    output = []
    y_true = []
    for i in range(int(len(data) / 3)):
        poet = data[3 * i]
        y_true.append(poet)
        token_list = data[3 * i + 1].split() + data[3 * i + 2].split()
        max_p = -10
        for poet in color.keys():
            p = 1
            for w in token_list:
                p *= CP(w,poet)
            p *= prob.get(poet)/94505
            if p> max_p:
                max_p = p
                max_poet = poet
        output.append(max_poet)
    print('f1_score , macro average is : ' ,metrics.f1_score(y_true,output,average='macro'))
    print('f1_score, micro average is : ' ,metrics.f1_score(y_true,output,average='micro'))
    print('precision, macro average is : ',metrics.precision_score(y_true,output,average='macro'))
    print('precision, micro average is : ',metrics.precision_score(y_true,output,average='micro'))
    print('recall, macro average is : ' ,metrics.recall_score(y_true,output,average='macro'))
    print('recall, micro average is : ' ,metrics.recall_score(y_true,output,average='micro'))


def classification_with_feature_selection():
    features = x_square()
    # features = information_gain()
    data = []
    my_normalizer = Normalizer()
    with codecs.open('test.txt', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(my_normalizer.normalize(line.strip()))
    output = []
    y_true = []
    for i in range(int(len(data) / 3)):
        poet = data[3 * i]
        y_true.append(poet)
        token_list = data[3 * i + 1].split() + data[3 * i + 2].split()
        max_p = -10
        for poet in color.keys():
            p = 1
            flag = False
            for w in token_list:
                if w in features:
                    p *= CP(w, poet)
                    flag = True
            if flag :
                p *= prob.get(poet)/94505
                if p > max_p:
                    max_p = p
                    max_poet = poet
        output.append(max_poet)
    print('f1_score , macro average is : ', metrics.f1_score(y_true, output, average='macro'))
    print('f1_score, micro average is : ', metrics.f1_score(y_true, output, average='micro'))
    print('precision, macro average is : ', metrics.precision_score(y_true, output, average='macro'))
    print('precision, micro average is : ', metrics.precision_score(y_true, output, average='micro'))
    print('recall, macro average is : ', metrics.recall_score(y_true, output, average='macro'))
    print('recall, micro average is : ', metrics.recall_score(y_true, output, average='micro'))


def bigram_classification():
    data = []
    my_normalizer = Normalizer()
    with codecs.open('test.txt', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(my_normalizer.normalize(line.strip()))
    output = []
    y_true = []
    for i in range(int(len(data) / 3)):
        poet = data[3 * i]
        y_true.append(poet)
        max_p = -10
        for poet in color.keys():
            B = len(bigram_train.get(poet))
            token_list = data[3 * i + 1].split()
            p = 1
            for t in range(len(token_list) - 1):
                p *= probability(poet, token_list[t]+' '+token_list[t+1], delta = 0.1, n_gram=2, B=B, model=2)
            token_list = data[3 * i + 2].split()
            for k in range(len(token_list) - 1):
                p *= probability(poet, token_list[k]+' '+token_list[k+1], delta = 0.1, n_gram=2, B=B, model = 2)
            p *= prob.get(poet)/94505
            if p > max_p:
                max_p = p
                max_poet = poet
        output.append(max_poet)
    print('f1_score , macro average is : ', metrics.f1_score(y_true, output, average='macro'))
    print('f1_score, micro average is : ', metrics.f1_score(y_true, output, average='micro'))
    print('precision, macro average is : ', metrics.precision_score(y_true, output, average='macro'))
    print('precision, micro average is : ', metrics.precision_score(y_true, output, average='micro'))
    print('recall, macro average is : ', metrics.recall_score(y_true, output, average='macro'))
    print('recall, micro average is : ', metrics.recall_score(y_true, output, average='macro'))
    return y_true,output


def confusion_matrix(y_true,y_pred):
    confusion_M = np.zeros([6,6])
    dict = {'moulavi': 0, 'amir': 1, 'bahar': 2, 'ghaani': 3, 'sanaee': 4, 'khosro': 5}
    for label,y in zip(y_pred,y_true):
        i = dict.get(label)
        j = dict.get(y)
        confusion_M[i,j] += 1
    # print(confusion_M)
    total = list(map(sum, confusion_M))
    new_M = np.empty([6, 6])
    for i in range(6):
        for j in range(6):
            new_M[i, j] = confusion_M[i, j] / total[i] * 100
    ax = plt.axes()
    df_cm = pd.DataFrame(new_M, ['moulavi','amir', 'bahar', 'ghaani', 'sanaee', 'khosro'],
                         ['moulavi','amir', 'bahar', 'ghaani', 'sanaee', 'khosro'])
    print(df_cm)

    sn.set(font_scale=1)
    sn.heatmap(df_cm, cmap="RdYlGn", annot=True, annot_kws={"size": 10},ax=ax,fmt='g')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    ax.set_title('Normalized Confusion Matrix (%)')
    plt.show()
    print("normalized Confusion Matrix")
    print(new_M)


def binary_classifier(poet1,poet2):
    data = []
    my_normalizer = Normalizer()
    with codecs.open('test.txt', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(my_normalizer.normalize(line.strip()))
    output = []
    y_true = []
    for i in range(int(len(data) / 3)):
        poet = data[3 * i]
        if poet == poet1 or poet == poet2:
            y_true.append(poet)
            B1 = len(bigram_train.get(poet1))
            B2 = len(bigram_train.get(poet2))
            token_list = data[3 * i + 1].split()
            p1 = 1
            p2 = 1
            for t in range(len(token_list) - 1):
                p1 *= probability(poet1, token_list[t] + ' ' + token_list[t + 1], delta=0.1, n_gram=2, B=B1, model=2)
                p2 *= probability(poet2, token_list[t] + ' ' + token_list[t + 1], delta=0.1, n_gram=2, B=B2, model=2)
            token_list = data[3 * i + 2].split()
            for k in range(len(token_list) - 1):
                p1 *= probability(poet1, token_list[k] + ' ' + token_list[k + 1], delta=0.1, n_gram=2, B=B1, model=2)
                p2 *= probability(poet2, token_list[k] + ' ' + token_list[k + 1], delta=0.1, n_gram=2, B=B2, model=2)
            p1 *= prob.get(poet1) / 94505
            p2 *= prob.get(poet2) / 94505
            if p1 > p2:
                output.append(poet1)
            else:
                output.append(poet2)
    print('f1_score , macro average is : ', metrics.f1_score(y_true, output, average='macro'))
    print('f1_score, micro average is : ', metrics.f1_score(y_true, output, average='micro'))
    print('precision, macro average is : ', metrics.precision_score(y_true, output, average='macro'))
    print('precision, micro average is : ', metrics.precision_score(y_true, output, average='micro'))
    print('recall, macro average is : ', metrics.recall_score(y_true, output, average='macro'))
    print('recall, micro average is : ', metrics.recall_score(y_true, output, average='macro'))





if __name__ == '__main__':
    #part b
    #
    # k = 1 means part a of the question and k = 2 means part b one
    # k = 2
    # if k == 1:
    #     uni, bi,counter = tokenizer_a("train.txt",1)
    #     print(len(uni))
    #     unigram_valid, bigram_valid,c = tokenizer_a("valid.txt",1)
    #     unigram_train = uni
    #     bigram_train = bi
    #     N = counter
    #     # perplexity('valid.txt', model=k, n_gram=1)
    #     perplexity('valid.txt', model=k, n_gram=2)
    # elif k == 2:
    #     uni, bi = tokenizer_b("train.txt",1)
    #     unigram_train = uni
    #     bigram_train = bi
    #     perplexity('valid.txt', model=k, n_gram=1)
    #     # perplexity('test.txt', model=k, n_gram=2)



    # part c
    #
    #
    # information_gain()
    # x_square()


    # part d-a
    #
    #
    # uni, bi = tokenizer_b("train.txt",mode=2)
    # unigram_train = uni
    # classification_with_feature_selection()

    # part d-b
    #
    #
    #
    # uni, bi = tokenizer_b("train.txt",mode=1)
    # unigram_train = uni
    # bigram_train = bi
    # classification_without_feature_selection()

    # part d-c
    #
    #
    # uni, bi = tokenizer_b("train.txt",1)
    # unigram_train = uni
    # bigram_train = bi
    # bigram_classification()


    # part e
    #
    #
    uni, bi = tokenizer_b("train.txt", 1)
    unigram_train = uni
    bigram_train = bi
    # y_true,y_pred = bigram_classification()
    # confusion_matrix(y_true,y_pred)
    # binary_classifier('sanaee','moulavi') # hard
    # binary_classifier('moulavi','amir') # easy
    binary_classifier('ghaani','amir') # intermediate