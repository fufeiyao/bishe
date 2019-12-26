#########   scrapy    4条     经测试可以聚类
#########   scikit-learn     14条     经测试可以聚类
#########   text clustering    3,252条     经测试可以聚类
#########   ctcs   137条
#########   z-xing  386条

from __future__ import print_function
import urllib.parse
import urllib.request
import json
import codecs
import time
import random
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
import nltk
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import feature_extraction
import seaborn as sns
from wordcloud import WordCloud,ImageColorGenerator
from scipy.misc import imread
from os import path
from scipy.optimize import leastsq
d = path.dirname(__file__)


def responseFromSearchKeyword(keyWord, pageNumber):
    searchUrl = 'http://ieeexplore.ieee.org/rest/search'
    params = {'queryText': keyWord, 'pageNumber': pageNumber, 'newsearch': 'true', 'rowsPerPage':'100'}
    params = json.dumps(params).encode('utf-8')
    header = {"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64; rv:32.0) Gecko/20100101 Firefox/32.0",
    "Content-Type": "application/json;charset=UTF-8", "Referer": "http://ieeexplore.ieee.org",
    "Cookie": "ipCheck=8.8.8.8"}

    req = urllib.request.Request(searchUrl, data=params, headers=header)
    try:
        response = urllib.request.urlopen(req, timeout=5)
        html_source = response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        print('Error:HTTP - The server failed to respond!')
        return ''
    except urllib.error.URLError as e:
        print('Error:URL - Trouble in URL')
        return ''
    except:
        print('Error:TLE - Connection timeouts')
        return ''
    else:
        return html_source

def responseFromDocumentLink(link):
    '''打开文章的链接地址，从response中获取所需信息的json字符串'''
    baseUrl = 'http://ieeexplore.ieee.org'
    url = baseUrl + link
    try:
        response = urllib.request.urlopen(url, timeout=15)
        html_source = response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        print('服务器未能相应请求')
        return ''
    except urllib.error.URLError as e:
        print('URL错误')
        return ''
    except:
        print('连接超时')
        return ''
    else:
        usfulInfo = re.search('{"userInfo.*"}', html_source).group()
        # print(usfulInfo)
        return usfulInfo

def getInfo(keyWord, pageNum):
    global totalnum  #####目前抓取的条数
    '''搜索关键词和总页数，用for循环抓取所需文章的信息 '''
    res = responseFromSearchKeyword(keyWord, pageNum)
    print('正在打印第%d页的记录，每页100个记录' %pageNum)
    fileName = keyWord + '.txt'
    with codecs.open(fileName, 'w', 'utf-8') as fp:   ###用关键词做为保存的文件名
        if res:
            info = json.loads(res)   ###这个res是个json格式的字符串，把它转换成字典的形式
        else:
            print('从一级页面返回了空值')
            return -1
        records = info['totalRecords']   ####从字典中循环获取文章的标题，链接网址
        for i in range(len(info['records'])):
            print('#'*30)
            print('%d页的第%d个记录' %(pageNum, i + 1))

            title = info['records'][i]['title']    ### 文章标题
            pagetitle.append(title.replace("[",'').replace("]",'').replace(":",''))   #####pagetitle保存title
            print(pagetitle[totalnum])
            # print('title:\t %s' %title.replace("[",'').replace("]",'').replace(":",''))
            # fp.write('title:\t %s\r\n' %title.replace("[",'').replace("]",'').replace(":",'')) ### 去除字符串中的[:] zyh


            pageyear.append(int(info['records'][i]['publicationYear']))  #####pageyear保存year
            print(pageyear[totalnum])
            # print('publicationYear:\t %s\n' %info['records'][i]['publicationYear']) ###输出发表年份zyh
            # fp.write('publicationYear:\t %s\n' %info['records'][i]['publicationYear'])

            documentLink = info['records'][i]['documentLink']   ###文章的连接网址
            if re.search('document', documentLink):
                subResponse = responseFromDocumentLink(documentLink)    ###打开文章的链接网址，从中获取文章的作者，摘要， 关联作者
            else:
                continue
            try:
                subInfo = json.loads(subResponse)
            except:
                print('还原json失败')
                continue
            #####不用摘要，被降维成关键字了#####
            # try:
            #     abstract = subInfo['abstract']    ### 摘要
            # except KeyError as e:
            #     abstract = ''
            # print('Abstract:%s' %abstract)
            # fp.write('Abstract:\t%s\r\n' %abstract)
            #####不用摘要，被降维成关键字了#####
            try:
                keyWords = subInfo['keywords']
            except KeyError as e:
                keyWords = ''
                continue
            else:
                for keyWord in keyWords:
                    # print('种类\t：%s\n' % keyWord['type'])
                    # print('%s\n' %keyWord['kwd'])
                    # flag = 1
                    if(keyWord['type'] == 'IEEE Keywords'):
                        # print(str('%s\n' %keyWord['kwd']))     #####查看输出
                        pagekwd.append(str(keyWord['kwd']).replace(", ", '_').replace(" ", '-').replace("[", '').replace("]", '').replace("'", '').replace("_", ' '))
                        print(pagekwd[totalnum])
                        # flag =1
                        break
                    if (keyWord['type'] == 'INSPEC: Controlled Indexing'):
                                        # if re.search('[a-zA-Z]', keyWord['kwd']):
                        # pagekwd.append(keyWord['kwd'].replace("'",'').replace(".",' '))  #####pageyear保存year
                        pagekwd.append(str(keyWord['kwd']).replace(", ", '_').replace(" ", '-').replace("[", '').replace("]", '').replace("'", '').replace("_", ' '))
                        print(pagekwd[totalnum])
                        # flag = 1
                        break
                    if (keyWord['type'] == 'INSPEC: Non-Controlled Indexing'):
                                        # if re.search('[a-zA-Z]', keyWord['kwd']):
                        # pagekwd.append(keyWord['kwd'].replace("'",'').replace(".",' '))  #####pageyear保存year
                        pagekwd.append(str(keyWord['kwd']).replace(", ", '_').replace(" ", '-').replace("[", '').replace("]", '').replace("'", '').replace("_", ' '))
                        print(pagekwd[totalnum])
                        # flag = 1
                        break
                    if (keyWord['type'] == 'Author Keywords '):
                                    # if re.search('[a-zA-Z]', keyWord['kwd']):
                        # pagekwd.append(keyWord['kwd'].replace("'",'').replace(".",' '))  #####pageyear保存year
                        pagekwd.append(str(keyWord['kwd']).replace(", ", '_').replace(" ", '-').replace("[", '').replace("]", '').replace("'", '').replace("_", ' '))
                        print(pagekwd[totalnum])
                        # flag = 1
                        break
                    # if keyWord['type'] == 'Author Keywords ':
                    #     # pagekwd.append(keyWord['kwd'].replace("'",'').replace(".",' '))  #####pageyear保存year
                    #     pagekwd.append(str(keyWord['kwd']).replace("[", '').replace("]", '').replace("'", '').replace(",", ' '))
                    #     print(pagekwd[totalnum])
                    # elif keyWord['type'] == 'IEEE Keywords ':
                    #     pagekwd.append(str(keyWord['kwd']).replace("[", '').replace("]", '').replace("'", '').replace(",", ' '))
                    #     print(pagekwd[totalnum])

                        #print(kdd)
                        # print('Author Keyword:\t%s\n' % keyWord['kwd'])     ###仅输出种类为Author KeyWords的标签 ionlylo改
                        # fp.write('Author Keyword:\t%s\r\n' % keyWord['kwd'])
            time.sleep(1+random.random()*2) ###手动加入延迟，避免速度过快被封延迟时间在1s-3s之间
            # fp.write('\r\n'*6)
            totalnum += 1
            print('totalnum = %d\n' %totalnum)

def getPageNum(keyWord):
    '''搜索关键词，在文章列表的第一页里可以找到，符合关键词的文章共有多少篇，除以每次请求的文章数量，就可以得到需要抓取文章列表的页数 '''
    res = responseFromSearchKeyword(keyWord, 1)
    if res:
        info = json.loads(res)
    else:
        print('从一级页面返回了空值')
        return -1
    records = info['totalRecords']
    print('totalRecords = %d' %records)
    sumPageNum = (records // 100) + 1
    print('总页数\t :%d' %((records // 100) + 1))
    return sumPageNum

# def main():
#     print('这个程序用于在http://ieeexplore.ieee.org站点，根据关键词抓取相关的文章信息')
#     print('请不要频繁的运行该程序，避免被服务器发觉后封锁IP')
#     print('开始运行程序')
#     print('='*30)
#     keyWord = input('请输入关键词:')
#     if keyWord == '':
#         print('关键词不能为空，请重新运行程序')
#         return
#     sumPageNum = getPageNum(keyWord)
#     for i in range(1, sumPageNum + 1):
#         getInfo(keyWord, i)

# 这里我定义了一个分词器（tokenizer）和词干分析器（stemmer），它们会输出给定文本词干化后的词集合

def tokenize_and_stem(text):
    # 首先分句，接着分词，而标点也会作为词例存在,并且词干化
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 过滤所有不含字母的词例（例如：数字、纯标点）
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # 首先分句，接着分词，而标点也会作为词例存在
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 过滤所有不含字母的词例（例如：数字、纯标点）
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def save2txt(keyWord):
    f1name = keyWord + '_title.txt'
    f2name = keyWord + '_year.txt'
    f3name = keyWord + '_keywords.txt'
    ######title year keywords存入到文件中#####
    with codecs.open(f1name, 'w', 'utf-8') as f1:
        for i in range(0, totalnum):
            f1.write('%s\n' %pagetitle[i])
    with codecs.open(f2name, 'w', 'utf-8') as f2:
        for i in range(0, totalnum):
            f2.write('%d\n' %pageyear[i])
    with codecs.open(f3name, 'w', 'utf-8') as f3:
        for i in range(0, totalnum):
            f3.write('%s\n' %pagekwd[i])
        # print('i is %d' %i)
        # print('\n')
    ##### 存入到文件中#####

#####回归分析函数#####
def func(p,x):
    a,b,c=p
    return a*x*x+b*x+c

def error(p,x,y):
    return func(p,x)-y

def predict(Yi):
    Xi=np.array([1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016])
    # Yi=np.array([0,1,2,1,0,0,1,0,8,9,10,16,21,33,42,42,47,27,38,43,42,50])
    #a,b,c的初始值，可以任意设定
    p0=[10,10,10]
    #把error函数中除了p0以外的参数打包到args中(使用要求)
    Para=leastsq(error,p0,args=(Xi,Yi))
    #读取结果
    a,b,c=Para[0]
    '''
    绘图，看拟合效果.
    matplotlib默认不支持中文，label设置中文的话需要另行设置
    如果报错，改成英文就可以
    '''
    #画拟合直线
    x17= 2017
    y17=a*x17*x17+b*x17+c
    return y17
#####回归分析函数#####

def showlist(text,kind,title,num17):
    print(text)
    # 读入背景图片
    bg_pic = imread('3.png')
    # 生成词云
    # wordcloud = WordCloud(mask=bg_pic,background_color='white',scale=1.5).generate(text)
    wordcloud = WordCloud(mask=bg_pic, background_color='white', scale=3).generate(text)
    image_colors = ImageColorGenerator(bg_pic)
    # 显示词云图

    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title('Wordcloud of Kind '+str(kind),fontsize=25,)
    plt.show()

    # 保存图片
    # file_name = 'wordcloud'+ str(kind) +'.jpg'
    # print(file_name)
    # wordcloud.to_file(file_name)
    # 保存图片

    kindlist = list()
    predictlist = list()
    for i in range(totalnum):
        if clusters[i] == kind :
            kindlist.append(int(pageyear[i]))

    yearplot = {'year': kindlist}
    frameplot = pd.DataFrame(yearplot, index = [kindlist] , columns = ['year'])  #####各类生成图表
    sns.distplot(frameplot['year'],bins=50,kde=False,rug = False,color="#C1F320")
    sns.plt.xlabel('Year')
    sns.plt.ylabel('Quantity of papers')
    sns.plt.title('Keywords of Kind '+str(kind)+ ' is:  '+title)
    sns.plt.show()

    plt.figure(2)
    for i in range(num17):
        kindlist.append(int(2017))
    yearplot = {'year': kindlist}
    frameplot = pd.DataFrame(yearplot, index = [kindlist] , columns = ['year'])  #####各类生成图表
    sns.distplot(frameplot['year'],bins=50,kde=False,rug = False,color="#FF00FF")
    sns.plt.xlabel('Year')
    sns.plt.ylabel('Quantity of papers')
    sns.plt.title('In 2017,The Predictive Value of Kind '+str(kind)+' is '+ str(num17))
    sns.plt.show()
    # plt.plot(np.random.normal(size=100), np.random.normal(size=100), 'ro')
    # normals = pd.Series(np.random.normal(size=10))
    # normals.cumsum().plot()




if __name__ == '__main__':
    #####添加停用词#####
    stopwords = nltk.corpus.stopwords.words('english')
    # print(stopwords[:10])    #####测试停用词
    #####添加停用词#####

    #####添加词干提取器#####
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    #####添加词干提取器#####

    #####主程序#####
    global totalnum
    totalnum = 0
    pagetitle = []
    pageyear = []
    pgyear = []
    pagekwd = []
    print('这个程序用于在http://ieeexplore.ieee.org站点，根据关键词抓取相关的文章信息')
    print('请不要频繁的运行该程序，避免被服务器发觉后封锁IP')
    print('开始运行程序')
    print('='*30)
    keyWord = input('请输入关键词:')
    if keyWord == '':
        print('关键词不能为空，请重新运行程序')
        exit()
    # sumPageNum = getPageNum(keyWord)
    # ##################1.读入数据1##################
    # for i in range(1, sumPageNum + 1):
    #     getInfo(keyWord, i)
    # ##################1.读入数据1##################

    ##################2.读入数据2##################
    f1name = keyWord + '_title.txt'
    f2name = keyWord + '_year.txt'
    f3name = keyWord + '_keywords.txt'
    with codecs.open(f1name, 'r', 'utf-8') as f1:
        pagetitle = f1.read().split('\n')
    with codecs.open(f2name, 'r', 'utf-8') as f2:
        pageyear = f2.read().split('\n')
    with codecs.open(f3name, 'r', 'utf-8') as f3:
        pagekwd = f3.read().split('\n')
    totalnum = len(pagekwd)
    ##################2.读入数据2##################


    ##### 测试保存下来的数组#####
    # for i in range(0,totalnum):
    #     print('i is %d' %i)
    #     print(pagetitle[i])
    #     print(pageyear[i])
    #     print(pagekwd[i])
    #     print('\n')
    ##### 测试保存下来的数组#####

    ##################1.保存数据1##################
    # save2txt(keyWord)     #####将结果保存到以关键词开头的文件中
    ##################1.保存数据1##################

    ##### 使用 词干化+分词/ 仅分词 遍历关键字列表以生成两个词汇表#####
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in pagekwd:
        allwords_stemmed = tokenize_and_stem(i)  # 对每篇文章的关键词进行分词和词干化
        totalvocab_stemmed.extend(allwords_stemmed)  # 扩充“totalvocab_stemmed”列表

        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
    ##### 使用 词干化+分词/ 仅分词 遍历关键字列表以生成两个词汇表#####
    # print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')   #####查看词汇表大小
    # print(vocab_frame.head())   #####查看词干化后的词例

    ##### 关键词向量化成TF-IDF矩阵#####



    # 定义向量化参数
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.01, stop_words='english',                 ##########在3000数据中
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(pagekwd)  # 向量化关键词文本
    # print(tfidf_matrix.shape)
    ##### 关键词向量化成TF-IDF矩阵#####



    terms = tfidf_vectorizer.get_feature_names()       #####TF-IDF特征表
    from sklearn.metrics.pairwise import cosine_similarity
    dist = 1 - cosine_similarity(tfidf_matrix)         #####计算TF-IDF距离
    # print(dist)


    from sklearn.cluster import KMeans
    num_clusters = 5     #####输入聚类中心簇的个数
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()

    print(clusters)  #####输出分类结果

    #
    # from sklearn.externals import joblib
    # # 注释语句用来存储你的模型
    # # 因为我已经从 pickle 载入过模型了
    # joblib.dump(km,  'doc_cluster.pkl')     #####保存模型
    # km = joblib.load('doc_cluster.pkl')     #####加载模型
    # clusters = km.labels_.tolist()
    #
    print('pagetitle=%d\n' %len(pagetitle))
    print('pageyear=%d\n' % len(pageyear))
    print('keyword=%d\n' % len(pagekwd))
    print('cluster=%d\n' % len(clusters))

    papers = { 'title': pagetitle, 'year': pageyear, 'keyword': pagekwd, 'cluster': clusters }
    frame = pd.DataFrame(papers, index = [clusters] , columns = ['cluster','year', 'title', 'keyword'])     #####啥意思？？？？？？
    total_nums = frame['cluster'].value_counts()     #####对每个种类的文献数量进行计数
    total_years = frame['year'].value_counts()
    print(total_nums)
    print(total_years)

    print("Top terms per cluster:")
    # 按离质心的距离排列聚类中心，由近到远
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

##############################################################################################################词云二维list
    ptwordcloud = [([] * 20) for i in range(5)]     #####构建词云二维list
    for i in range(num_clusters):
        print("Cluster %d words:" % i, end='')
        for ind in order_centroids[i, :20]:  # 每个聚类选 20 个词
            # print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0],end=',')
            ptwordcloud[i].append(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0])
            # print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'),end=',')
        print()  # 空行
        print()  # 空行

        # print("Cluster %d titles:" % i, end='')
        # for title in frame.ix[i]['title'].values.tolist():
        #     print(' %s,' % title, end='')
        # print()  # 空行
        # print()  # 空行
    print(ptwordcloud)
    num_pageyear = []
    for i in range(len(pageyear)):
        print(pageyear[i])
        num_pageyear.append(int(pageyear[i]))
        print(num_pageyear[i])
    print(len(num_pageyear))
    print(clusters)
    print(len(clusters))

    pp = {'year': num_pageyear,'cluster': clusters}
    fff = pd.DataFrame(pp, index=[clusters], columns=['cluster','year'])
    # sns.distplot(fff['cluster'],kde=False)
    # sns.plt.show()
    # print(fff)
    # print(fff.sort_index(axis=0,ascending=True))
    #


    # ii = 0
    # jj = 0
    # arr = [ii for ii in range(25),  jj for jj in range(5)]
    arr = np.array([[0] * 22] * 5)
    for i in range(totalnum):
        # print('No.%d :cluster=%d year=%d\n' %i %clusters[i] %num_pageyear[i])
        print('No.%d' %i)
        print(':cluster=%d' %clusters[i])
        print('year=%d\n' %(num_pageyear[i]-1995))
        if(num_pageyear[i]>=1995 and num_pageyear[i]<=2016):
            arr[clusters[i]][(num_pageyear[i]-1995)] += 1

    # for i in range(5):
    for i in range(5):
        ptarr = arr[i]
        print(ptarr)
        num17 = int(predict(ptarr))
        showtext = " ".join(ptwordcloud[i])
        showtitle = "; ".join(ptwordcloud[i][:3])
        print (showtitle)
        showlist(showtext, i, showtitle.replace("-",' '),num17)

#
#         # print()
#         # arr[clusters[ii]-1995][num_pageyear[ii]]+=1
# #######################################################################################0526##################################################
#     print(arr)
#
#     a0rr = np.array([0] * 23) #####聚类结果拆分
#     a1rr = np.array([0] * 23)  #####聚类结果拆分
#     a2rr = np.array([0] * 23)  #####聚类结果拆分
#     a3rr = np.array([0] * 23)  #####聚类结果拆分
#     a4rr = np.array([0] * 23)  #####聚类结果拆分
#     # axisyear = (1995,1996,1997,1998,1999,2000,2001,'2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017')
#     axisyear = [1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
#
#     print(axisyear)
#     a0rr = arr[0]
#     a1rr = arr[1]
#     a2rr = arr[2]
#     a3rr = arr[3]
#     a4rr = arr[4]
#     # for i in range(23) :
#     #     a0rr[i] = arr[0][i]
#     #     a1rr[i] = arr[1][i]
#     #     a2rr[i] = arr[2][i]
#     #     a3rr[i] = arr[3][i]
#     #     a4rr[i] = arr[4][i]
#     aa0rr = list()
#     aa1rr = list()
#     aa2rr = list()
#     aa3rr = list()
#     aa4rr = list()
#     for i in a0rr:
#         aa0rr.append(int(i))
#     for i in a1rr:
#         aa1rr.append(int(i))
#     for i in a2rr:
#         aa2rr.append(int(i))
#     for i in a3rr:
#         aa3rr.append(int(i))
#     for i in a4rr:
#         aa4rr.append(int(i))
#     print(aa0rr)
#     print(aa1rr)
#     print(aa2rr)
#     print(aa3rr)
#     print(aa4rr)
#     # print(a1rr)
#     # print(a2rr)
#     # print(a3rr)
#     # print(a4rr)
#     yearplot = { 'year': axisyear, 'kind0': aa0rr, 'kind1': aa1rr, 'kind2': aa2rr , 'kind3': aa3rr , 'kind4': aa4rr }
#     frameplot = pd.DataFrame(yearplot, index = [axisyear] , columns = ['year','kind0', 'kind1', 'kind2', 'kind3', 'kind4'])     #####各类生成图表
#     # sns.distplot(frameplot['kind0'],kde=False)
#     # sns.plt.show()
#
#
#     print(frameplot)
#
#
#


    #####主程序#####

    # main()

