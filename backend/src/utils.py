import pandas as pd
import json
# from sqlalchemy import create_engine
import sys
import jieba
import math
import numpy as np
import jieba.posseg as pseg

def read_config(path):
    try:
        with open(path, 'r') as file:
            configs = json.load(file)

        return configs
    except FileNotFoundError:
        print(f"The file {path} was not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {path}.")

# class MySQLAgent:
#     def __init__(self, config) -> None:
#         self.config = config
#         self.db_connector()

#     def db_connector(self):
#         user = self.config['user']
#         pw = self.config['pw']
#         host = self.config['host']
#         port = self.config['port']
#         database = self.config['database']

#         connection_string = f"mysql+pymysql://{user}:{pw}@{host}:{port}/{database}?charset=utf8mb4"

#         self.engine = create_engine(connection_string)

#     def read_table(self, query) -> pd.DataFrame:

#         df = pd.read_sql(query, con=self.engine)
#         df.columns = df.columns.str.lower()

#         return df

#     def write_table(self, data, table_name, if_exists, index, data_type):

#         data.to_sql(name=table_name, con=self.engine,
#                     if_exists=if_exists, index=index, dtype=data_type)

#-*- encoding:utf-8 -*-


sentence_delimiters=frozenset(u'。！？……')
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']

configs = read_config('.env/configs.json')
DICT_BIG_PATH = configs["DICT_BIG_PATH"]
MYDIC_PATH = configs['MYDIC_PATH']
jieba.set_dictionary(DICT_BIG_PATH)
jieba.load_userdict(MYDIC_PATH)

PY2 = sys.version_info[0] == 2
if not PY2:
    # Python 3.x and up
    text_type    = str
    string_types = (str,)
    xrange       = range

    def as_text(v):  ## 產生unicode字串
        if v is None:
            return None
        elif isinstance(v, bytes):
            return v.decode('utf-8', errors='ignore')
        elif isinstance(v, str):
            return v
        else:
            raise ValueError('Unknown type %r' % type(v))

    def is_text(v):
        return isinstance(v, text_type)

# else:
#     # Python 2.x
#     text_type    = unicode
#     string_types = (str, unicode)
#     xrange       = xrange

#     def as_text(v):
#         if v is None:
#             return None
#         elif isinstance(v, unicode):
#             return v
#         elif isinstance(v, str):
#             return v.decode('utf-8', errors='ignore')
#         else:
#             raise ValueError('Invalid type %r' % type(v))

#     def is_text(v):
#         return isinstance(v, text_type)

def cut_sentences(sentence):
    tmp = []
    for ch in sentence:  # 逐句比對字串中的每一個字
        tmp.append(ch)
        if sentence_delimiters.__contains__(ch):
            yield ''.join(tmp)
            tmp = []
    yield ''.join(tmp)

def cut_filter_words(cutted_sentences,stopwords,use_stopwords=False):
    sentences = []
    sents = []
    for sent in cutted_sentences:
        sentences.append(sent)
        if use_stopwords:
            sents.append([word for word in jieba.cut(sent) if word and word not in stopwords])  # 把句子分成詞彙
        else:
            sents.append([word for word in jieba.cut(sent) if word])
    return sentences,sents

def psegcut_filter_words(cutted_sentences,stopwords,use_stopwords=True,use_speech_tags_filter=True):
    sents = []
    sentences = []
    for sent in cutted_sentences:
        sentences.append(sent)
        jieba_result = pseg.cut(sent)
        if use_speech_tags_filter == True:
            jieba_result = [w for w in jieba_result if w.flag in allow_speech_tags]
        else:
            jieba_result = [w for w in jieba_result]
        word_list = [w.word.strip() for w in jieba_result if w.flag != 'x']
        word_list = [word for word in word_list if len(word) > 1]
        if use_stopwords:
            word_list = [word.strip() for word in word_list if word.strip() not in stopwords]
        sents.append(word_list)
    return  sentences,sents

def weight_map_rank(weight_graph,max_iter,tol):
    '''
    輸入相似度的圖（矩陣)
    返回各個句子的分數
    :param weight_graph:
    :return:
    '''
    # 初始分數設置為0.5
    #初始化每個句子的分子和老分數
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]
    denominator = caculate_degree(weight_graph)

    # 開始反覆運算
    count=0
    while different(scores, old_scores,tol):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        #計算每個句子的分數
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph,denominator, i)
        count+=1
        if count>max_iter:
            break
    return scores

def caculate_degree(weight_graph):
    length = len(weight_graph)
    denominator = [0.0 for _ in range(len(weight_graph))]
    for j in range(length):
        for k in range(length):
            denominator[j] += weight_graph[j][k]
        if denominator[j] == 0:
            denominator[j] = 1.0
    return denominator


def calculate_score(weight_graph,denominator, i):#i表示第i個句子
    """
    計算句子在圖中的分數
    :param weight_graph:
    :param scores:
    :param i:
    :return:
    """
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0

    for j in range(length):
        fraction = 0.0
        # 計算分子
        #[j,i]是指句子j指向句子i
        fraction = weight_graph[j][i] * 1.0
        #除以j的出度
        added_score += fraction / denominator[j]
    #算出最終的分數
    weighted_score = (1 - d) + d * added_score
    return weighted_score

def different(scores, old_scores,tol=0.0001):
    '''
    判斷前後分數有無變化
    :param scores:
    :param old_scores:
    :return:
    '''
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= tol:#原始是0.0001
            flag = True
            break
    return flag

def cosine_similarity(vec1, vec2):
    '''
    計算兩個向量之間的余弦相似度
    :param vec1:
    :param vec2:
    :return:
    '''
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value


def combine(word_list, window=2):
    """建構在window下的單詞組合，用來構造單詞之間的邊。

    Keyword arguments:
    word_list  --  list of str, 由單詞組成的列表。
    windows    --  int, window大小。
    """
    if window < 2: window = 2
    for x in xrange(1, window):
        if x >= len(word_list):
            break
        word_list2 = word_list[x:]
        res = zip(word_list, word_list2)
        for r in res:
            yield r


def two_sentences_similarity(sents_1, sents_2):
    '''
    計算兩個句子的相似性
    :param sents_1:
    :param sents_2:
    :return:
    '''
    counter = 0
    for sent in sents_1:
        if sent in sents_2:
            counter += 1
    if counter==0:
        return 0
    return counter / (math.log(len(sents_1) + len(sents_2)))