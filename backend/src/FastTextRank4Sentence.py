# -*- encoding:utf-8 -*-
import jieba
import math
from string import punctuation
from heapq import nlargest
from itertools import product, count
from gensim.models import keyedvectors

from src import utils
import numpy as np
import os
import codecs
from itertools import count


class FastTextRank4Sentence(object):
    def __init__(self, use_stopword=False, stop_words_file=None, use_w2v=False, dict_path=None, max_iter=100, tol=0.0001):
        """
        :param use_stopword: 是否使用停用詞
        :param stop_words_file: 停用詞檔路徑
        :param use_w2v: 是否使用詞向量計算句子相似性
        :param dict_path: 詞向量字典檔路徑
        :param max_iter: 最大反覆運算倫茨
        :param tol: 最大容忍誤差
        """
        if use_w2v == False and dict_path != None:
            raise RuntimeError("使用詞向量之前必須傳入參數use_w2v=True")
        self.__use_stopword = use_stopword
        self.__use_w2v = use_w2v
        self.__dict_path = dict_path
        self.__max_iter = max_iter
        self.__tol = tol
        if self.__use_w2v:
            self.__word2vec = keyedvectors.KeyedVectors.load_word2vec_format(
                self.__dict_path, binary=True)
        self.__stop_words = set()
        self.__stop_words_file = self.get_default_stop_words_file()
        if type(stop_words_file) is str:
            self.__stop_words_file = stop_words_file
        if use_stopword:
            for word in codecs.open(self.__stop_words_file, 'r', 'utf-8', 'ignore'):
                self.__stop_words.add(word.strip())
        # Print a RuntimeWarning for all types of floating-point errors
        np.seterr(all='warn')

    def get_default_stop_words_file(self):
        d = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(d, 'dict/stopwords.txt')

    # 可以改為刪除停用詞，詞性不需要的詞
    def filter_dictword(self, sents):
        """
        刪除詞向量字典裡不存的詞
        :param sents:
        :return:
        """
        _sents = []
        dele = set()
        for sentence in sents:
            for word in sentence:
                if word not in self.__word2vec:
                    dele.add(word)
            if sentence:
                _sents.append([word for word in sentence if word not in dele])
        return _sents

    def summarize(self, text, n):
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        text = utils.as_text(text)  # 處理編碼問題
        tokens = utils.cut_sentences(text)
        # sentences用於記錄文章原始句子，sents用在計算
        sentences, sents = utils.cut_filter_words(
            tokens, self.__stop_words, self.__use_stopword)
        if self.__use_w2v:
            sents = self.filter_dictword(sents)
        graph = self.create_graph_sentence(sents, self.__use_w2v)
        scores = utils.weight_map_rank(graph, self.__max_iter, self.__tol)
        sent_selected = nlargest(n, zip(scores, count()))
        sent_index = []
        for i in range(n):
            try:
                sent_index.append(sent_selected[i][1])  # 新增關鍵字在原來文章中的下標
            except:
                pass
        return [sentences[i] for i in sent_index]

    def create_graph_sentence(self, word_sent, use_w2v):
        """
        傳入句子鏈表  返回句子之間相似度的圖
        :param word_sent:
        :return:
        """
        num = len(word_sent)
        board = [[0.0 for _ in range(num)] for _ in range(num)]

        for i, j in product(range(num), repeat=2):
            if i != j:
                if use_w2v:
                    board[i][j] = self.compute_similarity_by_avg(
                        word_sent[i], word_sent[j])
                else:
                    board[i][j] = utils.two_sentences_similarity(
                        word_sent[i], word_sent[j])
        return board

    def compute_similarity_by_avg(self, sents_1, sents_2):
        '''
        兩個句子求平均詞向量
        :param sents_1:
        :param sents_2:
        :return:
        '''
        if len(sents_1) == 0 or len(sents_2) == 0:
            return 0.0
        # 把一個句子中的所有詞向量相加
        vec1 = self.__word2vec[sents_1[0]]
        for word1 in sents_1[1:]:
            vec1 = vec1 + self.__word2vec[word1]

        vec2 = self.__word2vec[sents_2[0]]
        for word2 in sents_2[1:]:
            vec2 = vec2 + self.__word2vec[word2]

        similarity = utils.cosine_similarity(
            vec1 / len(sents_1), vec2 / len(sents_2))
        return similarity
