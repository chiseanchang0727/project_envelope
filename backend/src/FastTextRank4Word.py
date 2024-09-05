# -*- encoding:utf-8 -*-
import jieba
import math
from string import punctuation
from heapq import nlargest
from itertools import product, count
# from gensim.models import word2vec
from src import utils
import numpy as np
import os
from itertools import count
import codecs


class FastTextRank4Word(object):
    def __init__(self, use_stopword=False, stop_words_file=None, max_iter=100, tol=0.0001, window=2):
        """
        :param max_iter: 最大的反覆運算輪次
        :param tol: 最大的容忍誤差
        :param window: 詞語 window
        :return:
        """
        self.__use_stopword = use_stopword
        self.__max_iter = max_iter
        self.__tol = tol
        self.__window = window
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
        return os.path.join(d, 'stopwords.txt')

    def build_worddict(self, sents):
        """
        構建字典，是詞語和下標之間生成一對一的聯繫，為之後的詞圖構建做準備
        :param sents:
        :return:
        """
        word_index = {}
        index_word = {}
        words_number = 0
        for word_list in sents:
            for word in word_list:
                if not word in word_index:
                    word_index[word] = words_number
                    index_word[words_number] = word
                    words_number += 1
        return word_index, index_word, words_number

    def build_word_grah(self, sents, words_number, word_index, window=2):
        graph = [[0.0 for _ in range(words_number)]
                 for _ in range(words_number)]
        for word_list in sents:
            for w1, w2 in utils.combine(word_list, window):
                if w1 in word_index and w2 in word_index:
                    index1 = word_index[w1]
                    index2 = word_index[w2]
                    graph[index1][index2] += 1.0
                    graph[index2][index1] += 1.0
        return graph

    def summarize(self, text, n):
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        text = utils.as_text(text)  # 處理編碼問題
        tokens = utils.cut_sentences(text)
        # sentences用於記錄文章最原本的句子，sents用於計算
        sentences, sents = utils.psegcut_filter_words(
            tokens, self.__stop_words, self.__use_stopword)

        word_index, index_word, words_number = self.build_worddict(sents)
        graph = self.build_word_grah(
            sents, words_number, word_index, window=self.__window)
        scores = utils.weight_map_rank(
            graph, max_iter=self.__max_iter, tol=self.__tol)
        sent_selected = nlargest(n, zip(scores, count()))
        sent_index = []
        for i in range(n):
            try:
                sent_index.append(sent_selected[i][1])  # 新增入關鍵字在原來文章中的下標
            except:
                pass
        return [index_word[i] for i in sent_index]
