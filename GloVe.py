import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Sentence similarity implementation using GloVe
class GloVe:
    def __init__(self, path):
        """
        Word vocabulary initialization
        :param path: GloVe word vectors path
        """
        print "[LOG GloVe] Loading word dictionary"
        self.dictionary = pd.read_table(filepath_or_buffer=path, delim_whitespace=True, header=None, index_col=0, doublequote=False, quoting=3)
        self.dimension = self.dictionary.shape[1] # vector dimension
        self.sentence_tokenizer = tokenizer = RegexpTokenizer(r'\w+')
        print "[LOG GloVe] dictionary ",self.dictionary.shape[0],"tokens, vector dimension=",self.dimension

    def word_vector(self, word):
        """
        :param word: a word
        :return: word vector (array of float), vector of 0 if the word is not found in dictionary
        """
        try:
            return self.dictionary.loc[word].tolist()
        except:
            return [0] * self.dimension

    def string_vector(self, string):
        """
        Convert a string into its vector form. Counted by average word vectors addition
        :param string: string
        :return: vector form of sentence.
        """
        tokens = self.sentence_tokenizer.tokenize(string)
        vector = [0] * self.dimension
        for word in tokens:
            temp_vector = self.word_vector(word)
            for i in range(len(temp_vector)):
                vector[i] += temp_vector[i]
        for i in range(len(vector)):
            vector[i]/=len(tokens)
        return vector

    def string_vector_tokenized(self, tokenized_string):
        """
        Convert a tokenized string into its vector form. Counted by average word vectors addition
        :param tokenized_string: tokenized string
        :return: vector form of tokenized string
        """
        vector = [0] * self.dimension
        for word in tokenized_string:
            temp_vector = self.word_vector(word)
            for i in range(len(temp_vector)):
                vector[i] += temp_vector[i]
        for i in range(len(vector)):
            vector[i]/=len(tokenized_string)
        return vector

    def vector_sim(self, vector1, vector2):
        """
        Compute cosine similarity between two linear vectors of same dimension
        :param vector1: first sentence vector
        :param vector2: second sentence vector
        :return: cosine similarity
        """
        if (len(vector1)!=len(vector2)):
            raise Exception('[GloVe] cosine similarity between 2 vectors of different dimension')
        else:
            return cosine_similarity(np.asmatrix(vector1), np.asmatrix(vector2))[0][0]

    def string_sim(self, s1, s2):
        """
        Compute cosine similarity between 2 strings
        :param s1: first  string
        :param s2: second string
        :return: cosine similarity
        """
        return cosine_similarity(np.asmatrix(self.string_vector(s1)), np.asmatrix(self.string_vector(s2)))[0][0]

    def string_sim_tokenized(self, s1, s2):
        """
        Compute cosine similarity between 2 tokenized strings
        :param s1: first tokenized string
        :param s2: second tokenized string
        :return: cosine similarity
        """
        return cosine_similarity(np.asmatrix(self.string_vector_tokenized(s1)), np.asmatrix(self.string_vector_tokenized(s2)))[0][0]