import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, sentence):
        pass


class UnderscoreSentenceTokenizer(Tokenizer): 
    """
    An example of custom tokenizer, it has to implement a "tokenize" function
    """

    def __init__(self):
        pass

    def tokenize(self, sentence):
        """
        tokenize a sentence into a series of words

        Args:
            sentence (String): a sentence which words are separated by "_" instead of space
        Returns:
            List[String]
        """
        return sentence.split("_")


class WhitespaceSentenceTokenizer(Tokenizer):
    """
    An example of custom tokenizer, it has to implement a "tokenize" function
    """

    def __init__(self):
        pass

    def tokenize(self, sentence):
        """
        tokenize a sentence into a series of words

        Args:
            sentence (String): a sentence which words are separated by whitespace
        Returns:
            List[String]
        """
        return sentence.split(" ")


class GloVe:
    """
    Sentence similarity implementation using GloVe
    """

    def __init__(self, path, tokenizer=None):
        """
        Word vocabulary initialization
        Args:
            path (String): GloVe word vector dictionary path (download one from https://nlp.stanford.edu/projects/glove/)
            tokenizer (Object): tokenizer object, default=RegexpTokenizer from nltk
        """
        print("[GloVe] Loading word vector dictionary")
        self.dictionary = pd.read_csv(filepath_or_buffer=path, delim_whitespace=True, header=None, index_col=0, doublequote=False, quoting=3)
        self.dimension = self.dictionary.shape[1] # vector dimension
        if tokenizer == None:
            self.sentence_tokenizer = RegexpTokenizer(r'\w+')
        else:
            self.sentence_tokenizer = tokenizer
        print("[GloVe] dictionary loaded: #tokens=%d dimension=%d" % (self.dictionary.shape[0], self.dimension))


    def word_vector(self, word):
        """
        Args:
            word (String) : a word
        Returns:
            np.array : word vector (array of float), vector of 0 if the word is not found in dictionary
        """
        try:
            return self.dictionary.loc[word].to_numpy()
        except: # word not found in dictionary
            return np.array([0] * self.dimension)


    def string_vector(self, sentence, mode="average"):
        """
        Convert a sentence into its vector form. Counted by average word vectors addition
        Args:
            sentence (String): a sentence you want to convert; the function then tokenize it into a series of words using the tokenizer you supplied
            mode (String): between "average", "maximum", "minimum"; default "average"
        Returns:
            np.array: vector form the sentence.
        """
        tokens = self.sentence_tokenizer.tokenize(sentence)
        return self.string_vector_tokenized(tokens, mode)


    def string_vector_tokenized(self, tokenized_string, mode="average"):
        """
        Convert a sentence into its vector form. Counted by average word vectors addition
        Args:
            sentence (List[String]): a sentence you want to convert, already segmented into its words
            mode (String): between "average", "maximum", "minimum"; default "average"
        Returns:
            np.array: vector form the sentence.
        """
        vectors = []
        for word in tokenized_string:
            temp_vector = self.word_vector(word)
            vectors.append(temp_vector)

        assert mode in {"average", "maximum", "minimum"}
        if mode == "average":
            result = np.average(vectors, axis=0)
        elif mode == "maximum":
            result = np.maximum.reduce(vectors)
        elif mode == "minimum":
            result = np.minimum.reduce(vectors)

        return result


    def vector_sim(self, vector1, vector2):
        """
        Compute cosine similarity between two linear vectors of same dimension
        Args:
            vector1 (np.array)
            vector2 (np.array)
        Returns:
            float: cosine similarity score
        """
        if (len(vector1)!=len(vector2)):
            raise Exception('[GloVe] cosine similarity between 2 vectors of different dimension')
        else:
            return cosine_similarity(np.asmatrix(vector1), np.asmatrix(vector2))[0][0]


    def string_sim(self, s1, s2):
        """
        Compute cosine similarity between two strings
        Args:
            s1 (String)
            s2 (String)
        Returns:
            float: cosine similarity score
        """
        return cosine_similarity(np.asmatrix(self.string_vector(s1)), np.asmatrix(self.string_vector(s2)))[0][0]


    def string_sim_tokenized(self, s1, s2):
        """
        Compute cosine similarity between two tokenized_strings
        Args:
            s1 (List[String])
            s2 (List[String])
        Returns:
            float: cosine similarity score
        """
        return cosine_similarity(np.asmatrix(self.string_vector_tokenized(s1)), np.asmatrix(self.string_vector_tokenized(s2)))[0][0]


if __name__ == "__main__": # Example
    tokenizer = WhitespaceSentenceTokenizer() # custom tokenizer
    
    glove = GloVe("GloVe_vectors/glove.6B.100d.txt", tokenizer)

    m = glove.word_vector("machine")
    print(m)
    
    ml = glove.string_vector("machine learning")
    ai = glove.string_vector("artificial intelligence")

    similarity1 = glove.vector_sim(ml, ai)
    similarity2 = glove.string_sim("machine learning", "artificial intelligence")
    similarity3 = glove.string_sim_tokenized(["machine", "learning"], ["artificial", "intelligence"])
    print(similarity1, similarity2, similarity3)
