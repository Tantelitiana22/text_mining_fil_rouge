from gensim.models.wrappers.fasttext import FastText as FT_wrapper
from gensim.test.utils import datapath
import numpy as np
import os 
from sklearn.preprocessing import MinMaxScaler

ft_home ='/home/tantely/fastText-0.2.0/fasttext'

class FastTextTransformer:
    def __init__(self,inputFile,ft_home,path_to_data,model="cbow",size=100, word_ngrams=3):
        
        if not os.path.isfile(ft_home) :
            print("Path" ,ft_home,"does not exist")
            sys.exit(1)
        if not os.path.isfile(inputFile) :
            print("Path" ,ft_home,"does not exist")
            sys.exit(1)
        corpus_file=datapath(path_to_data)
        
        self.ft_home=ft_home
        self.corpus_file=corpus_file
        self.inputFile=inputFile
        self.model=model
        self.size=size
        self.word_ngrams=word_ngrams
        
        global model_wrapper
        
   
    def fit(self,X,y=None):
        X.to_csv(self.inputFile,index=False)
        corpus_file=datapath(self.inputFile)
        self.model_wrapper = FT_wrapper.train(self.ft_home, self.inputFile,model=self.model,size=self.size,word_ngrams=self.word_ngrams)
        return self
    
    def __average_word(self,X):
        return np.array([np.mean([self.model_wrapper[w] for w in words.split()if w in self.model_wrapper], axis=0) for words in X])

    def transform(self,X,y=None):
        Xres = self.__average_word(X)
        scaler = MinMaxScaler()
        scaler.fit(Xres)
        return scaler.transform(Xres)