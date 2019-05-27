from gensim.models.wrappers.fasttext import FastText as FT_wrapper
from gensim.test.utils import datapath
import numpy as np
import os 
from sklearn.preprocessing import MinMaxScaler

ft_home ='/home/tantely/fastText-0.2.0/fasttext'

class FastTextTransformer:
    def __init__(self,inputFile,ft_home,model="cbow",size=100, word_ngrams=3,IncludeSimilarity=0.5):
        
        if not os.path.isfile(ft_home) :
            print("Path" ,ft_home,"does not exist")
            sys.exit(1)
        if not os.path.isfile(inputFile) :
            print("Path" ,inputFile,"does not exist")
            sys.exit(1)
        corpus_file=datapath(inputFile)
        
        self.ft_home=ft_home
        self.corpus_file=corpus_file
        self.inputFile=inputFile
        self.model=model
        self.size=size
        self.word_ngrams=word_ngrams
        if IncludeSimilarity> 1 or IncludeSimilarity<-1:
            print("IncludeSimilarity most be between -1 and 1!")
            sys.exit(1)
        self.IncludeSimilarity=IncludeSimilarity
        global model_wrapper
        
   
    def fit(self,X,y=None):
        X.to_csv(self.inputFile,index=False)
        corpus_file=datapath(self.inputFile)
        self.model_wrapper = FT_wrapper.train(self.ft_home, self.inputFile,model=self.model,size=self.size,word_ngrams=self.word_ngrams)
        return self
    
    def __average_word(self,X):
        return np.array([np.mean([self.model_wrapper[w] for w in words.split()if w in self.model_wrapper], axis=0) for words in X])
    
    def  __word_averaging_list(self, text_list):

        result=[]
        for words in text_list:
            
            test=np.random.randint(self.size,size=50)
            res_temp=np.mean([self.model_wrapper.wv[self.model_wrapper.wv.index2word[t]] for t in test],axis=0)
            
            for w in words:
                if w in self.model_wrapper.wv:
                    vect_word=self.model_wrapper.wv[w]
                    vect_word_norm=vect_word/np.linalg.norm(vect_word)
                    result_norm=np.linalg.norm(res_temp)
                    res_temp_norm=res_temp/result_norm
                    # cosinus des mots 
                    dists=np.dot(res_temp_norm,vect_word_norm)
                    # Si le cosinus est inférieur à un seuil on n'ajoute
                    # pas le vecteur
                    if np.abs(dists)>self.IncludeSimilarity:
                        res_temp=np.mean([res_temp,vect_word],axis=0)
                 
            result.append(res_temp)       
        
        return result

    def transform(self,X,y=None):
        Xres = self.__word_averaging_list(X)
        scaler = MinMaxScaler()
        scaler.fit(Xres)
        return scaler.transform(Xres)
    
    
    
    
    
    
    
    
    
    
    
    
    