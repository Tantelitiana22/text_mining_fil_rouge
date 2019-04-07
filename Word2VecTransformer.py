import gensim
import pandas as pd
import nltk
import numpy as np
import logging

class Embedding_skipGram:
    # Variable generale pour le modele skyp-gram
    
    def __init__(self,n_size,n_window,n_min_count,n_workers,n_sg=1,n_hs=1):
        self.n_size=n_size
        self.n_window=n_window
        self.n_min_count=n_min_count
        self.n_workers=n_workers
        self.n_hs=n_hs
        self.n_sg=n_sg
        global model
    # Calculer la moyenne des mots à patie des vecteurs de mots.
    @staticmethod
    def __word_averaging(self,wv, words):
        all_words, mean = set(), []
    
        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in wv.vocab:
                mean.append(wv.syn0norm[wv.vocab[word].index])
                all_words.add(wv.vocab[word].index)

        if not mean:
            #logging.warning("ne peut pas calculer la similarité sans entrée %s", words)
            #  On enleve les mots dont la moyenne ne peut être calculer
            return np.zeros(wv.vector_size,)
        else:
         
            mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
            return mean
    ## Calcul sur l'ensemble du corpus
    @staticmethod
    def  __word_averaging_list(self,wv, text_list):
        return np.vstack([self.__word_averaging(self,wv, post) for post in text_list ])
    
    # on fit notre modèle sur X ( X_train par exemple)
    def fit(self,X,y=None):
        test_tokenized=None 
        if isinstance(X, pd.Series):
            X_tokenized = list(X.apply(lambda r: gensim.utils.simple_preprocess(r)))
        else:
            X=pd.DataFrame(X)
            X_tokenized = list(X.apply(lambda r: gensim.utils.simple_preprocess(r)))  
        
        self.model=gensim.models.Word2Vec(X_tokenized,size=self.n_size,window=self.n_window,min_count=self.n_min_count,
                                          workers=self.n_workers,sg=self.n_sg,hs=self.n_hs)
        self.model.init_sims(replace=True)
        
        return self
    # On transforme notre corpus grace à cette fonction en un matrice qui représente notre corpus
    def transform(self,X,y=None):
        test_tokenized=None 
        
        if isinstance(X, pd.Series):
            X_tokenized = list(X.apply(lambda r: gensim.utils.simple_preprocess(r)))
        else:
            X=pd.DataFrame(X)
            X_tokenized = list(X.apply(lambda r: gensim.utils.simple_preprocess(r)))
        wv=self.model.wv
      
        X_word_average = self.__word_averaging_list(self,self.model.wv,X_tokenized)
        
        return(X_word_average)