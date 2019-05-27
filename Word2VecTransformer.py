import gensim
import pandas as pd
import nltk
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
import sys

class Embedding_Word2Vec:
    # Variable generale pour le modele skyp-gram/CBOW
    
    """
    - n_size: ici corréspond à la taille de la couche caché qui va nous servir  comme étant le vecteur qui représente un mot.
    - n_window: est la distance maximale entre le mot actuel et les mots qui lui corréspondent.
    - n_workers: est le nombre de processeur que l'on souaite utiliser.
    - n_min_count: on ignore les mots dont la fréquence est inféireur à cette paramètre.
    - n_sg: nous permet de choisir entre skip-gram et cbow (par défaut skip-gram)
    - n_sh: nous permet de choisir entre la fonction de soft max et le négatif simpling lorsque l'on va faire une déscente de gradien pour la partie optimisation.(valeur par défaut soft max)

    """
    
    def __init__(self,n_size,n_window,n_min_count,n_workers,n_sg=1,n_hs=1,IncludeSimilarity=0.5):
        self.n_size=n_size
        self.n_window=n_window
        self.n_min_count=n_min_count
        self.n_workers=n_workers
        self.n_hs=n_hs
        self.n_sg=n_sg
        if IncludeSimilarity> 1 or IncludeSimilarity<-1:
            print("IncludeSimilarity most be between -1 and 1!")
            sys.exit(1)
        self.IncludeSimilarity=IncludeSimilarity
        global model
    # Calculer la moyenne des mots à patie des vecteurs de mots.

    @staticmethod
    def  __word_averaging_list(self, text_list):

        result=[]
        for words in text_list:
            
            test=np.random.randint(self.n_size,size=10)
            res_temp=np.mean([self.model.wv[self.model.wv.index2word[t]] for t in test],axis=0)
            
            for w in words:
                if w in self.model.wv:
                    vect_word=self.model.wv[w]
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
      
        X_word_average = self.__word_averaging_list(self,X_tokenized)
  
        scaler = MinMaxScaler()
        scaler.fit(X_word_average)
        XresMinMax=scaler.transform(X_word_average)    
        return(XresMinMax)
    