import pandas as pd
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class EmbeddingDoc2Vec:
    def __init__(self,dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=1,alpha=0.05, min_alpha=0.05):
        self.dm=dm
        self.vector_size=vector_size
        self.negative=negative
        self.hs=hs
        self.min_count=min_count
        self.sample=sample
        self.workers=workers
        self.alpha=alpha
        self.min_alpha=min_alpha
        global model
        
    def __tag_document(self,corpus,target):
        X=pd.DataFrame({"Text":corpus,"Label":target})
        TargetedData=X.apply(lambda r:TaggedDocument(words=word_tokenize(r.Text), tags=[r.Label]),axis=1)
        return TargetedData
    
    def __vec_for_learning(self,model, tagged_docs):
        sents = tagged_docs.values
        targets, regresseurs = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
        return targets, regresseurs
    
    def fit(self,X,y):
        targeted=self.__tag_document(X,y)
        
        self.model=Doc2Vec(dm=self.dm, vector_size=self.vector_size, negative=self.negative, hs=self.hs, 
                           min_count=self.min_count, sample =self.sample, workers=self.workers,
                           alpha=self.alpha,min_alpha=self.min_alpha)
        
        self.model.build_vocab([x for x in tqdm(targeted.values)])
        
        self.model.train(utils.shuffle([x for x in tqdm(targeted.values)]), total_examples=len(targeted.values), epochs=5)
        
        return self
    
    def transform(self,X,y):
        targeted=self.__tag_document(X,y)
        
        yres,Xres=self.__vec_for_learning(self.model,targeted)
        
        scaler = MinMaxScaler()
        scaler.fit(Xres)
        XresMinMax=scaler.transform(Xres)
        
        return yres,XresMinMax