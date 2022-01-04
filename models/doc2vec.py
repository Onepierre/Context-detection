from gensim.test.utils import common_texts

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import string
import tqdm
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from numpy import dot
from numpy.linalg import norm

from transformers import BertTokenizerFast, TFBertModel

from dataloader import SQuAD
from evaluate import scores

def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

# There is a long time of preprocessing so I'll 
class doc2VecFilter():
    def __init__(self):
        self.contexts_encodings = []
        self.questions_encodings = []
        self.tokenizer = None
        self.model = None
        self.labels = []

    def encodeContexts(self,contexts):
        documents = [TaggedDocument(doc.lower().split(" "), [i]) for i, doc in enumerate(contexts)]
        self.model = Doc2Vec(documents, vector_size=1024, window=10, min_count=1, workers=4)
        self.model.build_vocab(documents)
        self.model.train(documents,total_examples=self.model.corpus_count,epochs=self.model.epochs)
        for i,context in enumerate(contexts[:100]):
            print(self.model.docvecs[i])
            self.contexts_encodings.append(self.model.docvecs[i])
    
    def encodeQuestions(self,questions):
        
        for question,a in questions[:500]:
            self.labels.append(a)
            words = nltk.word_tokenize(question)
            line = [word for word in words if (word.isalnum())]
            output = self.model.infer_vector(line)
            #output = self.model.infer_vector([question])


            self.questions_encodings.append(output)
    
    def computeSimilarity(self,context,question):
        # Cosine similarity
        return cos_sim(context,question)


    def computeNearestContext(self,question):
        score_max = -1
        predict_id = -1
        for i,context in enumerate(self.contexts_encodings):
            score = self.computeSimilarity(context,question)
            if score > score_max:
                score_max = score
                predict_id = i
        return predict_id

    def computeModel(self):
        results = []
        min = 0
        max = 300
        for question in self.questions_encodings[min:max]:
            results.append(self.computeNearestContext(question))
        # for i in range(100):
        #     print(self.questions[i])
        #     print(self.contexts[results[i]])
        #     print(self.contexts[self.labels[i]])
        print(scores(results, self.labels[min:max]))