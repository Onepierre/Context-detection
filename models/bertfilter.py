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
class BertFilter():
    def __init__(self):
        self.contexts_encodings = []
        self.questions_encodings = []
        self.tokenizer = None
        self.BERT = None
        self.labels = []

    def encodeContexts(self,contexts):
        if self.tokenizer == None:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        if self.BERT == None:
            self.model = TFBertModel.from_pretrained("bert-base-uncased")
        for context in contexts[:100]:
            
            encoded_input = self.tokenizer(context, return_tensors='tf',max_length = 512)
            output = self.model(encoded_input)
            self.contexts_encodings.append(output[0][0][0])
    
    def encodeQuestions(self,questions):
        if self.tokenizer == None:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        if self.BERT == None:
            self.model = TFBertModel.from_pretrained("bert-base-uncased")
        
        for question,a in questions[:500]:
            print(question)
            self.labels.append(a)
            encoded_input = self.tokenizer(question, return_tensors='tf',max_length = 512)
            output = self.model(encoded_input)
            self.questions_encodings.append(output[0][0][0])
    
    def computeSimilarity(self,context,question):
        # Cosine similarity
        return cos_sim(context.numpy(),question.numpy())


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