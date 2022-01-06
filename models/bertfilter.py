import string
import tqdm
import gc
import sys
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from numpy import dot
from numpy.linalg import norm
import torch as torch

from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModel

from dataloader import SQuAD
from evaluate import scores

def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

# There is a long time of preprocessing 
# A bert inference costs 50ms
class BertFilter():
    def __init__(self):
        self.contexts_encodings = []
        self.questions_encodings = []
        self.tokenizer = None
        self.BERT = None
        self.labels = []


        # self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        # self.model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

        # self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # self.model = AutoModel.from_pretrained('bert-base-uncased')

        # self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/nq-distilbert-base-v1')
        # self.model = AutoModel.from_pretrained('sentence-transformers/nq-distilbert-base-v1')

        # self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        # self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        
        
        
    def encodeContexts(self,contexts):

        self.model.eval()
        for context in contexts:
            tokens = {'input_ids': [], 'attention_mask': []}
            encoded_input = self.tokenizer(context, max_length = 512, truncation = True, padding='max_length', return_tensors='pt')

            tokens['attention_mask'].append(encoded_input['attention_mask'][0])
            tokens['input_ids'].append(encoded_input['input_ids'][0])
            tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
            tokens['input_ids'] = torch.stack(tokens['input_ids'])
            

            with torch.no_grad():
                output = self.model(**tokens)
            embeddings = output.last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()      

            masked_embeddings = embeddings * mask

            summed = torch.sum(masked_embeddings, 1)
            summed_mask = torch.clamp(mask.sum(1), min=1e-9)

            mean_pooled = summed / summed_mask

            self.contexts_encodings.append(mean_pooled[0])

    
    def encodeQuestions(self,questions):

        self.model.eval()
        for question,a in questions:
            self.labels.append(a)
            tokens = {'input_ids': [], 'attention_mask': []}
            encoded_input = self.tokenizer(question, max_length = 512, truncation = True, padding='max_length', return_tensors='pt')

            tokens['attention_mask'].append(encoded_input['attention_mask'][0])
            tokens['input_ids'].append(encoded_input['input_ids'][0])
            tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
            tokens['input_ids'] = torch.stack(tokens['input_ids'])
            

            with torch.no_grad():
                output = self.model(**tokens)
            embeddings = output.last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()      

            masked_embeddings = embeddings * mask

            summed = torch.sum(masked_embeddings, 1)
            summed_mask = torch.clamp(mask.sum(1), min=1e-9)

            mean_pooled = summed / summed_mask

            self.questions_encodings.append(mean_pooled[0])
    
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