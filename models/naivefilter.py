import string
import tqdm
import numpy as np
import json
import nltk
from nltk.corpus import stopwords

from dataloader import SQuAD
from evaluate import scores

class BasicFilter():
    def __init__(self,contexts):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.questions = []
        self.labels = []
        self.contexts = []
        for context in contexts:
            words = nltk.word_tokenize(context)
            q = [word.lower() for word in words if (word not in self.stopwords and word.isalnum())]
            self.contexts.append(q)
    
    def loadQuestions(self, questions):
        # Tokenize questions + remove stopwords
        
        
        for question,label in questions:
            q = [word.lower() for word in nltk.word_tokenize(question) if (word not in self.stopwords and word.isalnum())]
            self.questions.append(q)
            self.labels.append(label)

    def computeSimilarity(self,context,question):
        #per word/ count per word/ number of letters in the word
        score = 0
        for word in question:
            if word in context:
                score += 1
        return score
        # score = 0
        # for word in question:
        #     malus = 1
        #     for word2 in context:
        #         if word == word2:
        #             score += 1/malus
        #             malus += 1
        # return score

    def computeNearestContext(self,question):
        score_max = -1
        predict_id = -1
        for i,context in enumerate(self.contexts):
            score = self.computeSimilarity(context,question)
            if score > score_max:
                score_max = score
                predict_id = i
        return predict_id

    def computeModel(self):
        results = []
        min = 0
        max = 300
        for question in self.questions[min:max]:
            results.append(self.computeNearestContext(question))
        # for i in range(100):
        #     print(self.questions[i])
        #     print(self.contexts[results[i]])
        #     print(self.contexts[self.labels[i]])
        print(scores(results, self.labels[min:max]))
