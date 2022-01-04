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
        self.debug = []
        bonus = ["one",'us']
        for a in bonus:
            self.stopwords.append(a)
        #print(self.stopwords)
        self.questions = []
        self.labels = []
        self.contexts = []
        self.scores_id = []
        self.results = []
        for context in contexts:
            words = nltk.word_tokenize(context)
            q = [word.lower() for word in words if (word.lower() not in self.stopwords and word.isalnum())]
            self.contexts.append(q)
    
    def loadQuestions(self, questions):
        # Tokenize questions + remove stopwords
        
        
        for question,label in questions:
            q = [word.lower() for word in nltk.word_tokenize(question) if (word.lower() not in self.stopwords and word.isalnum())]
            self.questions.append(q)
            self.labels.append(label)

    def computeSimilarity(self,context,question):
        #per word/ count per word/ number of letters in the word
        score = 0
        for word in question:
            if word in context:
                # if len(word) <= 3:
                #     if word not in self.debug:
                #         print(word)
                #         self.debug.append(word)
                    score += 1

        return score/len(question)

    def computeNearestContext(self,question):
        score_max = -1
        predict_id = -1
        for i,context in enumerate(self.contexts):
            score = self.computeSimilarity(context,question)
            if score > score_max:
                score_max = score
                predict_id = i
        return predict_id,score_max

    def computeModel(self):
        self.results = []
        for question in self.questions:
            predict_id,score_max = self.computeNearestContext(question)
            self.results.append(predict_id)
            self.scores_id.append(score_max)
        # for i in range(100):
        #     print(self.questions[i])
        #     print(self.contexts[results[i]])
        #     print(self.contexts[self.labels[i]])
        print(scores(self.results, self.labels))

    def getConfidence(self):

        eq = 0
        neq = 0
        for i in range(len(self.results)):
            if not (self.results[i] == self.labels[i]):
                print("Error")
                print("Score max : " + str(self.scores_id[i]))
                th_score = self.computeSimilarity(self.contexts[self.labels[i]],self.questions[i])
                print("theorical best : " + str(th_score))
                if self.scores_id[i] == th_score:
                    eq += 1
                else:
                    neq += 1
        print(eq,neq)