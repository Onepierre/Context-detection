import string
import tqdm
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
 


from dataloader import SQuAD
from evaluate import scores

class BasicFilter():
    def __init__(self,contexts):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.debug = []
        self.lemmatizer = WordNetLemmatizer()
        bonus = ["one",'us',"much","many","gave","asked","say"]
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
            q = [self.lemmatizer.lemmatize(word).lower() for word in words if (word.lower() not in self.stopwords and word.isalnum())]
            self.contexts.append(q)
    
    def loadQuestions(self, questions):
        # Tokenize questions + remove stopwords + lemmatization
        
        for question,label in questions:
            q = [self.lemmatizer.lemmatize(word).lower() for word in nltk.word_tokenize(question) if (word.lower() not in self.stopwords and word.isalnum())]
            self.questions.append(q)
            self.labels.append(label)

    def computeSimilarity(self,context,question):
        #per word/ count per word/ number of letters in the word

        score = 0
        score_2 = 0
        for word in question:
            if word in context:
                score += 1
            # for  word2 in context:
            #     if word == word2:
            #         score += 1

        return score/len(question),score_2

    def computeSimilarityV2(self,context,question):
        context_text = ""
        for w in context:
            context_text += w + " "


        # Quadratic in length of the question
        # may be too long

        score = 0
        n = len(question)
        for i in range(n):
            for j in range(i+1,n+1):
                q_text = ""
                for w in question[i:j]:
                    q_text += w + " "
                if q_text in context_text:
                    #best 
                    score += 1/(j-i)
        return score/len(question),0



    def computeNearestContext(self,question):
        score_max = -1
        score_2_max = -1
        predict_id = -1
        for i,context in enumerate(self.contexts):
            score,score_2 = self.computeSimilarityV2(context,question)
            if score >= score_max:
                score_max = score
                score_2_max = score_2
                predict_id = i
            # elif score == score_max and score_2_max < score_2:
            #     score_max = score
            #     score_len_max = score_2
            #     predict_id = i
        return predict_id,score_max

    def computeModel(self):
        self.results = []
        for question in self.questions:
            predict_id,score_max = self.computeNearestContext(question)
            self.results.append(predict_id)
            self.scores_id.append(score_max)
        print(scores(self.results, self.labels))

    def getConfidence(self):
        
        # For tests only

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
                for word in self.questions[i]:
                    if word in self.contexts[self.results[i]]:
                        print(word)
        print(eq,neq)