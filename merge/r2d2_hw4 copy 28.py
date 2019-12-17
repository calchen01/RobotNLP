############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Qiannan Guo  Yucheng Ruan"

############################################################
# Imports
############################################################

from pymagnitude import *
import numpy as np
import string
import re
from collections import Counter

############################################################
# Helper Functions
############################################################

def loadTrainingSentences(file_path):
    commandTypeToSentences = {}

    with open(file_path, 'r') as fin:
        for line in fin:
            line = line.rstrip('\n')
            if len(line.strip()) == 0 or "##" == line.strip()[0:2]:
                continue
            commandType, command = line.split(' :: ')
            commandType = commandType[:-9]
            if commandType not in commandTypeToSentences:
                commandTypeToSentences[commandType] = [command]
            else:
                commandTypeToSentences[commandType].append(command)

    return commandTypeToSentences

############################################################
# Section 1: Natural Language Commands for R2D2
############################################################

### YOUR CODE HERE ###
my_driving_sentences = ["go ahead",
                       "turn right",
                        "slow down 10%",
                       "make a left turn",
                       "speed up 20%",
                       "increase the speed by 50%",
                       "stop driving",
                       "go to the east direction",
                       "go forward for five feet, then turn right",
                       "head 90 degrees"]

my_light_sentences = ["turn white light on",
                     "change the back light to red",
                     "change the front LED to green",
                     "start shining with yellow",
                     "set both lights to yellow",
                     "make the back light to blue",
                     "turn off the lights",
                     "lights on",
                     "shine the following colors for 2 seconds each: red, orange, yellow, green, blue, purple",
                     "increase the back light by 50%"]



my_head_sentences = ["turn back",
                    "look at the right",
                    "head to east",
                    "face front",
                    "turn to the forward",
                    "turn to 30 degrees",
                    "head to northeast direction",
                    "turn right",
                    "look backward",
                    "turn to the west direction"]

my_state_sentences = ["what is the front light color?",
                     "what is the heading?",
                     "what is the direction?",
                     "what is the speed?",
                     "what is the color of te back light?",
                     "Can I knwo the back light's color?",
                     "How many wheels are standing?",
                     "Can you tell me how much barrery is left?",
                     "Show me the heading degrees.",
                     "What's your name?"]

my_connection_sentences = ["connect the robot D2-55A2 to the computer",
                           "how many droids nearby?",
                           "disconnect from the computer",
                           "disconnect the robot",
                           "try to connect the robot D2-55A2",
                           "what’s the number of nearby robots?",
                           "stop the robot",
                           "connect the R2D2 D2-55A2",
                           "make a connection with D2-55A2",
                          "make a disconnection"]


my_stance_sentences = ["put down all wheels",
                       "stand with all wheels",
                       "lift the third wheels",
                       "stand on the tiptoes",
                       "make the stance to waddle",
                       "let the robot waddle",
                       "stop the robot from waddling",
                       "make the droids stop waddling",
                       "stop toddering",
                      "stand well"]

my_animation_sentences = ["start screaming",
                         "start laughing",
                         "start making noise",
                         "become noisy",
                         "lay down",
                         "couch",
                         "make laughing sounds",
                         "make noise",
                         "be a noisy droids",
                         "be quiet"]

my_grid_sentences = ["you are on 2 times 2 grid",
                    "the step unit is one",
                    "your position is (2, 2)",
                    "you are on (2, 3)",
                    "A blocker is on (1,1)",
                    "you can not on (1, 1)",
                    "you can't pass (1, 2) to (1, 1)",
                    "An barrier is set on (1, 1)",
                    "the edge of the square is 1",
                    "arrive (3, 3)"]

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    tokens = res = re.findall(r'\w+', sentence.lower())
    return tokens

def cosineSimilarity(vector1, vector2):
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))


def clean_str(string):

    string = re.sub(r"the", "", string)
    string = re.sub(r"on", "", string)
    string = re.sub(r"your", "", string)
    return string.strip().lower()


class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)

        wordVector = np.zeros(300)
        for word in tokens:
            wordVector = wordVector + self.vectors.query(word)
        return wordVector

    def sentenceToEmbeddings(self, commandTypeToSentences):
        '''Returns a tuple of sentence embeddings and an index-to-(category, sentence)
        dictionary.

        Inputs:
            commandTypeToSentences: A dictionary in the form returned by
            loadTrainingSentences. Each key is a string '[category]' which
            maps to a list of the sentences belonging to that category.

        Let m = number of sentences.
        Let n = dimension of vectors.

        Returns: a tuple (sentenceEmbeddings, indexToSentence)
            sentenceEmbeddings: A mxn numpy array where m[i:] containes the embedding
            for sentence i.

            indexToSentence: A dictionary with key: index i, value: (category, sentence).
        '''
        sentenceEmbeddings = []
        indexToSentence = {}
        index = 0
        for category in commandTypeToSentences:
            for sentence in commandTypeToSentences[category]:
                sentenceEmbeddings.append(self.calcSentenceEmbeddingBaseline(sentence))
                indexToSentence[index] = (sentence, category)
                index = index+1
        if(len(sentenceEmbeddings) == 0):
            sentenceEmbeddings = np.zeros(shape=(index, 300))
        else:
            sentenceEmbeddings = np.asarray(sentenceEmbeddings, dtype=np.float32)
        return (sentenceEmbeddings, indexToSentence)

    def closestSentence(self, sentence, sentenceEmbeddings):
        '''Returns the index of the closest sentence to the input, 'sentence'.

        Inputs:
            sentence: A sentence

            sentenceEmbeddings: An mxn numpy array, where m is the total number
            of sentences and n is the dimension of the vectors.

        Returns:
            an integer i, where i is the row index in sentenceEmbeddings 
            that contains the closest sentence to the input
        '''

        result = 0
        vec1 = self.calcSentenceEmbeddingBaseline(sentence)
        sim = float('-inf')
        for i in range(len(sentenceEmbeddings)): 
            if(cosineSimilarity(vec1, sentenceEmbeddings[i]) > sim):
                result = i
                sim = cosineSimilarity(vec1, sentenceEmbeddings[i])
        return result
    
    def closestSentenceWithSim(self, sentence, sentenceEmbeddings):
        '''Returns the index of the closest sentence to the input, 'sentence'.

        Inputs:
            sentence: A sentence

            sentenceEmbeddings: An mxn numpy array, where m is the total number
            of sentences and n is the dimension of the vectors.

        Returns:
            an integer i, where i is the row index in sentenceEmbeddings 
            that contains the closest sentence to the input
        '''

        result = 0
        vec1 = self.calcSentenceEmbeddingBaseline(sentence)
        sim = float('-inf')
        for i in range(len(sentenceEmbeddings)): 
            if(cosineSimilarity(vec1, sentenceEmbeddings[i]) > sim):
                result = i
                sim = cosineSimilarity(vec1, sentenceEmbeddings[i])
        return result, sim

    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(loadTrainingSentences(file_path))

        emb_sentence = self.calcSentenceEmbeddingBaseline(sentence)

        k = 10
        kNearest = []

        for i in range(sentenceEmbeddings.shape[0]):



            sim = cosineSimilarity(sentenceEmbeddings[i], emb_sentence)

            kNearest.append((i, sim))

        kNearest_ranked = sorted(kNearest, key=lambda x: x[1], reverse=True)

        labels = np.array([indexToSentence[j[0]][1] for j in kNearest_ranked[:k]])
        label = Counter(labels).most_common(1)[0][0]

        if kNearest_ranked[0][1] - kNearest_ranked[1][1] >= 0.07 or kNearest_ranked[0][1] > 0.9:
            label = indexToSentence[kNearest_ranked[0][0]][1]
        elif kNearest_ranked[0][1] < 0.7:
            label = 'no'

        return label

    def accuracy(self, training_file_path, dev_file_path):
        '''Returns the accuracy of your implementation of getCategory

        Inputs:
            training_file_path: path to training set

            dev_file_path: path to development set

        Let c = number of correctly categorized sentences in the development set.
        Let s = total number of sentences in the development set.

        Returns:
            A float equal to c/s.
        '''
        developing = loadTrainingSentences(dev_file_path)
        newDeveloping = {}
        developed = {}
        for d in developing:
            for c in developing[d]:
                newDeveloping[c] = d
        for d in newDeveloping:
            developed[d] = self.getCategory(d, training_file_path)
        diff = 0
        for d in newDeveloping:
            if(newDeveloping[d]!=developed[d]):
                diff = diff+1
        return 1- diff/len(newDeveloping)
        

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

        ### YOUR CODE HERE ###
        sen_lowered = command.lower()
        if 'holoemitter' in sen_lowered:
            slots["holoEmit"] = True

        if 'logical display' in sen_lowered:
            slots["logDisp"] = True


        if 'front light' in sen_lowered:
            slots["lights"] = ['front']
        elif 'back light' in sen_lowered:
            slots["lights"] = ['back']
        elif 'lights' in sen_lowered:
            slots["lights"] = ['front', 'back']

        if 'increase' in sen_lowered:
            slots['add'] = True

        if 'decrease' in sen_lowered:
            slots['sub'] = False

        if 'maximum' in sen_lowered:
            slots['on'] = True

        if 'minimum' in sen_lowered or 'off' in sen_lowered:
            slots['off'] = True

        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###

        sen_lowered = command.lower()

        if 'increase' in sen_lowered:
            slots['increase'] = True

        if 'decrease' in sen_lowered:
            slots['decrease'] = True


        if 'forward' in sen_lowered or 'north' in sen_lowered:
            slots['directions'].append('forward')
        if 'back' in sen_lowered or 'south' in sen_lowered:
            slots['directions'].append('back')
        if 'left' in sen_lowered or 'west' in sen_lowered:
            slots['directions'].append('left')
        if 'right' in sen_lowered or 'east' in sen_lowered:
            slots['directions'].append('right')

        return slots
