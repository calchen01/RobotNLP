############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Yonah Mann"

############################################################
# Imports
############################################################

from pymagnitude import *
import numpy as np
import string
import re

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

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    s = ""
    to_return = []
    for i in sentence:
        if i in string.whitespace:
            if s != "":
                to_return.append(s)
            s = ""
        elif i in string.punctuation:
            if (s != ""):
                to_return.append(s)
            s = ""
        else:
            s += str(i)
    if s != "":
        to_return.append(s.lower())
    return to_return

def cosineSimilarity(vector1, vector2):
    numerator = np.dot(vector1, vector2)
    denominator = np.linalg.norm(vector1, ord=2) * np.linalg.norm(vector2, ord=2)
    return numerator / denominator

class WordEmbeddings:

    def __init__(self, file_path):
        self.magnitude = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        shape = self.magnitude.query("the").shape
        tokens = tokenize(sentence)
        if(len(tokens) == 0):
            return np.zeros(shape)
        return sum([self.magnitude.query(word) for word in tokens])

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
        sentence_embeddings = []
        index_to_sentence = {}
        index = 0
        counter = 0
        for category in commandTypeToSentences:
            sentences = commandTypeToSentences[category]
            for sentence in sentences:
                embedding = self.calcSentenceEmbeddingBaseline(sentence)
                sentence_embeddings.append(embedding)
                chosen_index = index
                index_to_sentence[chosen_index] = (sentence, category)
                index += 1
        return (sentence_embeddings, index_to_sentence)

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
        pass

    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        pass

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
        pass

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

        ### YOUR CODE HERE ###

        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###

        return slots

# X = WordEmbeddings("/Users/ymann/Downloads/GoogleNews-vectors-negative300.magnitude")
# trainingSentences = loadTrainingSentences("/Users/ymann/Downloads/r2d2_hw4/data/r2d2TrainingSentences.txt")
# sentenceEmbeddings, indexToSentence = X.sentenceToEmbeddings(trainingSentences)
# # print(sentenceEmbeddings[14:])
# print(indexToSentence[14])
############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 2

feedback_question_2 = """
Didn't finish
"""

feedback_question_3 = """
Didn't finish
"""
