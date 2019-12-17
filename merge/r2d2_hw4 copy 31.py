############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Type your full name here."

############################################################
# Imports
############################################################

from pymagnitude import *
import sklearn.neighbors
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

my_driving_sentences = [str(i) for i in range(11)]

my_light_sentences = [str(i) for i in range(11)]

my_head_sentences = [str(i) for i in range(11)]

my_state_sentences = [str(i) for i in range(11)]

my_connection_sentences = [str(i) for i in range(11)]

my_stance_sentences = [str(i) for i in range(11)]

my_animation_sentences = [str(i) for i in range(11)]

my_grid_sentences = [str(i) for i in range(11)]

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    words = re.findall(r"[\w]+", sentence)
    return [x.lower() for x in words]

def cosineSimilarity(vector1, vector2):
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2))

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        words = tokenize(sentence)
        if words == []: return np.zeros(300)
        return self.vectors.query(words).sum(axis = 0)

    def sentenceToEmbeddings(self, commandTypeToSentences):
        '''Returns a tuple of sentence embeddings and an index-to-(sentence, category)
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

            indexToSentence: A dictionary with key: index i, value: (sentence, category).
        '''
        indexToSentence = {}

        i = 0
        for category in commandTypeToSentences:
            sentences = commandTypeToSentences[category]
            for sentence in sentences:
                indexToSentence[i] = (sentence, category)
                i += 1

        sentenceEmbeddings = np.zeros((len(indexToSentence), self.vectors.dim))

        for i in range(len(indexToSentence)):
            sentence = indexToSentence[i][0]
            sentenceEmbedding = self.calcSentenceEmbeddingBaseline(sentence)

            sentenceEmbeddings[i, :] = sentenceEmbedding

        return sentenceEmbeddings, indexToSentence

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
        sentenceEmbedding = self.calcSentenceEmbeddingBaseline(sentence)

        maxSimilarity = float("-inf")
        maxIndex = -1

        for i in range(sentenceEmbeddings.shape[0]):
            similarity = cosineSimilarity(sentenceEmbedding, sentenceEmbeddings[i, :])
            if similarity > maxSimilarity:
                maxSimilarity, maxIndex = similarity, i

        return maxIndex


    def calcSentenceEmbeddingBaseline2(self, sentence):
        # helper function for getCategory
        words = re.findall(r"[\w']+|[?]", sentence)
        words = [x.lower() for x in words if x.lower() not in ["", "a", "an", "the", "is", "to", "set"]]
        return self.vectors.query(words).sum(axis = 0)

    def sentenceToEmbeddings2(self, commandTypeToSentences):
        # near copy of above sentence embeddings, for calcSentenceEmbeddingBaselines2
        indexToSentence = {}

        i = 0
        for category in commandTypeToSentences:
            sentences = commandTypeToSentences[category]
            for sentence in sentences:
                indexToSentence[i] = (sentence, category)
                i += 1

        sentenceEmbeddings = np.zeros((len(indexToSentence), self.vectors.dim))

        for i in range(len(indexToSentence)):
            sentence = indexToSentence[i][0]
            sentenceEmbedding = self.calcSentenceEmbeddingBaseline2(sentence)

            sentenceEmbeddings[i, :] = sentenceEmbedding

        return sentenceEmbeddings, indexToSentence

    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        commandEmbedding = self.calcSentenceEmbeddingBaseline2(sentence)

        commandTypeToSentences = loadTrainingSentences(file_path)

        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings2(commandTypeToSentences)

        sortList = []

        for i in range(sentenceEmbeddings.shape[0]):
            similarity = cosineSimilarity(commandEmbedding, sentenceEmbeddings[i, :])
            sortList.append((i, similarity))

        similarSentences = sorted(sortList, key = lambda x: x[1], reverse = True)

        closestSentences = [x[0] for x in similarSentences]

        commandDict = {}
        for category in commandTypeToSentences:
            commandDict[category] = 0

        commandDict[indexToSentence[closestSentences[0]][1]] += 1
        commandDict[indexToSentence[closestSentences[1]][1]] += 0.5
        commandDict[indexToSentence[closestSentences[2]][1]] += 0.5
        commandDict[indexToSentence[closestSentences[3]][1]] += 0.2
        commandDict[indexToSentence[closestSentences[4]][1]] += 0.2

        if cosineSimilarity(commandEmbedding, sentenceEmbeddings[closestSentences[0], :]) < 0.70:
            return "no"

        return max(commandDict, key=commandDict.get)

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

        countTotal = 0
        countRight = 0
        developmentSentences = loadTrainingSentences(dev_file_path)
        for category in developmentSentences:
            for sentence in developmentSentences[category]:
                countTotal += 1
                if self.getCategory(sentence, training_file_path) == category:
                    countRight += 1

        return countRight / countTotal

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        # slot filler for lights
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

        ### YOUR CODE HERE ###

        command = command.lower()

        if "holoemitter" in command or "holo emitter" in command:
            slots["holoEmit"] = True
        if "logic display" in command:
            slots["logDisp"] = True

        # WANT TO MAKE INCREASE BETTER
        if "increase" in command or "add" in command:
            slots["add"] = True
        if "decrease" in command or "reduce" in command or "subtract" in command:
            slots["sub"] = True

        # front back too similar
        if "back" in command:
            slots["lights"].append("back")
        if "front" in command or "forward" in command:
            slots["lights"].append("front")
        if slots["lights"] == []:
            slots["lights"] = ["front", "back"]

        words = re.split('\W+', command)
        words = [x for x in words if x != ""]

        for word in words:
            if self.vectors.similarity("off", word) > 0.7 or "minimum" in command:
                slots["off"] = True
            elif self.vectors.similarity("on", word) > 0.7 or self.vectors.similarity("maximum", word) > 0.7:
                slots["on"] = True

        return slots

    def drivingParser(self, command):
        # slot filler for driving
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###

        command = command.lower()
        
        if re.search(r"\b(increase|faster)\b", command, re.I):
            slots["increase"] = True
        elif re.search(r"\b(decrease|slower|slow|lower)\b", command, re.I):
            slots["decrease"] = True
       
        tokens = re.split("[^a-zA-Z]", command)
        for token in tokens:
            if token in {"up", "forward", "ahead", "straight", "north"}:
                slots["directions"].append("forward")
            elif token in {"down", "back", "south", "backward"}:
                slots["directions"].append("back")
            elif token in {"left", "west"}:
                slots["directions"].append("left")
            elif token in {"right", "east"}:
                slots["directions"].append("right")

        return slots


############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 0

feedback_question_2 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""

feedback_question_3 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""
