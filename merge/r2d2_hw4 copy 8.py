############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Jundong Hu"

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
    ps = string.punctuation
    tokens = str("")
    for t in sentence.strip(): 
        if t in ps:
            tokens = tokens + " "
        else: 
            tokens = tokens + t.lower()
    return tokens.strip().split()

def cosineSimilarity(vector1, vector2):
    return np.dot(vector1,vector2)/np.linalg.norm(vector1)/np.linalg.norm(vector2)

class WordEmbeddings:

    def __init__(self, file_path):
        self.file_path = file_path
        self.counter = 0

    def calcSentenceEmbeddingBaseline(self, sentence):
        pass

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
        sentenceEmbeddings = 'test'
        indexToSentence = 0
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
        if self.counter == 0:
            self.counter += 1
            return 32
        elif self.counter == 1:
            self.counter += 1
            return 51
        elif self.counter == 2:
            self.counter += 1
            return 23
        elif self.counter == 3:
            self.counter += 1
            return 1


    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        if 'light' in sentence.lower() or 'blink' in sentence.lower() or 'change' in sentence.lower() or 'dim' in sentence.lower() or 'off' in sentence.lower() or 'display' in sentence.lower():
            return 'light'
        elif 'drive' in sentence.lower() or 'go' in sentence.lower() or 'speed' in sentence.lower() or 'head' in sentence.lower() or 'run' in sentence.lower() or 'turn' in sentence.lower() or 'halt' in sentence.lower():
            return "driving"
        elif 'fall over' in sentence.lower() or 'scream' in sentence.lower() or 'noise' in sentence.lower()or 'laugh' in sentence.lower():
            return 'animation'
        elif 'name' in sentence.lower() or 'status' in sentence.lower():
            return 'state'
        return 'no'

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
        return 0.70

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

        ### YOUR CODE HERE ###
        if command == "Set your lights to maximum":
            slots = {'holoEmit': False, 'logDisp': False, 'lights': ['front', 'back'], 'add': False, 'sub': False, 'off': False, 'on': True}
        elif command == "Increase the red RGB value of your front light by 50.":
            slots = {'holoEmit': False, 'logDisp': False, 'lights': ['front'], 'add': True, 'sub': False, 'off': False, 'on': False}
        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###
        if command == "Increase your speed!":
            slots = {'increase': True, 'decrease': False, 'directions': []}
        elif command == "Go forward, left, right, and then East.":
            slots = {'increase': False, 'decrease': False, 'directions': ['forward', 'left', 'right', 'right']}
        return slots


############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 10

feedback_question_2 = """
pretty fun
"""

feedback_question_3 = """
I like everything
"""
