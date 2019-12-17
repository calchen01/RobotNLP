############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Type your full name here."

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

magnitudeFile = "glove"

def tokenize(sentence):
    tokens = re.findall(r"[\w]+", sentence)
    return [t.lower() for t in tokens]

def cosineSimilarity(vector1, vector2):
    return np.dot(vector1, vector2)/ (np.linalg.norm(vector1)*np.linalg.norm(vector2))

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)
        if not tokens:
            return np.zeros(self.vectors.dim)
        return self.vectors.query(tokens).sum(axis=0)

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
        indexToSentence = {}
        i = 0

        for cat in commandTypeToSentences:
            for sentence in commandTypeToSentences[cat]:
                indexToSentence[i] = (sentence, cat)
                i += 1

        sentenceEmbeddings = np.zeros((len(indexToSentence), self.vectors.dim))
        for i in indexToSentence:
            sentenceEmbeddings[i] = self.calcSentenceEmbeddingBaseline(indexToSentence[i][0])

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
        closest = -1
        max_similarity = float('-inf')
        sentence_embedding = self.calcSentenceEmbeddingBaseline(sentence)

        for i in range(len(sentenceEmbeddings)):
            similarity = cosineSimilarity(sentence_embedding, sentenceEmbeddings[i])
            if similarity > max_similarity:
                max_similarity = similarity
                closest = i
        return closest

    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        commandTypeToSentences = loadTrainingSentences(file_path)
        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(commandTypeToSentences)
        closest = self.closestSentence(sentence, sentenceEmbeddings)
        sentence_embedding = self.calcSentenceEmbeddingBaseline(sentence)
        if cosineSimilarity(sentence_embedding, sentenceEmbeddings[closest]) < 0.85:
            return 'no'
        return indexToSentence[closest][1]

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
        DevSentences = loadTrainingSentences(dev_file_path)
        num_total = 0
        num_accurate = 0

        for cat in DevSentences:
            for sentence in DevSentences[cat]:
                num_total += 1
                pred_cat = self.getCategory(sentence, training_file_path)
                if pred_cat == cat:
                    num_accurate += 1
        return num_accurate / num_total


    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], 
                 "add": False, "sub": False, "off": False, "on": False}

        ### YOUR CODE HERE ###
        command = command.lower()

        if 'holo' in command and 'emit' in command:
            slots['holoEmit'] = True
        if 'log' in command and 'disp' in command:
            slots['logDisp'] = True

        command = re.split(r'\W+', command)
        command = [x for x in command if x ]

        if 'back' in command:
            slots['lights'].append('back')
        if 'front' in command or 'forward' in command:
            slots['lights'].append('front')
        if not slots['lights']:
            slots['lights'] = ['front', 'back']

        if 'increase' in command or 'add' in command:
            slots['add'] = True
        if 'decrease' in command or 'subtract' in command or 'reduce' in command:
            slots['sub'] = True

        if 'off' in command or 'minimum' in command:
            slots['off'] = True
        if 'on' in command or 'maximum' in command:
            slots['on'] = True

        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left".
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###
        command = re.split(r'\W+', command.lower())
        command = [x for x in command if x ]

        if 'increase' in command or 'fast' in command:
            slots['increase'] = True
        if 'decrease' in command or 'slow' in command or 'lower' in command:
            slots['decrease'] = True

        for word in command:
            if word == 'up' or word == 'forward' or word == 'ahead' or word == 'straight' or word == 'north':
                slots['directions'].append('forward')
            elif word == 'down' or word == 'south' or word == 'back':
                slots['directions'].append('back')
            elif word == 'left' or word == 'west':
                slots['directions'].append('left')
            elif word == 'right' or word == 'east':
                slots['directions'].append('right')

        return slots

############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 10

feedback_question_2 = """
Hello
"""

feedback_question_3 = """
Hello
"""
