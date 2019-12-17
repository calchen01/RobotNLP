############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Daniel Tao"

############################################################
# Imports
############################################################

from pymagnitude import *
from functools import reduce
from itertools import chain
from collections import Counter, defaultdict
from math import exp
import numpy as np
from numpy import dot
from numpy.linalg import norm
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

my_driving_sentences = [
    "Go forward for 3 feet, then turn left.",
    "North is at heading 40 degrees.",
    "Go South.",
    "Go West.",
    "Go North-by-northeast",
    "Run for it!",
    "Turn to heading 20 degrees.",
    "Reset your heading.",
    "Turn to face East.",
    "Start rolling backwards.",
    "Decrease your speed by 50%.",
    "Turn to your left.",
    "Stop moving.",
    "Set speed to be 1.",
    "Set speed to be 25%",
    "Rotate 180 degrees."]

my_light_sentences = [
    "Change the intensity on the holoemitter to minimum.",
    "Turn on the holoemitter.",
    "Turn off your logic display.",
    "Change the back LED to blue.",
    "Turn your back light yellow.",
    "Dim your holoemitter.",
    "Turn off your lights.",
    "Lights off.",
    "Set the RGB values on your lights to be (0,255,0).",
    "Add 100 to the blue value of your front LED.",
    "Increase the red value of your back LED by 20%.",
    "Display the following colors for 3 seconds each:  green, blue, purple, red, orange, yellow.",
    "Change the color on both LEDs to be red."]

my_head_sentences = [
    "Turn your head forward.",
    "Rotate your head 20 degrees to your left.",
    "Start spinning your head.",
    "Stop spinning your head.",
    "Look in front of you.",
    "Turn to look behind you.",
    "Turn your head 40 degrees to your right.",
    "Spin your head.",
    "Stop turning your head.",
    "Look behind yourself."]

my_state_sentences = [
    "What is your front light's color?",
    "Tell me what color your back light is set to.",
    "Is your holoemitter on?",
    "What's your stance?",
    "What's your orientation?",
    "What's your direction?",
    "Are you standing on 3 feet or 2?",
    "What's your current heading?",
    "How much battery do you have left right now?",
    "How's your battery doing?",
    "What's the status of your battery?",
    "Are you driving?",
    "How fast are you driving?",
    "What is your speed?",
    "Is your back light blue?",
    "Are you there?",
    "Are you alive?"]

my_connection_sentences = [
    "Connect to D2-66B3.",
    "Search for R2Q5 droids.",
    "Scan for droids."
    "Are there any other droids near?",
    "Are there other droids connected?",
    "Which droids are currently connected?",
    "Disconnect now.",
    "Disconnect from server.",
    "Scan for R2D2s.",
    "Find nearby droids.",
    "Connect to the server."]

my_stance_sentences = [
    "Stand on two legs.",
    "Retract your third wheel.",
    "Balance on two wheels.",
    "Get on your toes.",
    "Could you start rocking?",
    "Rock from left to right.",
    "Lean forward into biped stance.",
    "Stop rocking."
    "Undulate on your toes",
    "Please sit down.",
    "Waddle."]

my_animation_sentences = [
    "Fall.",
    "Scream!",
    "Make some noises!",
    "Start laughing.",
    "Sound an alarm.",
    "Turn off the alarm.",
    "Stop screaming.",
    "Stop making noise.",
    "Be quiet.",
    "Pretend to trip."]

my_grid_sentences = [
    'You are inside a 5 by 8 grid.',
    'Each square is 3 feet large.',
    'You are position (1,1).',
    'Go to position (4,2).',
    'There is an obstacle at position (3,1).',
    'Position (1,5) is blocked.',
    'Go to the right of the chair.',
    'It\'s not possible to go from (1,2) to (1,3).',
    'You cannot move from (4,2) to (3,2).',
    'Move to (6,3).']

stop_words = [
    'i', 
    'me', 
    'my', 
    'myself', 
    'we', 
    'our', 
    'ours', 
    'ourselves', 
    'you', 
    'your', 
    'yours', 
    'yourself', 
    'yourselves', 
    'he', 
    'him', 
    'his', 
    'himself', 
    'she', 
    'her', 
    'hers', 
    'herself', 
    'it', 
    'its', 
    'itself', 
    'they', 
    'them', 
    'their', 
    'theirs', 
    'themselves', 
    'what', 
    'which', 
    'who', 
    'whom', 
    'this', 
    'that', 
    'these', 
    'those', 
    'am', 
    'is', 
    'are', 
    'was', 
    'were', 
    'be', 
    'been', 
    'being', 
    'have', 
    'has', 
    'had', 
    'having', 
    'do', 
    'does', 
    'did', 
    'doing', 
    'a', 
    'an', 
    'the', 
    'and', 
    'but', 
    'if', 
    'or', 
    'because', 
    'as', 
    'until', 
    'while', 
    'of', 
    'at', 
    'by', 
    'for', 
    'with', 
    'about', 
    'against', 
    'between', 
    'into', 
    'through', 
    'during', 
    'before', 
    'after', 
    'above', 
    'below', 
    'to', 
    'from', 
    'up', 
    'down', 
    'in', 
    'out', 
    'on', 
    'off', 
    'over', 
    'under', 
    'again', 
    'further', 
    'then', 
    'once', 
    'here', 
    'there', 
    'when', 
    'where', 
    'why', 
    'how', 
    'all', 
    'any', 
    'both', 
    'each', 
    'few', 
    'more', 
    'most', 
    'other', 
    'some', 
    'such', 
    'no', 
    'nor', 
    'not', 
    'only', 
    'own', 
    'same', 
    'so', 
    'than', 
    'too', 
    'very', 
    's', 
    't', 
    'can', 
    'will', 
    'just', 
    'don', 
    'should', 
    'now']

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    token, tokens = "", []
    sentence = sentence.lower()
    for c in sentence:
        if c in string.punctuation:
            if token:
                tokens.append(token)
                token = ""
        elif c in string.whitespace:
            if token:
                tokens.append(token)
                token = ""
        else:
            token = token + c
    if token:
        tokens.append(token)
    return tokens

def cosineSimilarity(a, b):
    return dot(a, b)/(norm(a)*norm(b))


class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)
        self._stop = set(stop_words)

    def calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)
        if tokens:
            return reduce(np.add, self.vectors.query(tokens))
        return np.zeros(self.vectors.dim)

    def getEmbed(self, weights, freq, sentence):
        tokens = list(filter(lambda x: x not in self._stop, tokenize(sentence)))
        a = 0.001
        if tokens:
            x = np.concatenate([self.vectors.query(v).reshape(-1, self.vectors.dim) * (a / (a + (weights[v] / freq))) for v in tokens], axis=0)
            return np.average(x, axis=0)
        return np.zeros(self.vectors.dim)

    def sentenceToEmbeddings(self, c2s):
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
        m, y, X = 0, {}, None
        for k in c2s:
            for j in range(len(c2s[k])):
                y[m] = (c2s[k][j], k)
                m+=1 
            new_X = np.concatenate([self.calcSentenceEmbeddingBaseline(sent).reshape(-1, self.vectors.dim) for sent in c2s[k]], axis=0)
            if X is None:
                X = new_X
            else:
                X = np.concatenate((X, new_X), axis=0)
        return np.asarray([]).reshape(-1, self.vectors.dim) if X is None else X, y

    def sentEmbed(self, weights, freq, c2s):
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
        m, y, X = 0, {}, None
        for k in c2s:
            for j in range(len(c2s[k])):
                y[m] = (c2s[k][j], k)
                m+=1 
            new_X = np.concatenate([self.getEmbed(weights, freq, sent).reshape(-1, self.vectors.dim) for sent in c2s[k]], axis=0)
            if X is None:
                X = new_X
            else:
                X = np.concatenate((X, new_X), axis=0)
        return np.asarray([]).reshape(-1, self.vectors.dim) if X is None else X, y


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
        v = self.calcSentenceEmbeddingBaseline(sentence)
        return np.argmax((np.apply_along_axis(lambda x: cosineSimilarity(x, v), 1, sentenceEmbeddings)))

    
    def weights(self, sentences):
        sentences = sentences.values()
        sentences = chain(tokenize(sentence) for category in sentences for sentence in category)
        sentences = np.concatenate(list(sentences))
        ctr = Counter(sentences)
        return ctr, sum(ctr.values())

    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        if '?' in sentence:
            return "state"
        sentences = loadTrainingSentences(file_path)
        weights, freq = self.weights(sentences)
        X, y = self.sentEmbed(weights, freq, sentences)
        v = self.getEmbed(weights, freq, sentence)
        distances = np.apply_along_axis(lambda x: cosineSimilarity(x, v), 1, X)
        maxes = np.argpartition(distances, -11)[-11:]
        counter, tot = defaultdict(int), sum(distances[maxes])
        for i in maxes:
            _, label = y[i]
            counter[label] += distances[i] / tot
        label = max(counter, key=counter.get)
        prob = counter[label]
        return label if prob > .37 else "no"

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
        dat_test = loadTrainingSentences(dev_file_path)
        X_test, y_test = self.sentenceToEmbeddings(dat_test)
        count, correct = 0.0, 0.0
        for i in range(X_test.shape[0]):
            count+=1
            if y_test[i][1] == self.getCategory(y_test[i][0], training_file_path):
                correct+=1
        return correct / count


    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}
        command = tokenize(command.lower())
        if 'holoemitter' in command:
            slots['holoEmit'] = True
        if 'display' in command:
            slots['logDisp'] = True
        if 'front' in command or 'forward' in command:
            slots['lights'].append('front')
        if 'back' in command or 'backward' in command:
            slots['lights'].append('back')
        if 'lights' in command:
            slots['lights'] = ['front', 'back']
        if 'increase' in command or 'add' in command:
            slots['add'] = True
        if 'decrease' in command or 'subtract' in command or 'reduce' in command:
            slots['sub'] = True
        if 'on' in command or 'maximum' in command:
            slots['on'] = True
        if 'off' in command or 'minimum' in command:
            slots['off'] = True
        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}
        command = command.lower()
        if 'increase' in command:
            slots["increase"] = True
        if 'decrease' in command:
            slots["decrease"] = True
        for token in tokenize(command):
            if token == "forward" or token == "north":
                slots["directions"].append("forward")
            if token == "backward" or token == "south":
                slots["directions"].append("backward")
            if token == "left" or token == "west":
                slots["directions"].append("left")
            if token == "right" or token == "east":
                slots["directions"].append("right")
        return slots

# X = WordEmbeddings('../GoogleNews-vectors-negative300.magnitude')
# print(X.getCategory('Look behind you.', './data/r2d2TrainingSentences.txt'))

############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 8

feedback_question_2 = """
I enjoyed the challenge posed by part 2!
"""

feedback_question_3 = """
This assignment was very hard because it's too open ended, especially the slot part...
"""
