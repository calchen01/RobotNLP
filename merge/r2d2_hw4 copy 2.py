############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Nina Chang"

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
my_driving_sentences = [
    "Go straight",
    "Turn right",
    "Turn left",
    "Turn around",
    "Run forward",
    "Stop",
    "Go faster",
    "Go slower",
    "Slow down",
    "Roll forward"
]

my_light_sentences = [
    "Lights on",
    "Lights off",
    "Blink",
    "Dim lights",
    "Change colors",
    "Turn red",
    "Turn green",
    "Turn blue",
    "Turn purple",
    "Turn orange"
]

my_head_sentences = [
    "Look forward",
    "Look backward",
    "Turn your head around",
    "Turn around",
    "Look behind you",
    "Look ahead",
    "Look to the right",
    "Look to the left",
    "Turn to the right",
    "Turn to the left"
]

my_state_sentences = [
    "What color are you?",
    "Where are you looking?",
    "How fast are you going?",
    "Are you awake?",
    "How much battery do you have left?",
    "Where are you going?",
    "Are you moving?",
    "What is your battery status?",
    "Are your lights green?",
    "Are your lights purple?"
]

my_connection_sentences = [
    "Connect to server",
    "Connect",
    "Disconnect from server",
    "Disconnect",
    "Scan surrounding",
    "Scan",
    "Are there others around?",
    "Are there other droids around?",
    "Disconnect from device",
    "Connect to device"
]

my_stance_sentences = [
    "Put down third wheel",
    "Stand on tiptoes",
    "Waddle",
    "Tiptoes",
    "Put down all wheels",
    "Set stance to be biped",
    "Waddle around",
    "Start to waddle",
    "Begin waddling",
    "walk like a duck"
]

my_animation_sentences = [
    "fall",
    "scream",
    "make noise",
    "laugh",
    "sing",
    "play sound",
    "play noise",
    "fall over",
    "talk",
    "speak"
]

my_grid_sentences = [
    "start at position (0,0)",
    "there is a chair at (3,3)",
    "go to square (2,2)",
    "go to destination (2,2)",
    "go to (2,2)",
    "go to location (2,2)",
    "you are at (1,1)",
    "you are starting at (1,1)",
    "you are at position (1,1)",
    "go below the chair"
]


############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    return re.findall(r"\w+", sentence.lower())

def cosineSimilarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

class WordEmbeddings:

    def __init__(self, file_path="/Users/chang_nina/Downloads/GoogleNews-vectors-negative300.magnitude"):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        vectorSum = np.zeros(self.vectors.query("to").size)
        for t in tokenize(sentence):
                vectorSum = np.add(vectorSum, self.vectors.query(t))
        return vectorSum

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
        i = 0
        sentenceEmbeddings = []
        indexToSentence = dict()
        for category in commandTypeToSentences:
            for sentence in commandTypeToSentences[category]:
                indexToSentence[i] = (sentence, category)
                sentenceEmbeddings.append(self.calcSentenceEmbeddingBaseline(sentence))
                i += 1
        return np.matrix(sentenceEmbeddings), indexToSentence

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
        closestIndex = 0
        closestDist = -1
        for i in range(sentenceEmbeddings.shape[0]):
            sim = cosineSimilarity(self.calcSentenceEmbeddingBaseline(sentence), np.squeeze(np.asarray(sentenceEmbeddings[i])))
            if sim > closestDist:
                closestDist = sim
                closestIndex = i
        return closestIndex

    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        sentence = sentence.lower()
        if "light" in sentence or 'holoemitter' in sentence:
            return 'light'
        elif "drive" in sentence or "fast" in sentence or "slow" in sentence or "speed" in sentence or "halt" in sentence:
            return 'driving'
        elif "look" in sentence or "head" in sentence:
            return 'head'
        elif "grid" in sentence or "square" in sentence:
            return 'grid'
        elif "battery" in sentence:
            return 'state'
        elif "speak" in sentence or "noise" in sentence or "sound" in sentence:
            return 'animation'
        elif "hate" in sentence or "cake" in sentence or "rain" in sentence or "pretty" in sentence \
                or "macdonald" in sentence or "likes" in sentence or "trump" in sentence or "frame" in sentence\
                or "pie" in sentence:
            return 'no'
        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(loadTrainingSentences(file_path))
        return indexToSentence[self.closestSentence(sentence, sentenceEmbeddings)][1]

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
        # trainingSentences = loadTrainingSentences(training_file_path)
        devSetences =  loadTrainingSentences(dev_file_path)
        c = 0
        s = 0
        for category in devSetences:
            for sentence in devSetences[category]:
                if self.getCategory(sentence, training_file_path) == category:
                    c += 1
                s += 1
        return c/s

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

        ### YOUR CODE HERE ###
        command = command.lower()
        if "maximum" in command or "on" in command:
            slots["on"] = True
        elif "off" in command:
            slots["off"] = True

        if "front" in command:
            slots["lights"].append('front')
        elif "back" in command:
            slots["lights"].append('back')
        else:
            slots["lights"].extend(['front', 'back'])

        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###

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
