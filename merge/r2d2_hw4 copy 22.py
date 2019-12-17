############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Nayeong Kim"

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

my_driving_sentences = ["Move north", "Move south", "Move east", "Move west", "Go straight", "Stop now", "Turn now", "U-turn", "Move faster", "Move slower"]
my_light_sentences = ["turn off all the lights", "turn on all the lights", "turn on all lights for 2 seconds", "brighten up the lights", "dim down the lights", "red", "blue", "green", "blink for 2 seconds", "display random colors"]
my_head_sentences = ["face forward", "look behind", "face your right", "face your left", "turn to the right", "turn to your left", "make a full spin", "stop turning", "keep turning", "move to any direction"]
my_state_sentences = ["are you on?", "what is the color of your front light?", "what is the color of your back light?", "what direction are you moving", "at what speed are you moving", "how much battery life do you have?", "do you have red lights on?", "do you have green lights on?", "do you have blue lights on?", "are you moving right now?"]
my_connection_sentences = ["connect to server", "connect", "disconnect", "disconnect to server", "are there other droids nearby?", "how many droids are there near by?", "reconnect to server", "half connection", "exit connection", "check if there are other droids"]
my_stance_sentences = ["waddle", "put down your first wheel", "put down your second wheel", "put down your thrid wheel", "put down your forth wheel", "waddle harder", "start waddling", "stop waddling", "keep waddling", "waddle slowly"]
my_animation_sentences = ["make some noise", "shut up", "yell", "scream", "say something", "play a sound", "alarm on", "talk", "play sound", "mute"]
my_grid_sentences = ["go to origin", "go to (0,1)", "go to (2,1)", "go to (2,3)", "go to (3,3)", "reset position to (0,0)", "go the right of obstacle", "go to left of obstacle", "go to north of obstacle", "go to south of obstacle"]

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    words = re.findall(r"[\w]+", sentence)
    return [x.lower() for x in words]

def cosineSimilarity(vector1, vector2):
    return np.dot(vector1, vector2) / np.linalg.norm(vector1) / np.linalg.norm(vector2)

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        words = tokenize(sentence)
        if words:
            return self.vectors.query(words).sum(axis = 0)
        else:
            return np.zeros(300)


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
        for category, sentences in commandTypeToSentences.items():
            for sentence in sentences:
                indexToSentence[i] = (sentence, category)
                i += 1;
        l = len(indexToSentence)
        sentenceEmbeddings = np.zeros((l, self.vectors.dim))
        for index, (category, sentence) in indexToSentence.items():
            sentenceEmbeddings[index, :] = self.calcSentenceEmbeddingBaseline(sentence)

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
        maxSim = float("-inf")
        maxIndex = -1
        for i in range(sentenceEmbeddings.shape[0]):
            sim = cosineSimilarity(self.calcSentenceEmbeddingBaseline(sentence), sentenceEmbeddings[i, :])
            if maxSim < sim:
                maxSim = sim
                maxIndex = i

        return maxIndex

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
        c = 0
        s = 0
        commandType = loadTrainingSentences(dev_file_path)
        for category in commandType:
            for sentence in commandType[category]:
                if self.getCategory(sentence, training_file_path) == category:
                    c += 1
                s += 1;

        return c / s

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": True}

        if re.search(r"\b(holoEmit|holo emitter|holoemitter)\b", command, re.IGNORECASE): 
            slots["holoEmit"] = True
        if re.search(r"\b(logDisp|logic display)\b", command, re.IGNORECASE): 
            slots["logDisp"] = True
        if re.search(r"\b(forward|front)\b", command, re.IGNORECASE): 
            slots["lights"].append("front")
        if re.search(r"\b(back|backward|behind)\b", command, re.IGNORECASE): 
            slots["lights"].append("back")
        if not slots["lights"]:
            slots["lights"].append("front")
            slots["lights"].append("back")
        if re.search(r"\b(increase|add|brighter|brighten)\b", command, re.IGNORECASE): 
            slots["add"] = True
        if re.search(r"\b(decrease|dim)\b", command, re.IGNORECASE): 
            slots["sub"] = True
        if re.search(r"\b(off|shut)\b", command, re.IGNORECASE): 
            slots["off"] = True
            slots["on"] = False

        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}
        
        if re.search(r"\b(increase|fast|high|faster|higher)\b", command, re.IGNORECASE):
            slots["increase"] = True
        elif re.search(r"\b(decrease|slow|low|slower|lower)\b", command, re.IGNORECASE):
            slots["decrease"] = True
        tokens = tokenize(command)
        for token in tokens:
            if re.search(r"(forward|front|straight|north)", token, re.IGNORECASE): 
                slots["directions"].append("forward")
            elif re.search(r"(back|south|backward|behind)", token, re.IGNORECASE): 
                slots["directions"].append("back")
            elif re.search(r"(right|east)", token, re.IGNORECASE): 
                slots["directions"].append("right")
            elif re.search(r"(left|west)", token, re.IGNORECASE): 
                slots["directions"].append("left")

        return slots

############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 10

feedback_question_2 = """
Understanding how categories and sentences are stored as data types """

feedback_question_3 = """
Testing out different command similarities
"""
