############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Yeonjune Kang"

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
    "You are driving too fast",
    "There is a speed limit ahead",
    "You are driving too slow",
    "Stop right now!",
    "Slow down when you see yellow light",
    "Set speed to 50",
    "Go South",
    "Turn 30 degrees counter-clockwise",
    "GO West",
    "Start going in reverse direction",
]
my_light_sentences = [
    "Turn on all your lights",
    "Blink red light twice",
    "Alternate between red and blue lights",
    "Turn off all the back lights",
    "Turn on the front light only",
    "Change the back light to green",
    "Turn off the logic display",
    "Dim the lights 50%",
    "Set the RGB values to 30, 30, 30",
    "Change all light intensities to maximum",
]
my_head_sentences = [
    "Turn head 90% clockwise",
    "Turn head right",
    "There's someone behind you",
    "Turn slightly left",
    "Turn head 30% clockwise",
    "Turn head 90% counter clockwise",
    "Tilt your head up 30%",
    "lift your head",
    "Look down",
    "Look straight ahead",
]
my_state_sentences = [
    "How long have you been awake?",
    "Is your battery at least half charged?",
    "Are you facing North?",
    "Are you facing South?",
    "What is your battery status?",
    "Is your headlight on?",
    "Is your back light on?",
    "What color is your back light?",
    "Is your logic board on?",
    "What is your current driving speed?",
]
my_connection_sentences = [
    "Connect to server",
    "Disconnect to server",
    "Connect",
    "Disconnect",
    "Are there other bots nearby?",
    "Reconnect",
    "Reconnect to server",
    "Shut down connection",
    "Check connection",
    "Report connection status",
]
my_stance_sentences = [
    "Fold your third wheel",
    "Stand on tippy toes",
    "Unfold third wheel",
    "Fold fourth wheel",
    "Unfold fourth wheel",
    "Tilt body left",
    "Tilt your body right",
    "Make your stance like a tripod",
    "Get ready to run",
    "Make your stance biped",
]
my_animation_sentences = [
    "Start running",
    "Go around in circles",
    "Play a song",
    "Stop playing the song",
    "Fall down",
    "Dance",
    "Dance to Justin Bieber's newest song",
    "Shout!",
    "Bring the cup on the table",
    "search the room and find the fridge",
]
my_grid_sentences = [
    "You are on position (2,1)",
    "There is a fridge on position (3,3)",
    "There are two tables on position (4,1) and (4,2)",
    "Go to position (5,5)",
    "Go back to previous position",
    "Go to position (1,1) in 3 minutes",
    "Go to initial position",
    "Go stand behind the chair",
    "Go under the high chair",
    "There is an obstacle right in front of you",
]


############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"


def tokenize(sentence):
    for c in sentence:
        if c in string.punctuation:
            sentence = sentence.replace(c, " ")
    tokens = sentence.split(" ")
    output = []
    for token in tokens:
        temp = token.strip()
        if temp != "":
            output.append(str.lower(temp))
    return output


def cosineSimilarity(vector1, vector2):
    dot_prod = np.dot(vector1, vector2)
    sq1 = np.sum(np.square(vector1))
    sq2 = np.sum(np.square(vector2))
    output = dot_prod / (np.sqrt(sq1) * np.sqrt(sq2))
    return output


class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        if len(sentence) == 0:
            return [0 for x in range(self.vectors.dim)]
        tokens = tokenize(sentence)
        vectors = self.vectors.query(tokens)
        output = [0 for x in range(len(vectors[0]))]
        for vector in vectors:
            for i in range(len(vector)):
                output[i] += vector[i]
        return output

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
        pass

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


# path = '/Users/june_kang/Downloads/'
# vectors = Magnitude(path + "GoogleNews-vectors-negative300.magnitude")
# v = vectors.query("cat")
# w = vectors.query("dog")
#
# print(cosineSimilarity(v, w))

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
