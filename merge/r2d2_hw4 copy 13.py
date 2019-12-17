############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Monil Ghodasra and Jerrison Li"

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
my_driving_sentences = ["Start driving.", "Enter drive mode.", "Drive north.", "Drive forwards.", "Turn left.", "Go faster.", "Stop moving.", "Keep rolling.", "Decrease speed by 50%.", "Turn around."]
my_light_sentences = ["Turn on all lights.", "Turn off back light.", "Change all lights to blue.", "Turn on the front lights.", "Make the lights brighter.", "Dim the lights.", "Set RGB of all lights to 255,0,0.", "Blink the lights.", "Cycle through lights.", "Switch the colors of the LEDs."]
my_head_sentences = ["Turn your head to the left.", "Look to the left.", "Look right.", "Turn your head 360 degrees.", "Look behind you.", "Spin your head for 10 seconds.", "Look where you're going.", "Turn your head side to side.", "Spin your head once.", "Look 90 degrees to the right."]
my_state_sentences = ["What direction is your body facing?", "Where are you looking?", "What is your stance?", "Are you driving?", "How many feet are you standing on?", "What color is your front LED?", "What is your current speed?", "How much battery do you have?", "Are you awake?", "How bright is your holo projector?"]
my_connection_sentences = ["Connect D2-55A2 to the server.", "Disconnect.", "Are there any other droids near?", "Disconnect D2-55A2.", "Disconnect Q5-55A2.", "Connect Q5-55A2 to the server.", "Exit.", "Scan.", "Do you see a Q5 nearby?", "Who else is near?"]
my_stance_sentences = ["Set stance to be biped.", "Set stance to be triped.", "Stand on 2 wheels.", "Use all 3 wheels.", "Put your third wheel down.", "Waddle.", "Start waddling.", "Stop waddling.", "Lift your middle leg.", "Walk on two legs."]
my_animation_sentences = ["Scream.", "Fall over.", "Making beeping noises.", "Boop boop beep.", "Laugh.", "Play an alarm.", "Make sad noises.", "Talk to me.", "Play dead.", "Make some noise."]
my_grid_sentences = ["You are on a 4 by 4 grid.", "The grid is 4 by 4.", "The squares are 1 foot each.", "Go to position (3,3).", "You can't go on (2,1).", "Go right three squares.", "You can't go from (3,3) to (2,2).", "You are at (0,0).", "Go right then up then down.", "Get to (3,3)."]

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    tokens = []
    curr_token = ""
    for i in sentence:
        if i in string.punctuation or i in string.whitespace:
            if len(curr_token) > 0: tokens.append(curr_token.lower())
            curr_token = ""
        else:
            curr_token += i
    if len(curr_token) > 0:
        tokens.append(curr_token.lower())
    return tokens

def cosineSimilarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)
        sentence_vector = np.zeros(self.vectors.dim)
        for token in tokens:
            if token in self.vectors:
                sentence_vector = np.add(sentence_vector, self.vectors.query(token))
        return sentence_vector
            

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
        return (np.zeros(1), {None:None})

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
        return 0

    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        return ''

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
        return 0

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


############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 2

feedback_question_2 = """
There is no question.
"""

feedback_question_3 = """
There is no question.
"""
