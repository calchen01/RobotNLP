
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Chloe Sheen"

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
    "Go forward for 3 feet, then turn left.",
    "North is at heading 50 degrees.",
    "Go South.",
    "Go East.",
    "Go South-by-southeast",
    "Run away!",
    "Turn to heading 50 degrees.",
    "Reset your heading to 0",
    "Turn to face South.",
    "Start rolling forward."]

my_light_sentences = [
    "Change the intensity on the holoemitter to maximum.",
    "Turn off the holoemitter.",
    "Blink your logic display.",
    "Change the back LED to green.",
    "Turn your back light green.",
    "Dim your lights holoemitter.",
    "Turn off all your lights.",
    "Lights out.",
    "Set the RGB values on your lights to be 255,0,0.",
    "Add 100 to the red value of your front LED."
]

my_head_sentences = [
    "Turn your head to face forward.",
    "Look behind you.",
    "Turn to face backwards.",
    "Look in front of you.",
    "Look ahead.",
    "Look behind.",
    "Look backwards.",
    "Turn to look backwards,",
    "Look ahead in front.",
    "Look not towards the back."
]

my_state_sentences = [
    "What color is your front light?",
    "Tell me what color your front light is set to.",
    "Is your logic display on?",
    "What is your stance?",
    "What is your orientation?",
    "What direction are you facing?",
    "Are you standing on 2 feet or 3?",
    "What is your current heading?",
    "How much battery do you have left?",
    "What is your battery status?"
]

my_connection_sentences = [
    "Connect D2-55A2 to the server",
    "Are there any other droids nearby?",
    "Disconnect.",
    "Disconnect from the server.",
    "Connect again.",
    "Reconnect.",
    "Connect back to the server.",
    "Scan for droids.",
    "Scan and connect D2-55A2.",
    "Disconnect D2-55A2."
]
my_stance_sentences = [
    "Set your stance to be biped.",
    "Put down your third wheel.",
    "Stand on your tiptoes.",
    "Put down your wheel",
    "Waddle",
    "Wobble",
    "Wobble and stand on your tiptoes",
    "Stop waddling",
    "Waddle and wobble",
    "Begin wobbling"
]
my_animation_sentences = [
    "Fall over",
    "Scream",
    "Make some noise",
    "Laugh",
    "Play an alarm",
    "Play some sound",
    "Make a laughing sound",
    "Fall and laugh",
    "Play some noise",
    "Laugh and scream",
]
my_grid_sentences = [
    "You are on a 4 by 5 grid.",
    "Each square is 1 foot large.",
    "You are at position (0,0).",
    "Go to position (3,3).",
    "There is an obstacle at position 2,1.",
    "There is a chair at position 3,3",
    "Go to the left of the chair.",
    "It’s not possible to go from 2,2 to 2,3.", 
    "Go back to position (3,1).",
    "Go to the right of the obstacle."
]


############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    return " ".join("".join([" " if char in string.punctuation else char for char in sentence]).split()).lower().split()

def cosineSimilarity(vector1, vector2):
    vector_length = np.dot(vector1,vector2) 
    dot_prod = (np.sqrt(np.dot(vector2,vector2)) * np.sqrt(np.dot(vector1,vector1)))
    return vector_length / dot_prod

class WordEmbeddings:

    def __init__(self, file_path):
        pass

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
