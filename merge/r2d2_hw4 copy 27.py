############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Mark Bloom"

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
    "Don't Move",
    "Go into drive mode",
    "Turn right",
    "Turn left",
    "Move right",
    "Move left",
    "Go full speed",
    "Spin",
    "Move backwards",
    "Turn 180 degrees"
]
my_light_sentences = [
    "Turn your back light red",
    "Turn your front light green",
    "50% holoemitter intensity",
    "Full logic display intensity",
    "Randomly change the front LED color",
    "Lights full intensity",
    "Let me see your logic display",
    "Turn on the back light",
    "Turn off the back light",
    "Lights on"
]
my_head_sentences = [
    "Spin your head around",
    "Spin your head completely",
    "Look to your right",
    "Look where you're rolling",
    "Look around",
    "Turn your head to the North",
    "Turn your head to where you're going",
    "Turn your head to your left",
    "Look to the South",
    "Turn your head 10 degrees to the right"
]
my_state_sentences = [
    "Is drive mode on?",
    "Are you waddling?",
    "What speed are you moving?",
    "What is the intensity of your logic display",
    "Are you connected?",
    "What color is your back light?",
    "Battery?",
    "How much red is in your front LED?",
    "How much blue is in your back LED",
    "How intense is your holoemitter?"
]
my_connection_sentences = [
    "Connect to the server",
    "Connect",
    "Who else is connected?",
    "Is anyone nearby?",
    "Connect r2d2 to the server",
    "Connect to r2q5",
    "Exit",
    "Exit the server",
    "Connect then disconnect",
    "Test the connection"
]
my_stance_sentences = [
    "Pick up your third wheel",
    "Set to waddle",
    "Stop waddling",
    "Waddle from now on",
    "Biped mode",
    "Turn off biped mode",
    "Waddle in biped mode",
    "Get off your third wheel",
    "Third wheel down",
    "Tiptoe"
]
my_animation_sentences = [
    "Play an alarm in 5 seconds",
    "Fall over in 5 seconds",
    "Scream in a minute",
    "Make any noise",
    "Animate yourself",
    "Get shaking",
    "Play dead",
    "LOL",
    "ROFL",
    "Emergency mode"
]

my_grid_sentences = [
    "The grid is 4 rows and 5 columns",
    "Move to (1,1)",
    "Watch out for (2,1)",
    "Go to square (4,1)",
    "You can't move from (1,1) to (2,1)",
    "Obstacle at (1,1)",
    "The grid is 10 feet big",
    "Every square is the same size of 1 foot",
    "You must go around (1,1)",
    "Get to (2,1) without (1,1)"
]

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    toReturn = []
    for word in sentence.split():
        curr_word = ""
        for c in word:
            if c in string.punctuation:
                if curr_word:
                    toReturn += [curr_word.lower()]
                curr_word = ""
            else:
                curr_word += c
        if curr_word:
            toReturn += [curr_word.lower()]
    return toReturn

def cosineSimilarity(vector1, vector2):
    prod = np.dot(vector1,vector2)
    len1,len2 = sum(vector1*vector1)**.5, sum(vector2*vector2)**.5
    return prod/(len1*len2)

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path) 

    def calcSentenceEmbeddingBaseline(self, sentence):
        words = tokenize(sentence)
        vect = np.array([0.0]*self.vectors.dim)
        if words:
            for w in words:
                vect += self.vectors.query(w)
        return vect

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
        sentenceEmbeddings = []
        indexToSentence = {}
        i = 0
        for category in commandTypeToSentences:
            sentences = commandTypeToSentences[category]
            for sen in sentences:
                sen_vect = self.calcSentenceEmbeddingBaseline(sen)
                sentenceEmbeddings.append(sen_vect)
                indexToSentence[i] = (sen, category)
                i += 1
        sentenceEmbeddings = np.array(sentenceEmbeddings)
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
        curr_i = -1
        maxSimilarity = -1.1
        in_vect = self.calcSentenceEmbeddingBaseline(sentence)
        for i in range(len(sentenceEmbeddings)):
            curr_vect = sentenceEmbeddings[i]
            curr_cosine = cosineSimilarity(in_vect, curr_vect)
            if curr_cosine > maxSimilarity:
                curr_i, maxSimilarity = i, curr_cosine
        return curr_i
                

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
        embed, indexToSentence = self.sentenceToEmbeddings(commandTypeToSentences)
        sugg_i = self.closestSentence(sentence, embed)
        sugg_category = indexToSentence[sugg_i][1]
        return sugg_category

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
feedback_question_1 = 5

feedback_question_2 = """
I liked this homework, but less than I did EC4
"""

feedback_question_3 = """
Maybe the coding was too difficult, rather than the concept?
"""
