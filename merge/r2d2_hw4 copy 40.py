############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Joslyn Jung"

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
my_driving_sentences = ["Set speed to half of current speed.", "Run at max speed!", "Go forward for 2 feet.",
                        "North is at heading 90 degrees.", "Go South.", "Go West.", "Go North-by-northeast", "Run!",
                        "Turn to heading 0 degrees.", "Reset your heading."]
my_light_sentences = ["Blink twice.", "Change the intensity on the holoemitter to maximum.", "Turn off the holoemitter.",
                      "Blink your logic display.", "Change the back LED to green.", "Turn your back light green.",
                      "Dim your lights holoemitter.", "Turn off all your lights.", "Lights out.",
                      "Set the RGB values on your lights to be 255,0,0."]
my_head_sentences = ["Look around you.", "Turn your head to face forward.", "Look behind you.",
                     "Look right.", "Look left.", "Turn your head all the way around.", "Turn your head to face backwards",
                     "Look left then right.", "Turn your head to face right.", "Look right then left."]
my_state_sentences = ["What heading are you facing?", "What color is your back light?",
                      "What color is your front light?", "Tell me what color your front light is set to.",
                      "Is your logic display on?", "What is your stance?", "What is your orientation?",
                      "What direction are you facing?", "Are you standing on 2 feet or 3?",
                      "How much battery do you have left?"]
my_connection_sentences = ["Disconnect from server.", "Connect D2-55A2 to the server", "Are there any other droids nearby?",
                           "Disconnect.", "Connect.", "Connect to the server.", "Are there other droids?",
                           "Disconnect D2-55A2 from the server.", "Are there any other droids?", "Disconnect D2-55A2."]
my_stance_sentences = ["Set stance to triped.", "Set your stance to be biped.", "Put down your third wheel.",
                       "Stand on your tiptoes.", "Stand flat on your feet.", "Switch stance to the other one.",
                       "Pick up your third wheel", "Stand on your tiptoes then stand flat on your feet.",
                       "Switch stance.", "Stand on three feet!"]
my_animation_sentences = ["Squeak!", "Fall over", "Scream", "Make some noise", "Laugh", "Play an alarm",
                          "Beep twice.", "Stand up.", "Make noise!", "Play some noise."]
my_grid_sentences = ["There is an obstacle 1 square in front of you.", "You are on a 4 by 5 grid.",
                     "Each square is 1 foot large.", "You are at position (0,0).", "Go to position (3,3).",
                     "There is an obstacle at position 2,1.", "There is a chair at position 3,3",
                     "Go to the left of the chair.", "It’s not possible to go from 2,2 to 2,3.",
                     "There's something 2 squares behind you."]


############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    str = '[^a-zA-Z0-9]'
    regex = re.compile(str)
    return regex.sub(' ', sentence).lower().split()

def cosineSimilarity(vector1, vector2):
    dot = np.sum(np.multiply(vector1, vector2))
    v1_2 = np.sum(np.square(vector1))
    v2_2 = np.sum(np.square(vector2))
    return dot/(np.sqrt(v1_2*v2_2))

class WordEmbeddings:
    vectors = None

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        words = tokenize(sentence)
        size = np.size(self.vectors.query("cat"))
        sentence_vec = np.zeros(size)
        if len(words) == 0:
            return sentence_vec
        for word in words:
            sentence_vec = np.add(self.vectors.query(word), sentence_vec)
        return sentence_vec

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
        size = self.vectors.query("cat").size
        sentenceEmbeddings = np.zeros((0, size))
        indexToSentence = {}
        arr = []
        i = 0
        for category, sentence_list in commandTypeToSentences.items():
            for sentence in sentence_list:
                arr.append(self.vectors.query(sentence))
                indexToSentence[i] = (sentence, category)
                i += 1
        if len(arr) > 0:
            sentenceEmbeddings = np.array(arr)
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
        closeness = 0
        max_ind = 0
        ind = 0
        for sen in sentenceEmbeddings:
            score = cosineSimilarity(self.vectors.query(sentence), sen)
            if score > closeness:
                closeness = score
                max_ind = ind
            ind += 1
        return max_ind

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
        # sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(loadTrainingSentences(file_path))
        # ind = self.closestSentence(sentence, sentenceEmbeddings)
        # return indexToSentence[ind][1]

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

# vectors = Magnitude("GoogleNews-vectors-negative300.magnitude")
# v = vectors.query("cat") # vector representing the word 'cat'
# w = vectors.query("dog") # vector representing the word 'dog'
# sim = cosineSimilarity(v, w)
# print(sim)
# # 0.76094574

# X = WordEmbeddings("GoogleNews-vectors-negative300.magnitude")
# sentenceEmbeddings, _ = X.sentenceToEmbeddings(loadTrainingSentences("data/r2d2TrainingSentences.txt"))
# print(X.closestSentence("Lights on.", sentenceEmbeddings))

# print(X.getCategory("Turn your lights green.", "data/r2d2TrainingSentences.txt"))
#  # 'light'
# print(X.getCategory("Drive forward for two feet.", "data/r2d2TrainingSentences.txt"))
#  # 'driving'
# print(X.getCategory("Do not laugh at me.", "data/r2d2TrainingSentences.txt"))
#  # 'no'

# print(X.accuracy("data/r2d2TrainingSentences.txt", "data/r2d2DevelopmentSentences.txt"))
# 0.75

############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 4

feedback_question_2 = """
The k-nearest neighbors implementation for getCategory was challenging.
"""

feedback_question_3 = """
I enjoyed learning about word embeddings and the implementation of language models in general.
"""
