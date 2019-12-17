############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Saniyah Shaikh"

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

my_driving_sentences = ["Start driving.", 
                        "Go forward for 10 feet then stop.", 
                        "Stop rolling.",
                        "Increase speed by 20%.",
                        "Turn to your left.",
                        "Set your speed to 0.",
                        "Turn 180 degrees.", 
                        "Move to the North.",
                        "Turn to a heading of 75 degrees.",
                        "Reset your speed to 0.",
                        "Reset the direction of travel."]

my_light_sentences = ["Set holoemitter intensity to max.", 
                      "Turn the back light red.", 
                      "Dim the holoemitter.", 
                      "Set the blue value of the front light to 255.",
                      "Turn off the lights.", 
                      "Set the RGB values of the lights to 0, 0, 255", 
                      "Set both lights to green.", 
                      "Show these colors for 1 second each: red, purple, blue.",
                      "Turn on the logic display.", 
                      "Increase all RGB values on the front lights by 50.",
                      "Flicker the front LED lights for two seconds.", 
                      "Disengage the logic display.", 
                      "Display rainbow lights on the front LED for 10 seconds."]

my_head_sentences = ["Look to the left.",
                     "Move your head to the right.", 
                     "Watch out behind you!", 
                     "Turn your head forward.", 
                     "Look out behind you.",
                     "Turn your head to the left.",
                     "Can you see anything behind you?",
                     "Look to the right.",
                     "Turn your head by 45 degrees.",
                     "Turn your head to the right by 20 degrees."] 

my_state_sentences = ["What is your stance?", 
                      "What is the color of the back light?", 
                      "Is the holo projector on?", 
                      "How much battery is left?", 
                      "Is the back LED on?", 
                      "What is the currect heading?",
                      "What is the status of the battery?", 
                      "Are you moving now?",
                      "How fast are you moving now?",
                      "Tell me what the speed setting is.", 
                      "Is waddle mode active?"]
 
my_connection_sentences = ["Connect this robot.",
                           "Connect this robot to the server.",
                           "Disconnect this robot.",
                           "Scan for other robots.", 
                           "Are there any other robots on the server?",
                           "How many robots can be detected by this server?",
                           "Can you connect to the server?",
                           "Can you disconnect?", 
                           "If possible, connect to the server.", 
                           "Connect D2-43A2 to the server."]
 
my_stance_sentences = ["Start waddling."
                       "Set the stance to waddle.", 
                       "Walk like a ducky.",
                       "Waddle.", 
                       "Stop waddling.", 
                       "Stand straight.", 
                       "Act like a duck."
                       "Stop walking like a duck."
                       "Set the stance to biped.", 
                       "Pull up the third wheel.", 
                       "Set a random stance.", 
                       "Change the stance to biped.", 
                       "Place your third wheel on the floor."]

my_animation_sentences = ["Chirp.", 
                          "Talk to me.", 
                          "Make a noise.",
                          "Speak.",
                          "Play the alarm.", 
                          "Start screaming.", 
                          "Fall down.", 
                          "Squeak.", 
                          "What does the droid say?", 
                          "Laugh with me."]

my_grid_sentences = ["You are on a 3 by 5 grid.",
                     "A square on the grid is 1 foot long.",
                     "You start at (1, 1) on the grid.",
                     "Move to (2, 4).", 
                     "There is an obstacle at 2, 1.", 
                     "There is a table at 3, 3.",
                     "Go to the right of the table.",
                     "It is not possible to reach 2,3 from 2,2.",
                     "This is a 3 x 5 grid.", 
                     "It is not possible to go from 1,2 to 2,2."]

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    tokens = []
    parts = sentence.lower().split()
    for part in parts:
        last_index = 0
        i = 0
        while i < (len(part)):
            if part[i] in string.punctuation:
                first_bit = part[last_index:i]
                if first_bit != "":
                    tokens.append(first_bit)
                last_index = i + 1
                i = i + 1
            else:
                i = i + 1
        last_bit = part[last_index:]
        if last_bit != "":
            tokens.append(last_bit)
    return tokens

def cosineSimilarity(vector1, vector2):
    dot_prod = np.dot(vector1, vector2)
    len_1 = np.linalg.norm(vector1)
    len_2 = np.linalg.norm(vector2)
    return dot_prod / (len_1 * len_2)

class WordEmbeddings:

    def __init__(self, file_path):
        self.fp = file_path
        self.vectors = Magnitude(file_path)
        self.shape = self.vectors.query("dog").shape
        

    def calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)
        if tokens:
            result = self.vectors.query(tokens[0])
        else:
            return np.zeros(self.shape)
        for i in range(1, len(tokens)):
            result = np.add(result, self.vectors.query(tokens[i]))
        return result
            

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
        index = 0
        array = []
        indexToSentence = {}
        for category in commandTypeToSentences.keys():
            for sentence in commandTypeToSentences[category]:
                array.append(self.calcSentenceEmbeddingBaseline(sentence))
                indexToSentence[index] = (sentence, category)
                index += 1
        return np.array(array), indexToSentence
                

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
        vector = self.calcSentenceEmbeddingBaseline(sentence)
        i = 0
        best, score = -1, -1
        for row in sentenceEmbeddings:
            sim = cosineSimilarity(row, vector)
            if sim > score:
                best, score = i, sim
            i += 1
        return best

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
