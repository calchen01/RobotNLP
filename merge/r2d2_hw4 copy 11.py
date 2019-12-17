############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Xinyu Ma"
# leaderboard: pdqxwNGuKo

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
"Go forward for 2 feet, then turn left.",
"Go West.",
"Go East.",
"Go North-by-northwest.",
"Decrease your speed by 50%.",
"Turn to your right.",
"Turn to your left.",
"Turn away.",
"Increase your speed by 100%.",
"Stop"
]

my_light_sentences = [
"Increase the intensity on the holoemitter by 50%.",
"Decrease the the intensity on the holoemitter by 100%.",
"Change the back LED to red.",
"Turn your back light red.",
"Turn off the lights.",
"Set the RGB values on your lights to be 255,255,255.",
"Set the RGB values on your lights to be 0,0,0.",
"Change the color on both LEDs to be red.",
"Turn up the lights",
"Lights on"
]

my_head_sentences = [
"Turn your head to face backward.",
"Turn your head at 90 degree angle.",
"Turn your head to face forward three times.",
"Turn your head to face first forward and then backward.",
"Turn your head at 45 degree angle.",
"Turn your head at 60 degree angle.",
"Turn your head at 75 degree angle.",
"Turn your head at 180 degree angle.",
"Turn your head clockwise.",
"Turn your head anti-clockwise."
]

my_state_sentences = [
"Are you not awake?",
"How much battery is left?",
"What is your speed now?",
"What is on your logic display?",
"What is the color of your front light?",
"What direction are you heading to?",
"What is the intensity on your logic display?",
"What is your stance?",
"Are you waddling?",
"What is your holoemitter projection intensity?"
]

my_connection_sentences = [
"Connect D2-55A2 to a server.",
"Connect yourself to a server.",
"Do you find any other droids nearby?",
"Disconnect yourself from the server.",
"Disconnect.",
"Scan for server.",
"Scan.",
"Exit.",
"Connect to a R2D2 droid.",
"Connect to a R2Q5 droid."
]

my_stance_sentences = [
"Set your stance to be biped.",
"Put down your third wheel.",
"Stand on your tiptoes.",
"Waddle.",
"Totter.",
"Todder.",
"Teater.",
"Wobble.",
"Rock from side to side on your toes.",
"Imitate a duck's walk."]

my_animation_sentences = [
"Fall over.",
"Scream.",
"Make some noise.",
"Laugh.",
"Play an alarm.",
"Cry like a monkey.",
"Laugh as loud as you can.",
"Shout.",
"Roar.",
"Howl."]

my_grid_sentences = [
"You are on a 5 by 4 grid.",
"Each square is 2 foot large.",
"You are at position (1,1).",
"Go to position (2,2).",
"There is an obstacle at position 2,3.",
"There is a chair at position 3,2",
"Go to the left of the chair.",
"Itâ€™s not possible to go from 2,2 to 2,3.",
"Go to the right of the chair.",
"Itâ€™s possible to go from 2,2 to 2,3."
]

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    PUNCTUATIONS = string.punctuation
    for mark in PUNCTUATIONS:
        sentence = sentence.replace(mark, " ").lower()
    return sentence.split()

def tokenize2(sentence):
    sentence = tokenize(sentence)
    sentence = [word for word in sentence if word not in my_stop_words]
    return ' '.join(sentence) + '.'

def cosineSimilarity(vector1, vector2):
    return np.inner(vector1, vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2))

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        v = tokenize(sentence)
        if not v: return np.zeros(300)
        v = [self.vectors.query(componet) for componet in v]
        return np.array([sum(i) for i in zip(*v)])

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
        ind = 0
        svecs = []
        indexes = {}
        for key, value in commandTypeToSentences.items():
            for sentence in value:
                svec = self.calcSentenceEmbeddingBaseline(sentence)
                svecs.append(svec)
                indexes[ind] = (sentence, key)
                ind += 1

        if len(svecs) >=  1:
            result = np.stack(tuple(svecs))
            return(result, indexes)

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
        svec = self.calcSentenceEmbeddingBaseline(sentence)
        def myfunction(x):
            return cosineSimilarity(x, svec)
        res = np.apply_along_axis(myfunction, axis=1, arr=sentenceEmbeddings)
        return int(np.where(res == np.amax(res))[0])

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
        sentenceEmbeddings, indexToSentence  = self.sentenceToEmbeddings(commandTypeToSentences)
        ind = self.closestSentence(sentence, sentenceEmbeddings)
        svec = self.calcSentenceEmbeddingBaseline(sentence)
        score = cosineSimilarity(sentenceEmbeddings[ind], svec)
        if score < 0.7 or (indexToSentence[ind][1] == 'state' and score < 0.8):
            return "no"
        else:
            return indexToSentence[ind][1]

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
        commandTypeToSentences_devp = loadTrainingSentences(dev_file_path)
        correct_count = 0
        total_count = 0
        for key, value in commandTypeToSentences_devp.items():
            for sentence in value:
                total_count += 1
                if (self.getCategory(sentence, training_file_path) == key):
                    correct_count += 1
        return correct_count/total_count
    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        print(command)
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

        ### YOUR CODE HERE ###

        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        print(command)
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