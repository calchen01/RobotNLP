############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Seung Hyouk Shin"

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
"Move forward.",
"Move back 1 feet.",
"Go South.",
"Stop moving.",
"Set your speed to 30%.",
"Turn left, then go forward for 1 feet.",
"Turn to heading 45 degrees.",
"Reset heading to 0.",
"Increase speed to 50%.",
"Move 2 feet to North."
]

my_light_sentences = [
"Turn off all lights.",
"Turn back light to red.",
"Turn front light to green.",
"Set RGB values on lights to 0,255,0.",
"Only turn on the front light.",
"Raise holoemitter intensity to maximum.",
"Dim the lights.",
"Change front light to blue, and back light to yellow.",
"Change all lights to green for 5 seconds.",
"Blink logic display."
]

my_head_sentences = [
"Turn around.",
"Turn head to face back.",
"Look back.",
"Rotate head to face forward.",
"Rotate head to face backward.",
"Keep rotating head.",
"Rotate head to 45 degrees.",
"Rotate head to 90 degrees.",
"Turn head to 30 degrees.",
"Turn head to 60 degrees.",
]

my_state_sentences = [
"How much better is left?",
"Are you driving?",
"What is your stance?",
"What is your speed right now?",
"What angle are you driving in?",
"What is the color of your back light?",
"Are you awake?",
"Are you connected?",
"What is orientation?",
"Where are you heading?"
]

my_connection_sentences = [
"Disconnect.",
"Disconnect from server.",
"Connect to server D2-55A2.",
"Connect D2-55A2 to server.",
"Scan for other droids.",
"Are there other droids nearby?",
"Connect to R2D2.",
"Connect to R2Q5.",
"Exit.",
"What other droids are connected to server?"
]

my_stance_sentences = [
"Set stance to be biped.",
"Waddle.",
"Put down the first wheel.",
"Tiptoe.",
"Start waddling.",
"Waddle for 5 seconds.",
"Set stance to waddle.",
"Stop waddling.",
"Reset your stance.",
"Stop tiptoeing."
]

my_animation_sentences = [
"Make noise for 3 seconds.",
"Fall over.",
"Stand up.",
"Play a song.",
"Ring an alarm.",
"Get up.",
"Fall down.",
"Scream.",
"Play sound for a minute.",
"Spin for 10 seconds."
]

my_grid_sentences = [
"You are on a 5 by 5 grid.",
"Go to position (0,0).",
"There is an obstacle at position 1,1.",
"You cannot go from 2,2 to 2,1.",
"Each square is 2 feet large.",
"Move to position (3,3).",
"Move to grid (1,2).",
"This is a 10 by 10 grid.",
"You are on (6,6) grid.",
"Obstacle at (4,1)."
]

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    for i in sentence:
        if i in string.punctuation:
            sentence = sentence.replace(i, ' ')
    out = sentence.split()
    return [word.lower() for word in out]

def cosineSimilarity(vector1, vector2):
    dot = np.dot(vector1, vector2)
    v1length = np.sqrt(vector1.dot(vector1))
    v2length = np.sqrt(vector2.dot(vector2))

    return dot/(v1length*v2length)

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        token = tokenize(sentence)
        total = np.zeros(self.vectors.dim)
        if len(token) == 0:
        	return total
        for w in token:
        	total = np.add(total, self.vectors.query(w))

        return total

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
        dict = {}
        list = []
        i = 0
        for category in commandTypeToSentences.keys():
        	for sntn in commandTypeToSentences[category]:
        		dict[i] = (sntn, category)
        		list.append(self.calcSentenceEmbeddingBaseline(sntn))
        		i += 1

        return (np.array(list), dict)


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
        out = (0, 0)

        for i in range(len(sentenceEmbeddings)):
        	cos = cosineSimilarity(v, sentenceEmbeddings[i])
        	if cos > out[0]:
        		out = (cos, i)

        return out[1]

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

def main():
	print('Tests')

	# print(tokenize("  This is an example.  "))
	# print(tokenize("'Medium-rare,' she said."))

	# print(cosineSimilarity(np.array([2, 0]), np.array([0, 1])))
	# print(cosineSimilarity(np.array([1, 1]), np.array([1, 1])))
	# print(cosineSimilarity(np.array([10, 1]), np.array([1, 10])))
	# v1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	# v2 = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
	# print(cosineSimilarity(v1, v2))

	# path = '/Users/seunghyoukshin/Downloads/'
	# vectors = Magnitude(path + "GoogleNews-vectors-negative300.magnitude")
	# v = vectors.query("cat")
	# w = vectors.query("dog")
	# print(cosineSimilarity(v,w))

	# X = WordEmbeddings("/Users/seunghyoukshin/Downloads/GoogleNews-vectors-negative300.magnitude")
	# svec1 = X.calcSentenceEmbeddingBaseline("drive forward")
	# svec2 = X.calcSentenceEmbeddingBaseline("roll ahead")
	# svec3 = X.calcSentenceEmbeddingBaseline("set your lights to purple")
	# svec4 = X.calcSentenceEmbeddingBaseline("turn your lights to be blue")
	# print(cosineSimilarity(svec1, svec2))
	# print(cosineSimilarity(svec1, svec3))
	# print(cosineSimilarity(svec1, svec4))
	# print(cosineSimilarity(svec2, svec3))
	# print(cosineSimilarity(svec2, svec4))
	# print(cosineSimilarity(svec3, svec4))

	# trainingSentences = loadTrainingSentences("data/r2d2TrainingSentences.txt")
	# X = WordEmbeddings("/Users/seunghyoukshin/Downloads/GoogleNews-vectors-negative300.magnitude")
	# sentenceEmbeddings, indexToSentence = X.sentenceToEmbeddings(trainingSentences)
	# print(sentenceEmbeddings[14:])
	# print(indexToSentence[14])

	# sentenceEmbeddings, _ = X.sentenceToEmbeddings(loadTrainingSentences("data/r2d2TrainingSentences.txt"))
	# print(X.closestSentence("Lights on.", sentenceEmbeddings))

	# print(X.getCategory("Turn your lights green.", "data/r2d2TrainingSentences.txt"))
	# print(X.getCategory("Drive forward for two feet.", "data/r2d2TrainingSentences.txt"))
	# print(X.getCategory("Do not laugh at me.", "data/r2d2TrainingSentences.txt"))

	# print(X.accuracy("data/r2d2TrainingSentences.txt", "data/r2d2DevelopmentSentences.txt"))


if __name__ == '__main__':
    main()


############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 7

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
