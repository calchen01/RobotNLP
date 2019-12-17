############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Edward Cohen"

############################################################
# Imports
############################################################

from pymagnitude import *
from collections import Counter
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
    "march forward",
    "march backwards",
    "march north",
    "march east",
    "march south",
    "march west",
    "halt march",
    "stop march",
    "freeze march",
    "speed march"
]

my_light_sentences = [
    "party mode",
    "disco lights",
    "black out",
    "mood lighting",
    "turn down the lights",
    "lights brighter",
    "let there be light",
    "stealth mode",
    "random light",
    "favorite light"
]

my_head_sentences = [
    "watch your six",
    "attention",
    "look forward",
    "watch out behind",
    "check your sides",
    "look north",
    "look east",
    "look south",
    "look west",
    "look around"
]

my_state_sentences = [
    "which way are you facing?",
    "what's your name?",
    "what's your condition?",
    "how are you?",
    "where are you?",
    "are your lights on?",
    "what colors are your lights?",
    "where did you come from?",
    "where are you going?",
    "what is your goal?"
]

my_connection_sentences = [
    "are there droids nearby?",
    "are there enemies nearby?",
    "are there friends nearby?",
    "end connection",
    "link up",
    "full environment scan",
    "who's near us?",
    "connect to the mainframe",
    "uplink to the server",
    "connect droid to the server"
]

my_stance_sentences = [
    "stand straight",
    "fix your posture",
    "at ease",
    "get on your tippy toes",
    "bring down third wheel",
    "go bipedal",
    "on your toes",
    "go tripedal",
    "prepare to waddle",
    "prepare to march"
]

my_animation_sentences = [
    "talk to me",
    "karoake",
    "exercise in place",
    "play music",
    "play dead",
    "call for help",
    "get backup",
    "giggle",
    "sound the alarm",
    "hammer time"
]

my_grid_sentences = [
    "your position is (0,0)",
    "go from your position to (3,2)",
    "your grid is a 3 by 3 grid",
    "there is an obstacle at (2,2)",
    "you can't go from (1,1) to (1,2)",
    "there's a trap at position (2,3)",
    "go to (3,2)",
    "each square is a square foot",
    "get to position (0,1)",
    "go around the obstacle"
]

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    tokenized = []
    sentence = sentence.strip()
    sentence = sentence.replace("\n","")
    punct = string.punctuation
    current = ""
    for char in sentence:
        char = char.lower()
        if char==" " or char in punct:
            if current!="":
                tokenized.append(current)
                current = ""
        elif char.isalnum():
            current+=char
    if current!="":
        tokenized.append(current)
    return tokenized

def cosineSimilarity(vector1, vector2):
    return np.dot(vector1,vector2) / (np.linalg.norm(vector1)*np.linalg.norm(vector2))

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)
        baseline = np.zeros(300)
        for token in tokens:
            baseline = np.add(baseline, self.vectors.query(token))
        return baseline

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
        values = list(commandTypeToSentences.values())
        sentences = sum([len(sentences) for sentences in values])
        sentenceEmbeddings = np.zeros((sentences,300))
        indexToSentence = {}
        index = 0
        for category in list(commandTypeToSentences.keys()):
            for sentence in commandTypeToSentences[category]:
                sentenceEmbeddings[index,:] = self.calcSentenceEmbeddingBaseline(sentence)
                indexToSentence[index] = (sentence,category)
                index+=1
        return sentenceEmbeddings,indexToSentence


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
        index = -1
        sentence_embedding = self.calcSentenceEmbeddingBaseline(sentence)
        max_similarity = float("-inf")
        for ind,embedding in enumerate(sentenceEmbeddings):
            similarity = cosineSimilarity(sentence_embedding,embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                index = ind
        return index



    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        training_sentences = loadTrainingSentences(file_path)
        training_embeddings, categories = self.sentenceToEmbeddings(training_sentences)
        sentence_embedding = self.calcSentenceEmbeddingBaseline(sentence)
        distance_to_categories = Counter()
        count_of_categories = Counter()
        command = ""
        for i in range(20):
            closest_sentence = self.closestSentence(sentence,training_embeddings)
            category = categories[closest_sentence][1]
            count_of_categories[category] = count_of_categories[category] + 1
            distance_to_categories[category] = distance_to_categories[category] + cosineSimilarity(self.calcSentenceEmbeddingBaseline(sentence),training_embeddings[closest_sentence,:])
            training_embeddings = np.delete(training_embeddings,closest_sentence,0)
        keys = list(count_of_categories.keys())
        averages = [0]*len(keys)
        total_count = sum(list(count_of_categories.values()))
        for index,category in enumerate(keys):
            averages[index] = distance_to_categories[category] / count_of_categories[category] * (1 + count_of_categories[category] / total_count)
        max_sim = max(averages)
        ind = averages.index(max_sim)
        command = keys[ind]
        for avg in averages:
            if max_sim!=avg and max_sim-avg < .05:
                return 'no'
        if command=="state" and "?" not in sentence:
            return 'no'
        return command

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
        dev_sentences = loadTrainingSentences(dev_file_path)
        values = list(dev_sentences.values())
        sentences = [sentence for category in values for sentence in category]
        _, dev_categories = self.sentenceToEmbeddings(dev_sentences)
        c = 0
        s = 0
        for sentence in sentences:
            category = self.getCategory(sentence,training_file_path)
            if dev_categories[s][1] == category:
                c+=1
            s+=1
        return c/s

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

        light = ["front light", "back light", "lights", "light", "forward light", "lower lights", "your lights", "holoemitter", "logic display"]
        light = np.array([self.calcSentenceEmbeddingBaseline(l) for l in light])
        c = self.closestSentence(command, light)

        if c == 0 or c == 4:
            slots["lights"] = ['front']
        elif c == 1:
            slots["lights"] = ['back']
        else:
            slots["lights"] = ['front', 'back']

        value = ["increase light", "reduce light", "change light", "maximum light", "set holoemitter", "set logic display", "set light to 0"]
        value = np.array([self.calcSentenceEmbeddingBaseline(v) for v in value])
        c = self.closestSentence(command, value)
        if c == 0:
            slots["add"] = True
        elif c == 1 or c == 6:
            slots["sub"] = True

        onoff = ["lights on", "turn light off", "light change", "light set", "set holoemitter", "set logic display", "red", "blue", "green", "lights maximum", "set light", "set value"]
        onoff = np.array([self.calcSentenceEmbeddingBaseline(o) for o in onoff])
        c = self.closestSentence(command, onoff)
        if c == 1 or c == 4:
            slots["off"] = True

        color = ["red light", "green light", "blue light", "logic display", "set holoemitter", "light", "lights increase", "lights decrease", "set light"]
        color = np.array([self.calcSentenceEmbeddingBaseline(c) for c in color])
        c = self.closestSentence(command, color)
        if c == 3:
            slots["logDisp"] = True
        if c == 4:
            slots["holoEmit"] = True

        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left".
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        tokenized_command = tokenize(command)
        for token in tokenized_command:
            if token=="increase":
                slots["increase"] = True
            elif token=="decrease":
                slots["decrease"] = True
            if "back" in token or token=="south":
                slots["directions"].append("back")
            if token=="forward" or token=="north":
                slots["directions"].append("forward")
            if "left" in token or "east" in token:
                slots["directions"].append("left")
            if "right" in token or "west" in token:
                slots["directions"].append("right")
        return slots

def euclidean_distance(vec1,vec2):
    return np.linalg.norm(vec1-vec2)

""" trainingSentences = loadTrainingSentences("data/r2d2TrainingSentences.txt")
X = WordEmbeddings("GoogleNews-vectors-negative300.magnitude")
print (X.accuracy("data/r2d2TrainingSentences.txt", "data/r2d2DevelopmentSentences.txt"))  """

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
