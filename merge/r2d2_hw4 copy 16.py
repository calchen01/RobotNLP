############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Yue Lian"

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
"Go left.",
"Turn left, then go forward for 3 feet.",
"Decrease your speed by 10%.",
"Don't move.",
"Turn around, then stop.",
"Increase your speed by 60%.",
"Turn to heading 50 degrees.",
"Roll.",
"Go South.",
"East is at heading 30 degrees.",
]

my_light_sentences = [
"Change your back light to blue.",
"Turn off all your lights.",
"Decrease the red value of your front light by 10%.",
"Set the RGB values on both LEDs to be 100,100,0.",
"Change the intensity on the holoemitter to minimum.",
"Blink your logic display for 2 seconds.",
"Add 50 to the green value of your back LED.",
"Display the colors for 1 seconds each: green, red, yellow, orange, blue.",
"Turn your red light blue.",
"Dim your lights holoemitter for 10 seconds.",
]

my_head_sentences = [
"Head left.",
"Face right.",
"Look ahead.",
"Turn your head to face back.",
"Turn 30 degrees to your left.",
"Set angle to be 180 degrees.",
"Turn 70 degrees right.",
"Rotate 40 degrees left.",
"Turn 10 degrees left, then turn 90 degrees right.",
"Turn around.",
]

my_state_sentences = [
"What is your current angle?",
"Are you awake?",
"What color is your back LED?",
"What is your current speed?",
"Is your front LED blue?",
"Are you waddling?",
"What direction are you facing?",
"Is your stance 2?",
"Is your holoemitter on?",
"Tell me your drive mode.",
]

my_connection_sentences = [
"Disconnect.",
"Connect D2-55A3 to the server.",
"Scan.",
"Are there any droids around?",
"Disconnect from the server.",
"Is there a droid connected to the server?",
"Exit.",
"Shut down.",
"Close all connections.",
"Do not connect.",
]

my_stance_sentences = [
"Put down your second wheel.",
"Stand on your tiptoes.",
"Use two legs for walking.",
"Waddle.",
"Start waddling.",
"Put down all your wheels.",
"Stop waddling.",
"Stand still.",
"Walk like a duck.",
"Stop walking like a duck.",
]

my_animation_sentences = [
"Start laughing.",
"Do not laugh.",
"Cry.",
"Scream out loud.",
"Stop laughing.",
"Play random sound.",
"Laugh for 3 seconds.",
"Play some music.",
"Stop screaming.",
"Play sound number 3.",
]

my_grid_sentences = [
"You are on a 6 by 6 grid.",
"Go to position 3,5.",
"There is a lamp at position (2,1).",
"You are at position (5,5).",
"Go to the right of the lamp.",
"Each square is 2 feet large.",
"You cannot go to position 2,1.",
"There is something at 6,2.",
"Go around the lamp.",
"Go to 1,1.",
]


############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

stop_words = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 
'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 
'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 
'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 
'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 
'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 
'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 
'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 
'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 
'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 
'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 
'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 
'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 
'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

def tokenize(sentence):
    translator = re.compile('[%s]' % re.escape(string.punctuation))
    a = translator.sub(' ', sentence)
    b = re.sub('\w+', lambda m: m.group(0).lower(), a)
    return b.strip().split()

def cosineSimilarity(vector1, vector2):
    norm_1 = np.linalg.norm(vector1)
    norm_2 = np.linalg.norm(vector2)
    if (norm_2) <= 0.01:
        return np.dot(vector1, vector2)/norm_2
    else:
        return np.dot(vector1, vector2)/(norm_1*norm_2)

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        tokenized = tokenize(sentence)
        if len(tokenized) == 0:
            return np.zeros(300)
        res = self.vectors.query(tokenized[0])
        for i in range(1, len(tokenized)):
            res = np.add(res, self.vectors.query(tokenized[i]))
        return res

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

            indexToSentence: A dictionary with key: index i, value: (sentence, category).
        '''
        i = 0
        sentenceEmbeddings = None
        indexToSentence = dict()
        for key, value in commandTypeToSentences.items():
            for sentence in value:
                if sentenceEmbeddings is None:
                    sentenceEmbeddings = self.calcSentenceEmbeddingBaseline(sentence)
                else:
                    sentenceEmbeddings = np.vstack((sentenceEmbeddings, self.calcSentenceEmbeddingBaseline(sentence)))
                indexToSentence[i] = (sentence, key)
                i += 1
        if sentenceEmbeddings is None:
            sentenceEmbeddings = np.zeros((i, 300))
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
        curEmbedding = self.calcSentenceEmbeddingBaseline(sentence)
        maxCos = 0
        closest = 0
        if sentenceEmbeddings is None:
            return closest
        for i in range(sentenceEmbeddings.shape[0]):
            cos = cosineSimilarity(curEmbedding, sentenceEmbeddings[i])
            if cos > maxCos:
                maxCos = cos
                closest = i
        if maxCos < 0.4:
            return 1
        return closest

    def calcSentenceEmbeddingBetter(self, sentence):
        tokenized_pre = tokenize(sentence)
        tokenized = [w for w in tokenized_pre if w not in stop_words]
        if len(tokenized) == 0:
            return np.zeros(300)
        res = self.vectors.query(tokenized[0])
        for i in range(1, len(tokenized)):
            res = np.add(res, self.vectors.query(tokenized[i]))
        return res

    def sentenceToEmbeddingsBetter(self, commandTypeToSentences):
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

            indexToSentence: A dictionary with key: index i, value: (sentence, category).
        '''
        i = 0
        sentenceEmbeddings = None
        indexToSentence = dict()
        for key, value in commandTypeToSentences.items():
            for sentence in value:
                if sentenceEmbeddings is None:
                    sentenceEmbeddings = self.calcSentenceEmbeddingBetter(sentence)
                else:
                    sentenceEmbeddings = np.vstack((sentenceEmbeddings, self.calcSentenceEmbeddingBetter(sentence)))
                indexToSentence[i] = (sentence, key)
                i += 1
        return (sentenceEmbeddings, indexToSentence)

    def closestSentenceBetter(self, sentence, sentenceEmbeddings):
        '''Returns the index of the closest sentence to the input, 'sentence'.

        Inputs:
            sentence: A sentence

            sentenceEmbeddings: An mxn numpy array, where m is the total number
            of sentences and n is the dimension of the vectors.

        Returns:
            an integer i, where i is the row index in sentenceEmbeddings 
            that contains the closest sentence to the input
        '''
        curEmbedding = self.calcSentenceEmbeddingBetter(sentence)
        maxCos = 0
        closest = 0
        if sentenceEmbeddings is None:
            return closest
        for i in range(sentenceEmbeddings.shape[0]):
            cos = cosineSimilarity(curEmbedding, sentenceEmbeddings[i])
            if cos > maxCos:
                maxCos = cos
                closest = i
        if maxCos < 0.4:
            return 1
        return closest

    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        sentences = loadTrainingSentences(file_path)
        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddingsBetter(sentences)
        closest = self.closestSentenceBetter(sentence, sentenceEmbeddings)
        if closest == 1:
            return 'no'
        return indexToSentence[closest][1]

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
        trainingSentences = loadTrainingSentences(training_file_path)
        testingSentences = loadTrainingSentences(dev_file_path)

        sentenceEmbeddings, indexToTrainSentence = self.sentenceToEmbeddingsBetter(trainingSentences)
        c = 0
        s = 0
        for key, value in testingSentences.items():
            for sentence in value:
                predict_idx = self.closestSentenceBetter(sentence, sentenceEmbeddings)
                if predict_idx == 1:
                    predict_cat = 'no'
                else:
                    predict_cat = indexToTrainSentence[predict_idx][1]
                s += 1
                if predict_cat == key:
                    c += 1
        return c/s

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

        tokenized = tokenize(command)

        if "holoemitter" in tokenized or "dim" in tokenized:
            slots["holoEmit"] = True
        if "logic" in tokenized or "blink" in tokenized:
            slots["logDisp"] = True
        if "front" in tokenized:
            slots["lights"] += ["front"]
        if "back" in tokenized:
            slots["lights"] += ["back"]
        if "lights" in tokenized or "both" in tokenized:
            slots["lights"] = ["front", "back"]
        if "increase" in tokenized or "add" in tokenized:
            slots["add"] = True
        if ("decrease" in tokenized or "reduce" in tokenized or "subtract" in tokenized 
            or "lower" in tokenized or "weaken" in tokenized):
            slots["sub"] = True
        if "on" in tokenized and "turn" in tokenized:
            slots["on"] = True
        if "off" in tokenized and "turn" in tokenized:
            slots["off"] = True

        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###
        tokenized = tokenize(command)
        
        for w in tokenized:
            if w in ["north", "front", "forward"]:
                slots["directions"] += ["forward"]
            elif w in ["south", "back", "around"]:
                slots["directions"] += ["back"]
            elif w in ["west", "left"]:
                slots["directions"] += ["left"]
            elif w in ["east", "right"]:
                slots["directions"] += ["right"]

        if "increase" in tokenized or "add" in tokenized:
            slots["increase"] = True
        if ("decrease" in tokenized or "reduce" in tokenized or "subtract" in tokenized 
            or "stop" in tokenized or "halt" in tokenized):
            slots["decrease"] = True

        return slots


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
