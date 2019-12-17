############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Peng Li"

############################################################
# Imports
############################################################

from pymagnitude import *
import numpy as np
import string
import re
from sklearn.neighbors import KNeighborsClassifier
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
"Go straight and don't stop.",
"Moving for 10 seconds at speed of 1.",
"Drive forward.",
"Move at speed of 1.0.",
"Move to your right.",
"Face right.",
"Move backwards.",
"Stop moving.",
"Brake.",
"Lower your speed.",
"Move faster.",
"Move 2 meters to your left in 10 seconds."]
## "Circling clockwise.",
## "Circling counterclockwise.",
my_light_sentences = [
"Set back LED color rgb value to 255, 255, 255.",
"Set front LED color to red.",
"Turn on front light.",
"Blink.",
"Add 10 to red value to your front led.",
"Turn all lights off.",
"Set holoemitter projector intensity to maximum.",
"Set colors of both LEDs to be purple.",
"Dim holoemitter projector.",
"Display colors in the following sequence: red, yellow, green.",
"Set display intensity to 50 percent."]

my_head_sentences = [
"Head to 30 degrees.",
"Turn left.",
"Face north.",
"Turn right.",
"Face to 3 o'clock direction.",
"Look around.",
"To your right.",
"Turn around.",
"Watch your back.",
"Look ahead."]

my_state_sentences =[
"Tell me your speed.",
"What is your current direction?",
"Is your light on now?",
"Show me your battery level.",
"What is your projector intensity?",
"Are you wadding?",
"Are you in logic display mode?",
"What is your front LED color?",
"Are both LEDs set to the same color?",
"What is your current drive mode?"]

my_connection_sentences = [
"Connect Q5-D009 to server.",
"Find nearby robots.",
"Scan robots.",
"Disconnect robot.",
"Any robots available?",
"Are there any robots nearby?",
"Find nearby droids",
"Disconnect droid.",
"Scan droids.",
"Connect droid Q5-D009."]

my_stance_sentences = [
"Walk like a man.",
"Walk like an old man.",
"Use three legs.",
"Waddle.",
"Use two legs.",
"Dance.",
"Start waddle.",
"Stop waddle.",
"Step out your extra leg.",
"Don't wobble."]

my_animation_sentences = [
"Move head.",
"Music.",
"Smile.",
"LOL.",
"Dance.",
"Pick a song to play.",
"Show me you are alive.",
"Animate 5 seconds.",
"Laugh out loud.",
"Say hi."]

my_grid_sentences = [
"No way to (3,4) from current position.",
"Go to (3,4)",
"Someone took (2,2).",
"Stay still.",
"The goal is at (3,4).",
"Move forward to (3,4).",
"The grid is 5 by 5.",
"Your are in (3,4).",
"Your current position is (2,2).",
"Head to the flag."]

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    result = []
    for token in sentence.split():
        punFlag = False
        piece = ''
        for char in token:
            if char in string.punctuation:
                if piece != '':
                    result += [piece.lower()]
                    piece = ''
            else:
                piece += char
        if piece != '':
            result += [piece.lower()]
    return result

def cosineSimilarity(vector1, vector2):
    a = vector1.dot(vector2)
    b = np.sqrt(vector1.dot(vector1))*np.sqrt(vector2.dot(vector2))
    return a/b

class WordEmbeddings:
    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)
        self.len = 0
        for key, vector in self.vectors:
            self.len = len(vector)
            break
    def calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)
        result = np.zeros(self.len)
        for token in tokens:
            if token in self.vectors:
                result += self.vectors.query(token)
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

            indexToSentence: A dictionary with key: index i, value: (category, sentence).(should be (sentence, category))
        '''
        indexToSentence = {}
        m = 0
        for key in commandTypeToSentences:
            m += len(commandTypeToSentences[key])
        sentenceEmbeddings = np.zeros((m, self.len))
        i = 0
        for key in commandTypeToSentences:
            for sentence in commandTypeToSentences[key]:
                # print(type(sentence))
                sentenceEmbeddings[i,:] = self.calcSentenceEmbeddingBaseline(sentence)
                indexToSentence[i] = (sentence, key)
                i+=1
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
        close = -1
        i = 0
        m, n = sentenceEmbeddings.shape
        v1 = self.calcSentenceEmbeddingBaseline(sentence)
        for a in range(m):
            v2 = sentenceEmbeddings[a,:]
            temp = cosineSimilarity(v1, v2)
            if temp > close:
                close = temp
                i = a
        return i
    def kclosestSentence(self, sentence, sentenceEmbeddings, indexToSentence, k):
        '''Returns the index of the closest sentence to the input, 'sentence'.
        Inputs:
            sentence: A sentence

            sentenceEmbeddings: An mxn numpy array, where m is the total number
            of sentences and n is the dimension of the vectors.
        Returns:
            a list of integer i, where i is the row index in sentenceEmbeddings 
            that contains the closest sentence to the input
        '''
        close = -1
        i = []
        m, n = sentenceEmbeddings.shape
        v1 = self.calcSentenceEmbeddingBaseline(sentence)
        for a in range(m):
            v2 = sentenceEmbeddings[a,:]
            temp = cosineSimilarity(v1, v2)
            if len(i) < k:
                if temp > close:
                    i=[(indexToSentence[a][1], temp)]+i
                    close = temp
                else:
                    i+=[(indexToSentence[a][1],temp)]
            else:
                if temp > i[k-1][1]:
                    i[k-1] = (indexToSentence[a][1],temp)
        i.sort(key=lambda tup:tup[1], reverse = True)
        # print(i)
        return i
    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        driving = ["forward", "roll", "move", "rolling", "run"]
        light = ["blink", "logic", "dim"]
        state = ["what", "how", "status", "awake"]#most confusing
        # connection = ["disconnect", "connect"]
        grid = ["grid", "position", "obstacle"]
        tokens = tokenize(sentence)
        animation = ["sing", "fall", "speak", "noise"]
        if (len(tokens) == 0):
            return "no"
        for token in tokens:
            if token in state:
                return "state"
            elif token in driving:
                return "driving"
            elif token in light:
                return "light"
            elif token in animation:
                return "animation"
            elif token in grid:
                return "grid"
        stopwords = ["i", "we", "you", "he", "she", "it", "at", \
            "a", "an", "bit","of", "too", "the", "there", "you", \
            "to", "is", "are", "am", "that", "this", "me", "from", "by",\
            "many", "lot", "much", "very", "just"]
        newsentence = ""
        for token in tokens:
            if token not in stopwords:
                newsentence = newsentence + token + " "
        (sentenceEmbeddings, indexToSentence) = self.sentenceToEmbeddings(loadTrainingSentences(file_path))
        k = 10
        klist = self.kclosestSentence(newsentence, sentenceEmbeddings, indexToSentence, k)
        klist = [i for i in klist if i[1] > 0.4]
        if len(klist) == 0:
            return 'no'
        if klist[0][1] > 0.8 or (len(klist) == 1 and klist[0][1] > 0.6) or (len(klist) > 1 and klist[0][1] - klist[1][1] > 0.1):
            return klist[0][0]
        else:
            catagory = [i[0] for i in klist if i[1]>0.6]
            if len(catagory) == 0:
                return "no"
            maxcatagory = max(set(catagory), key = catagory.count)
            return maxcatagory
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
        test = loadTrainingSentences(dev_file_path)
        c = 0
        s = 0
        for catagory in test:
            for sentence in test[catagory]:
                predict = self.getCategory(sentence, training_file_path)
                s+=1
                if predict == catagory:
                    c+=1
        return c/s

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}
        ### YOUR CODE HERE ###
        holoEmit = ["holoemitter", "projector", "holo"]
        logDisp = ["logic", "display", "blink"]
        front = ["front", "forward"]
        back = ["back"]
        both = ["both", "lights", "all", "leds"]
        on = ["on", "maximum"]
        off = ["off", "minimum", "out", "0"]
        add = ["increase", "add"]
        sub = ["subtract", "minus", "weaken","reduce"]
        token = tokenize(command)
        # lights = set()
        for i in token:
            if i in holoEmit:
                slots["holoEmit"] = True
            elif i in logDisp:
                slots["logic"] = True
            elif i in front:
                slots["lights"].append("front")
            elif i in back:
                slots["lights"].append("back")
            elif i in both: 
                slots["lights"].append("front")
                slots["lights"].append("back")
            elif i in on:
                slots["on"] = True
            elif i in off:
                slots["off"] = True
            elif i in add:
                slots["add"] = True
            elif i in sub:
                slots["sub"] = True
        # slots["lights"] = list(lights)
        return slots
    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}
        ### YOUR CODE HERE ###
        increase = ["increase", "faster", "up", "add"]
        decrease = ["decrease", "slow", "down", "lower"]
        forward = ["front", "forward", "straight", "north"]
        backward = ["south", "backward"]
        right = ["right", "east"]
        left = ["left", "west"]
        token = tokenize(command)
        for i in token:
            if i in increase:
                slots["increase"] = True
                # slots["increase"] = False
            elif i in decrease:
                slots["decrease"] = True
                # slots["increase"] = False
            elif i in forward:
                slots["directions"].append("forward")
            elif i in backward:
                slots["directions"].append("backward")
            elif i in right: 
                slots["directions"].append("right")
            elif i in left:
                slots["directions"].append("left")
        return slots

############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 10

feedback_question_2 = """
.
"""

feedback_question_3 = """
.
"""
