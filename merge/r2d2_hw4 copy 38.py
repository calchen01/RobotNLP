#! /usr/bin/env python
############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Qiongying Jiang, Jiaying Guo"

############################################################
# Imports
############################################################

from pymagnitude import *
import math
import numpy as np
from numpy import dot
from numpy import array
from numpy.linalg import norm
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from math import sqrt
from random import seed
from random import randrange


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

my_driving_sentences = ['Turn to heading 60 degrees.',
                        'Turn to your left', 'Stop now', 'Turn to face east',
                        'Go straight ahead for 3 feet, and then turn around',
                        'Set speed to be 50%', ' Turn right', 'Face East',
                        'Face West', 'Increase speed by 20%', ' Go to north side']
my_light_sentences = ['Turn off lights.', 'No lights.',
                      'Turn front light green.', 'Turn off front light.', ' Turn back light green.',
                      ' Turn off back light.',
                      'Turn all lights to green.', 'Turn all lights to blue.', 'Turn lights to red.',
                      'Display lights in order: red, green, blue.']
my_head_sentences = ['Turn head to face backward.',
                     'Turn head to the front.', 'Turn head to east.', ' Turn head to west.',
                     'Turn head to north.', 'Turn head to south.', 'Turn around head.',
                     'Shake your head.', 'Rotate your head 90 degrees.', 'Look behind.']
my_state_sentences = ['What direction facing now?',
                      'What is the front light?', 'What is the back light?', 'Is the front light green?',
                      'What is battery status?', 'How much battery left?',
                      'What is current speed?', 'How fast will you get off?',
                      'Are you awake or off?', 'What is the current direction?',
                      'What is current orientation?']
my_connection_sentences = ['Connect to server.', 'Connect now.', 'Reconnect.', 'Are you connected to server?',
                           'Disconnect.', 'Disconnect with server.', 'Are you disconnected with server?',
                           'Look for other nearby droids.', 'Are there any nearby droids?', 'Any droids nearby?']
my_stance_sentences = ['Stand on your toes.', 'Stand on two feet.', 'Stand on two feet and third wheel.',
                       'Use third wheel.',
                       'Take back third wheel.', 'Put down third wheel.', 'Lean forward', 'Change stand position',
                       'Do not use third wheel.', 'Usese two feet only.']
my_animation_sentences = ['fall down.', 'fall over.', 'laugh now', 'scream', 'make sound', 'make some noise',
                          'alarm', 'say something', 'drop the beat', 'play an alarm']
my_grid_sentences = ['You are on 3 by 4 grid.', 'Each square is half foot size.', 'You are currently at (0,0).',
                     'Move to (1,2).', 'There is an obstacle at (1,1).', 'There is an obstacle at (2,2).',
                     'It cannot go from (1,2) to (1,1).',
                     'There is a toy at (1,0).', 'Do not go through (0,0).', 'Get the toy.']

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"


def tokenize(sentence):
    for punction in string.punctuation:
        sentence = sentence.replace(punction, ' ')
    return sentence.strip().lower().split()


def cosineSimilarity(vector1, vector2):
    return dot(vector1, vector2) / (norm(vector1) * norm(vector2))


class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)
        self._n = self.vectors.dim

    def calcSentenceEmbeddingBaseline(self, sentence):
        result = np.zeros(self._n)
        words = tokenize(sentence)
        for word in words:
            result += self.vectors.query(word)
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
        m = sum([len(commandTypeToSentences[k]) for k in commandTypeToSentences])
        sentenceEmbeddings = np.zeros((m, self._n))
        indexToSentence = {}
        idx = 0
        for key in commandTypeToSentences:
            for sentence in commandTypeToSentences[key]:
                sentenceEmbeddings[idx,] = self.calcSentenceEmbeddingBaseline(sentence)
                indexToSentence[idx] = (sentence, key)
                idx += 1
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

        max = -math.inf
        index = -1
        i = 0
        current = self.calcSentenceEmbeddingBaseline(sentence)
        for s in sentenceEmbeddings:
            result = cosineSimilarity(current, s)
            if result > max:
                index = i
                max = result
            i += 1
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
        #
        driving = ["run", "move", "roll", "rolling", "forward", "speed","south","north","west","east"]
        light = ["blink", "logic", "dim","LED", "display", "blue", "red", "green", "orange", "yellow", "purple"]
        state = ["what", "how", "status", "awake"]
        grid = ["grid", "position", "obstacle"]
        animation = ["sing", "fall", "speak", "noise","laugh","play","scream"]

        tokens = tokenize(sentence)
        if len(tokens) == 0:
            return "no"
        for word in tokens:
            if word in state and "?" in sentence:
                return "state"
            if word in driving:
                return "driving"
            if word in light:
                return "light"
            if word in grid:
                return "grid"
            if word in animation:
                return "animation"

        devSentenceEmbedding = self.calcSentenceEmbeddingBaseline(sentence)
        trainingSentences = loadTrainingSentences(file_path)
        trainingSentenceEmbeddings, indexToTrainingSentence = self.sentenceToEmbeddings(trainingSentences)
        # trainingSentenceEmbeddings, indexToTrainingSentence = X.sentenceToEmbeddings(trainingSentences)
        distances = list()
        for i in range(trainingSentenceEmbeddings.shape[0]):
            dist = cosineSimilarity(devSentenceEmbedding, trainingSentenceEmbeddings[i])
            # extremely near 1 training sentence, directly return
            # set it to 0.8 can achieve 0.95 accuracy
            # 0.7 to 0.74 accuracy
            if dist >= 0.8:
                # print("Extreme similar case", dist, indexToTrainingSentence[i][0], indexToTrainingSentence[i][1])
                return indexToTrainingSentence[i][1]
            distances.append((i, dist))
        # return "exclude"
        distances.sort(key=lambda tup: tup[1], reverse=True)
        # print(distances)
        # no sentence has a particularly high similarity
        if distances[0][1] <= 0.3:
            return "no"

        dict = {}
        maxCnt = 0
        maxCategory = None
        # k set to sqrt of training data size
        # k = math.floor(math.sqrt(trainingSentenceEmbeddings.shape[0]))
        k = 5
        for i in range(k):
            index = distances[i][0]
            if distances[i][1] < 0.3:
                break
            # print(distances[i][1], indexToTrainingSentence[index][1], indexToTrainingSentence[index][0])
            neighborCategory = indexToTrainingSentence[index][1]
            neighborCategoryCnt = 0
            if neighborCategory in dict:
                neighborCategoryCnt = dict[neighborCategory] + 1
            else:
                neighborCategoryCnt = 1
            dict[neighborCategory] = neighborCategoryCnt
            if neighborCategoryCnt > maxCnt:
                maxCnt = neighborCategoryCnt
                maxCategory = neighborCategory
        # this can be configured to achieve better result
        # if maxCnt >= 3:
        #     print(maxCategory)
        #     return maxCategory
        # return "no"
        if maxCnt < 4:
            return "no"
        if maxCategory == "state" and "?" not in sentence:
            return "no"
        return maxCategory



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
        devSentences = loadTrainingSentences(dev_file_path)
        total_cnt = 0
        correct_cnt = 0
        for key, value in devSentences.items():
            for sentence in value:
                category = self.getCategory(sentence, training_file_path)
                # if category == "exclude":
                #     continue
                if category == key:
                    correct_cnt += 1
                total_cnt += 1
        return correct_cnt/total_cnt


    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False,
                 "on": False}

        ### YOUR CODE HERE ###
        tokens = tokenize(command)

        if 'increase' in tokens or 'add' in tokens:
            slots['add'] = True
        if 'decrease' in tokens or 'subtract' in tokens or 'lower' in tokens or 'Weaken' in tokens or 'dim' in tokens:
            slots['sub'] = True
        if 'holoemitter' in tokens or 'holoEmit' in tokens:
            slots['holoEmit'] = True
        if 'logic' in tokens or 'logDisp' in tokens:
            slots['logDisp'] = True
        if 'off' in tokens or 'end' in tokens or 'minimum' in tokens or 'out' in tokens:
            slots['off'] = True
        if 'on' in tokens or 'start' in tokens or 'maximum' in tokens:
            slots['on'] = True
        slots['lights'] = []
        if 'front' in tokens or 'forward' in tokens:
            slots['lights'].append('front')
        if 'back' in tokens or 'backward' in tokens:
            slots['lights'].append('back')
        if 'both' in tokens or 'all' in tokens:
            slots['lights'] = ['front', 'back']
        if len(slots['lights']) == 0:
            slots['lights'] = ['front', 'back']
        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left".
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        tokens = tokenize(command)
        tokens = ['back' if token == 'south' else token for token in tokens]
        tokens = ['right' if token == 'east' else token for token in tokens]

        if 'increase' in tokens:
            slots['increase'] = True
        if 'decrease' in tokens:
            slots['decrease'] = True
        possible_dirs = ['forward', 'back', 'right', 'left']
        dirs = [token for token in tokens if token in possible_dirs]
        slots['directions'] = dirs

        return slots





############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 20

feedback_question_2 = """
NO
"""

feedback_question_3 = """
No
"""