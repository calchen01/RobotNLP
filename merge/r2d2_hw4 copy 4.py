############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Bharath Jaladi, Romit Nagda"

############################################################
# Imports
############################################################

from pymagnitude import *
import numpy as np
import string
import re
from collections import Counter

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
"Drive.",
"Begin driving.",
"Move forward.",
"Move at speed 50%.",
"Move forward for 5 seconds.",
"Stop moving.",
"Stop.",
"Restart.",
"Turn to face backward.",
"Decrease speed by 25%.",
"Slow down by 25%."]

my_light_sentences = [
"Set back light to blue.",
"Change back light to blue.",
"Set RGB on back light to 0, 0, 255.",
"Set your RGB on front light to 0,0,155.",
"Increase blue value in front light by 100.",
"Make your holoemitter logic blink.",
"Turn off your holoemitter logic.",
"Lights off.",
"Set front and back lights to blue.",
"Turn off all lights."]

my_head_sentences = [
"Turn your head 90 degrees.",
"Turn to your right.",
"Turn your head all the way around.",
"Turn your head 360 degrees.",
"Look right.",
"Look back.",
"Turn your head 180 degrees.",
"Look to the front.",
"Turn your head 0 degrees.",
"Look to your side."]

my_state_sentences = [
"Am I awake?",
"Am I facing forward?",
"What is my battery?",
"Is my blue value in my back light 255?",
"What is my RGB value in my back light?",
"Is my back light on?",
"Waddling?",
"Connected?",
"Am I waddling?",
"What is my continuous roll timer?"]

my_connection_sentences = [
"Connect D2-FF2.",
"Search for drone.",
"Connect D2-FF2 to yarn server.",
"Close connection.", 
"Terminate connection.",
"Leave.",
"Look for drones.",
"Disconnect from yarn server.",
"Exit.",
"Scan."]

my_stance_sentences = [
"Stand.",
"Set stance to stand.",
"Reset stance to standing.",
"Set waddle to true.",
"Waddle.",
"Stop waddling.",
"Set waddling to false.",
"Stop standing.",
"Stand still.",
"Begin waddling."]

my_animation_sentences = [
"Fall over immediately.",
"Fall over after 3 seconds.",
"Make any noise.",
"Make any noise in 1 minute.",
"Make a screeching noise.",
"Say ‘ouch’ and fall over.",
"Make any noise for 10 seconds.",
"Make any noise indefinitely.",
"Don’t make any noise.",
"Play an alarm in 3 minutes."]

my_grid_sentences = [
"You are on a 10 by 10 grid.",
"You are at position (4,9).",
"Go to position (9,4), if possible.",
"There is an edge from (4,9) to (5,9).",
"There is a barrier between (4,9) and (4,8).",
"Go around the barrier between (4,9) and (4,8).",
"Follow the given path.",
"Follow the optimal path from (4,9) to (9,4), if possible.",
"There is an edge from (9,4) to all of its neighboring squares in the grid.",
"Find the shortest path from (4,9) to (9,4)."]

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    return "".join(" " if c in string.punctuation + '’' else c for c in sentence).lower().split() 

def cosineSimilarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path) 

    def calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)
        if len(tokens) == 0:
            return np.zeros(np.shape(self.vectors.query("test")))
        else:
            queries = []
            for token in tokens:
                queries.append(self.vectors.query(token).astype(np.float64))
            return np.sum(queries, axis = 0)

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
        vectors = []
        mapping = {}
        count = 0
        for category in commandTypeToSentences.keys():
            for sentence in commandTypeToSentences[category]:
                vectors.append(self.calcSentenceEmbeddingBaseline(sentence))
                mapping[count] = (sentence, category)
                count += 1
        if len(vectors) > 0:
            return np.array(vectors), mapping
        return np.empty((0, self.vectors.query("test").shape[0])), mapping

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
        max_similarity = -np.float('inf')
        max_index = -1
        for index, comparison_vector in enumerate(sentenceEmbeddings):
            temp_similarity = cosineSimilarity(vector, comparison_vector)
            if temp_similarity > max_similarity:
                max_index = index
                max_similarity = temp_similarity
        return max_index

    def kClosestSentences(self, sentence, sentenceEmbeddings, k):
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
        cosine_similarities = []
        for index, comparison_vector in enumerate(sentenceEmbeddings):
            cosine_similarities.append((cosineSimilarity(vector, comparison_vector), index))
        return sorted(cosine_similarities, key=lambda x: x[0], reverse=True)[:k]

    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        trainingSentences = loadTrainingSentences(file_path)
        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(trainingSentences)
        kClosest = [(val[0], indexToSentence[val[1]][1]) for val in self.kClosestSentences(sentence, sentenceEmbeddings, 5)]
        if kClosest[0][0] > 0.8 and kClosest[0][1] != 'state':
            return kClosest[0][1]
        elif kClosest[0][1] == 'state':
            if kClosest[0][0] > 0.85 or (kClosest[0][0] > 0.8 and kClosest[1][1] == 'state' and kClosest[1][0] > 0.75):
                return 'state'
            freq = Counter([closest[1] for closest in kClosest])
            size = len(kClosest) 
            unique = len(np.unique([closest[1] for closest in kClosest]))
            for (category, count) in freq.items(): 
                if (count > size / 2) and kClosest[1][1] == category:
                    if category != 'state':
                        return category
            return 'no'
        else:
            freq = Counter([closest[1] for closest in kClosest])
            size = len(kClosest) 
            unique = len(np.unique([closest[1] for closest in kClosest]))
            if unique >= 3 and kClosest[0][0] - kClosest[size - 1][0] < 0.065:
                return 'no'
            for (category, count) in freq.items(): 
                if (count > size / 2) and (kClosest[0][1] == category or kClosest[1][1] == category):
                    if category != 'state':
                        return category
            if kClosest[0][0] - kClosest[size - 1][0] < 0.1:
                return 'no'
            return kClosest[0][1]

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
        c = 0
        s = 0
        development_set = loadTrainingSentences(dev_file_path)
        for category in development_set.keys():
            for sentence in development_set[category]:
                category_pred = self.getCategory(sentence, training_file_path)
                if category_pred == category:
                    c += 1
                s += 1
        return c / s

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

        ### YOUR CODE HERE ###
        command = tokenize(command)
        slots['holoEmit'] = 'holoemitter' in command
        slots['logDisp'] = 'logic display' in command
        if 'front' in command:
            slots['lights'] = ['front']
        if 'back' in command:
            slots['lights'] = ['back']
        if len(slots['lights']) == 0:
            slots['lights'] = ['front', 'back']
        slots['add'] = 'increase' in command
        slots['sub'] = 'decrease' in command
        slots['on'] = 'on' in command or 'maximum' in command
        slots['off'] = 'off' in command or 'minimum' in command

        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}
        command = tokenize(command)

        ### YOUR CODE HERE ###
        slots['increase'] = 'increase' in command
        slots['decrease'] = 'decrease' in command
        for token in command:
            if token == 'forward' or token == 'north':
                slots['directions'].append('forward')
            elif token == 'right' or token == 'east':
                slots['directions'].append('right')
            elif token == 'back' or token == 'south':
                slots['directions'].append('back')
            elif token == 'left' or token == 'west':
                slots['directions'].append('left')

        return slots


############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 3.5

feedback_question_2 = """
We found getCategory most challenging, but there were no significant stumbling blocks. 
"""

feedback_question_3 = """
We liked the entire assignment and there is nothing we would have changed.
"""

