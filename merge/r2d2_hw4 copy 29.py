############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Ezaan Mangalji, Rishab Jaggi"

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
    "go straight 2 feet",
    "move forward",
    "move north",
    "drive forward",
    "cease movement",
    "halt!",
    "full speed ahead",
    "turn to be east",
    "turn to be south",
    "speed up by 50%"
]

my_light_sentences = [
    "turn off your lights",
    "turn your lights red",
    "blink your lights",
    "back lights red",
    "lights up r2",
    "lights maximum",
    "set the rgb values of your lights to be 100, 100, 100",
    "decrease the red value of your rear lights",
    "turn off your back lights",
    "dim your lights"
]

my_head_sentences = [
    "spin around", 
    "look to your right",
    "look to your left",
    "spin to look north",
    "face east",
    "face north",
    "face south",
    "face west",
    "look behind you",
    "180 spin"
]

my_state_sentences = [
    "whats your stance?",
    "whats your lights colour?",
    "sup with your logic display?",
    "sup with your battery?",
    "are you charged?",
    "are you alive?",
    "what speed are you at?",
    "do you like darth vader?",
    "which way are you facing",
    "diagnostics"
]

my_connection_sentences = [
    "connect",
    "interface with the server",
    "disconnect",
    "quit the server",
    "join the server connection",
    "do you see other droids?",
    "scan the server",
    "quit your connection",
    "connect r2d2",
    "connect r2q5"
]

my_stance_sentences = [
    "put all your legs down",
    "stand on 2 legs",
    "full stance",
    "2 feet only",
    "3 feet",
    "itty bitty tippy toesies",
    "regular stance",
    "pull your third leg up",
    "put your third leg down",
    "stick to three feet"
]

my_animation_sentences = [
    "noise",
    "alert the surroundings",
    "sound the alarm",
    "play dead",
    "fall",
    "yell",
    "make some sounds",
    "make a noise",
    "bird sounds",
    "get on the floor"
]

my_grid_sentences = [
    "youre on a 3 by 3 grid",
    "squares are 2 feet",
    "go to (3, 2)",
    "theres obstacles from 2,2 to 3,3",
    "youre at (3,3)",
    "you cant go (1,1)",
    "move around the dog at (2,2)",
    "youre positioned at (5,3)",
    "the grid is 2 by 2",
    "grid sizes are 1 foot"
]



############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    sentence = sentence.lower()
    for p in string.punctuation:
        sentence = sentence.replace(p, " "+p+" ")
        
    sentence = sentence.replace("\n", "")
    sentence = sentence.replace("\t", "")
    sentence = sentence.replace("\r", "")
    sentence = sentence.replace("\x0b", "")
    sentence = sentence.replace("\x0c", "")
    
    spaceSplit = list(filter(lambda x: len(x) > 0 and x not in string.punctuation, sentence.split(" ")))
    return spaceSplit

def cosineSimilarity(vector1, vector2):
    dot = np.dot(vector1, vector2)
    length1 = np.sqrt(np.dot(vector1, vector1))
    length2 = np.sqrt(np.dot(vector2, vector2))
    return dot / (length1 * length2)

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)
        vect = np.zeros(300)
        for token in tokens:
            v = self.vectors.query(token)
            vect = np.add(vect, v)
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
        matrix = np.zeros((0, 300))
        s2c = {}
        i = 0
        for (command, sentList) in commandTypeToSentences.items():
            for sent in sentList:
                index = matrix.shape[0]
                sentEmbedding = self.calcSentenceEmbeddingBaseline(sent)
                matrix = np.vstack([matrix, sentEmbedding])
                s2c[index] = (sent, command)
        return (matrix, s2c)

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
        embedding = self.calcSentenceEmbeddingBaseline(sentence)
        index = 0
        maxSimilarity = -1
        for i in range(len(sentenceEmbeddings)):
            similarity = cosineSimilarity(embedding, sentenceEmbeddings[i])
            if similarity > maxSimilarity:
                index = i
                maxSimilarity = similarity
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
        command = sentence.lower()
        for p in string.punctuation:
            command = command.replace(p, " "+p+" ")
        command = command.split()
        embedding = self.calcSentenceEmbeddingBaseline(sentence)
        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(loadTrainingSentences(file_path))
        similarities = [-1] * len(sentenceEmbeddings)
        for i in range(len(sentenceEmbeddings)):
            similarities[i] = (i, cosineSimilarity(embedding, sentenceEmbeddings[i]))
        similarities.sort(key=lambda pair: pair[1], reverse = True)

        k = 20

        #  k tuples of the form (category, similarity) for top k most similar sentences
        top_k = [(indexToSentence[pair[0]][1], pair[1]) for pair in similarities[:k]]

        # dict from category -> (# sentences in that cat in top_k, avg similarity of sentences in that cat)
        category_map = {}
        max_sim = -1
        for cat, sim in top_k:
            if cat not in category_map:
                category_map[cat] = (1, sim)
            else:
                count, cur = category_map[cat]
                category_map[cat] = (count+1, (cur*count+sim)/(count+1))
            max_sim = max(max_sim, sim)
        
        # category with highest average similarity
        max_avg = -1
        for cat, tup in category_map.items():
            if tup[1] > max_avg:
                max_avg = tup[1]
                result = cat

        if 'sing' in command or 'speak' in command or 'noise' in command:
            return 'animation'

        if max_sim < 0.75:
            return 'no'
        
        if result == 'state' and sentence[-1] != "?":
            return 'no'
        
        return result



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
        devCommandTypeToSentences = loadTrainingSentences(dev_file_path)        
        c, s = 0, 0
        for ctype, sentences in devCommandTypeToSentences.items():
            for sentence in sentences:
                s += 1
                if ctype == self.getCategory(sentence, training_file_path):
                    c += 1
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
        command = command.lower()
        for p in string.punctuation:
            command = command.replace(p, " "+p+" ")
        command = command.split()

        # HoloEmit
        if 'holoemitter' in command or 'holo' in command:
            slots['holoemitter'] = True
        
        # LogDisp
        if 'logic' in command or 'display' in command:
            slots['logDisp'] = True

        #Lights
        if 'lights' in command or 'both' in command:
            slots['lights'] = ['front', 'back']
        else:
            if 'front' in command or 'forward' in command:
                slots['lights'].append('front')
            if 'back' in command:
                slots['lights'].append('back')
        
        # Add
        if 'add' in command or 'brighten' in command or 'increase' in command:
            slots['add'] = True

        # Sub
        if 'add' in command or 'dim' in command or 'decrease' in command or 'reduce' in command:
            slots['sub'] = True

        # Off
        if 'off' in command or 'out' in command:
            slots['off'] = True
        
        # On
        if 'on' in command or 'maximum' in command:
            slots['on'] = True

        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###
        command = command.lower()
        for p in string.punctuation:
            command = command.replace(p, " "+p+" ")
        command = command.split()

        # Increase
        if 'increase' in command or 'start' in command or 'run' in command:
            slots['increase'] = True
        
        # Decrease
        if 'decrease' in command or 'stop' in command or '0' in command:
            slots['decrease'] = True
        
        # Directions
        directions = {'forward' : 'forward', 'back' : 'back', 'left' : 'left', 'right' : 'right'}
        directions['north'] = 'forward'
        directions['east'] = 'right'
        directions['west'] = 'left'
        directions['south'] = 'back'
        for piece in command:
            if piece in directions:
                slots['directions'].append(directions[piece])

        return slots


############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 4

feedback_question_2 = """
No significant stumbling blocks.
"""

feedback_question_3 = """
I really liked trying it out at the end!
"""
