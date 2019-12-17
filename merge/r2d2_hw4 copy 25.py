############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Emily Tan"

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
    'Turn right.',
    'Turn left.',
    'Stop rolling.',
    'Turn to face backwards.',
    'Get ready to roll.',
    'Back out for 2 seconds.',
    'Spin in a circle.',
    'Full speed ahead.',
    'Shuffle right for 2 seconds.',
    'Shuffle left for 2 seconds.',]

my_light_sentences = [
    'Turn off both lights.',
    'Turn off the front LED light.',
    'Set the Holo Projector to max brightness.',
    'Set the Logic Displays to max brightness.',
    'Set the front LED light to fuchsia.',
    'Set the front LED light to green.',
    'Set the back LED light to blue.',
    'Partly dim the Holo Projector.',
    'Partly dim the Logic Displays.',
    'Turn off both the Holo Projector and Logic Displays.',]

my_head_sentences = [
    'Look infront you.',
    'Look backwards.',
    'Look behind you.',
    'Look to your right.',
    'Look to your left.',
    'Look slightly right.',
    'Look slight left.',
    'Turn your head back around.',
    'Look over your right shoulder.',
    'Look over your left shoulder.',]

my_state_sentences = [
    'Which way are you facing?',
    'Is your front light on?',
    'Is your back light on?',
    'Are you in waddle position?',
    'How much battery life do you have left?',
    'How are you standing?',
    'Are you rolling right now?',
    'How fast are you rolling?',
    'r2d2, are you awake or not?',
    'Are both your Holo Projector and Logic Displays on?',]

my_connection_sentences = [
    'Scan for nearby droids.',
    'Is there potential danger nearby?',
    'Are there any friends near us?',
    'Connect r2d2 to server.',
    'Disconnect the droid from the server.',
    'Exit the connection.',
    'Connect droid.',
    'Disconnect droid.',
    'Leave the server.',
    'Search surroundings.',]

my_stance_sentences = [
    'Stand on two feet.',
    'Stand on three feet.',
    'Raise your third foot.',
    'Drop your third foot.',
    'Waddle stance, go.',
    'Get ready to waddle.',
    'Stablilize.',
    'Destabilize.',
    'Stop waddling.',
    'Training wheel off.',]

my_animation_sentences = [
    'Faint.',
    'Rev your engine.',
    'Sound the alarm!',
    'Act surprised.',
    'Get excited.',
    'Start laughing.',
    'Do a double take.',
    'Say no.',
    'Act scared.',
    'Start yelling.',]

my_grid_sentences = [
    'You are on a 4 by 4 grid.',
    'You are currently starting at (0,0).',
    'Find a path to (3,3).',
    'There is no edge between (1,1) and (2,2).',
    'Go to (2,2).',
    'Your starting position is (0,0).',
    'The grid has 4 rows and 4 columns.',
    'There is an obstacle at (0,3).',
    'You are not allowed to travel from (1,1) to (2,3).',
    'You finish when you reach (3,3).',]


############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"


def tokenize(sentence):
    tokens = []
    token = ''
    for i in sentence:
        if i.isalnum():
            token = token + i.lower()
        else:
            if token != '':
                tokens.append(token)
                token = ''
    if token != '':
        tokens.append(token)
    return tokens


def cosineSimilarity(vector1, vector2):
    l_v1 = np.linalg.norm(vector1)
    l_v2 = np.linalg.norm(vector2)
    dot_prod = np.dot(vector1, vector2)
    cos = dot_prod / (l_v1 * l_v2)
    return cos


class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        words = tokenize(sentence)
        s_v = np.zeros((300,))
        for wd in words:
            w_v = self.vectors.query(wd)
            s_v = np.add(s_v, w_v)
        return s_v

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
            sentenceEmbeddings: A mxn numpy array where m[i:] contains the embedding
            for sentence i.

            indexToSentence: A dictionary with key: index i, value: (category, sentence).
        '''
        i = 0
        temp = []
        indexToSentence = {}
        for category, sentences in commandTypeToSentences.items():
            for each in sentences:
                s_v = self.calcSentenceEmbeddingBaseline(each)
                temp.append(s_v)
                indexToSentence.update({i: (each, category)})
                i = i + 1
        sentenceEmbeddings = np.reshape(temp, (len(temp), 300))
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
        max = -1
        i = 0
        index = 0
        s_v = self.calcSentenceEmbeddingBaseline(sentence)
        for s in sentenceEmbeddings:
            if cosineSimilarity(s_v, s) > max:
                max = cosineSimilarity(s_v, s)
                index = i
            i = i + 1
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
        wds = tokenize(sentence)
        for each in wds:
            if each == 'your' or each == 'the':
                wds.remove(each)
        new_sentence = ' '.join(wds)
        s_v = self.calcSentenceEmbeddingBaseline(new_sentence)
        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(loadTrainingSentences(file_path))
        i = self.closestSentence(new_sentence, sentenceEmbeddings)
        nearest, category = indexToSentence[i]
        if cosineSimilarity(self.calcSentenceEmbeddingBaseline(nearest), s_v) < 0.6 or 'I' in sentence \
                or 'We' in sentence:
            category = 'no'
        return category

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
        s = 0
        c = 0
        dev_sent = loadTrainingSentences(dev_file_path)
        for category, sentences in dev_sent.items():
            for sentence in sentences:
                est = self.getCategory(sentence, training_file_path)
                if est == category:
                    c = c + 1
                s = s + 1
        return float(c / s)

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
        wds = tokenize(command)
        for each in wds:
            if each.casefold() == 'holoemitter'.casefold():
                slots['holoEmit'] = True
            if each.casefold() == 'logic'.casefold():
                slots['logDisp'] = True
            if each.casefold() == 'front'.casefold() or each.casefold() == 'lights'.casefold() \
                    or each.casefold() == 'LEDs' or each.casefold() == 'forward'.casefold():
                if 'front' not in slots["lights"]:
                    slots["lights"].append("front")
            if each.casefold() == 'back'.casefold() or each.casefold() == 'lights'.casefold() \
                    or each.casefold() == 'LEDs' or each.casefold() == 'backward'.casefold():
                if 'back' not in slots["lights"]:
                    slots["lights"].append("back")
            if (each.casefold() == "on".casefold() and 'turn' in wds) or each == "maximum" \
                    or each.casefold() == "Brighten".casefold() or each.casefold() == "maximize".casefold():
                slots["on"] = True
            if each.casefold() == "off".casefold() or each == "minimum" or each.casefold() == "Dim".casefold() \
                    or each == "out" or each.casefold() == "Weaken".casefold() or each.casefold() == "minimize".casefold():
                slots["off"] = True
            if each.casefold() == "increase".casefold() or each.casefold() == "add".casefold()\
                    or each.casefold() == 'increment'.casefold():
                slots["add"] = True
            if each.casefold() == "decrease".casefold() or each.casefold() == "subtract".casefold()\
                    or each.casefold() == 'reduce'.casefold() or each == '0':
                slots["sub"] = True
        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###
        wds = tokenize(command)
        for w in wds:
            if w.casefold() == 'increase'.casefold():
                slots["increase"] = True
            if w.casefold() == 'decrease'.casefold():
                slots["decrease"] = True
            if w.casefold() == 'east'.casefold() or w.casefold() == 'right'.casefold():
                slots["directions"].append('right')
            if w.casefold() == 'north'.casefold() or w.casefold() == 'forward'.casefold():
                slots["directions"].append('forward')
            if w.casefold() == 'west'.casefold() or w.casefold() == 'left'.casefold():
                slots["directions"].append('left')
            if w.casefold() == 'south'.casefold() or w.casefold() == 'backwards'.casefold()\
                    or w.casefold() == 'back'.casefold():
                slots["directions"].append('back')
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
