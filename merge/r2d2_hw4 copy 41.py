############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Arjun Lal"

############################################################
# Imports
############################################################

from pymagnitude import *
import numpy as np
import string
import re
import math



############################################################
# Helper Functions
############################################################

def loadTrainingSentences(file_path):
    commandTypeToSentences = {}

    with open(file_path, 'r', encoding = "utf-8") as fin:
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
my_driving_sentences = ["Set speed to be 50%", "Set speed to be 10%", "Set speed to be 30%", "Set speed to be 40%", "Set speed to be 60%",\
                        "Set speed to be 70%", "Set speed to be 80%", "Set speed to be 90%", "Set speed to be 100%", "Move fast!"]
my_light_sentences = ["Lights on", "Lights off now", "Lights up!", "only red lights", "only blue lights", "only green lights",\
                        "only yellow lights", "only orange lights", "all lights", "no lights at all"]
my_head_sentences = ["turn around!", "back up!", "look over there!", "look up", "look down", "look to the right", "look to the left",\
                        "rotate your head all around", "do a 180", "do a 360"]
my_state_sentences = ["where are you?", "what color are the lights?", "what's your speed?", "send me your location", "send me your speed",\
                        "where are you going?", "are you asleep?", "where are you headed?", "how many feet are you on?", "stable?"]

my_connection_sentences = ["disconnect all", "connect to the server", "how many droids nearby?", "Connect D2-55A1 to the server", \
                            "Connect D2-55A3 to the server", "Connect D2-55A4 to the server", "Connect D2-55A6 to the server"\
                            , "Connect D2-55A8 to the server", "Connect D2-66A2 to the server", "Connect D2-88A2 to the server"]
my_stance_sentences = ["use all wheels", "use one wheel",  "use three wheels", "use four wheels", "use a random number of wheels", \
                        "use five wheels", "use no wheels", "stand straight", "get down!", "go on tiptoes"]
my_animation_sentences = ["do something",  "do anything", "do nothing", "cry", "make a gesture", "do whatever you want", "be quiet", \
                            "jump!", "do a pushup", "do a situup"]
my_grid_sentences = ["Go to position (2,2)","Go to position (1,2)", "Go to position (2,1)" , "Go to position (3,2)", "Go to position (2,10)", \
                    "go to the origin", "go 2 steps back", "go 2 steps forward", "obstacle ahead", "obstacle behind you"]


############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    # pass
    # print(text)
    # print("just printed")
    # print(sentence)
    sentence = sentence.lower().strip()
    returned = []
    punc = string.punctuation

    cur = ""

    for i in sentence:
        if i.isspace():
            if cur == "":
                continue
            else:
                returned.append(cur)
                cur = ""
        elif i in punc:
            if cur == "":
                # returned.append(i)
                continue
            else:
                returned.append(cur)
                # returned.append(i)
                cur = ""
        else:
            cur += i

    if cur != "":
        returned.append(cur)

    return returned

def cosineSimilarity(vector1, vector2):
    # pass
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

class WordEmbeddings:

    def __init__(self, file_path):
        # pass
        self.vectors = Magnitude(file_path) 


    def calcSentenceEmbeddingBaseline(self, sentence):
        # pass
        returned = np.zeros(300)
        if sentence is None or sentence == "":
            return returned

        tokens = tokenize(sentence)

        for t in tokens:
            returned += self.vectors.query(t)

        return returned


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
        # pass

        num_sent = 0
        for k in commandTypeToSentences:
            num_sent += len(commandTypeToSentences[k])

        sentenceEmbeddings = np.zeros((num_sent, 300))
        indexToSentence = {}

        cur_index = 0
        for k in commandTypeToSentences:
            for sent in commandTypeToSentences[k]:
                cur_vec = self.calcSentenceEmbeddingBaseline(sent)
                # print(sentenceEmbeddings.shape)
                # print(cur_vec.shape)

                for i in range(300):
                    sentenceEmbeddings[cur_index, i] = cur_vec[i]

                
                indexToSentence[cur_index] = (sent, k)
                cur_index += 1

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
        # pass
        targ = self.calcSentenceEmbeddingBaseline(sentence)
        min_dist = -math.inf
        best_index = -1

        for i in range(sentenceEmbeddings.shape[0]):
            # dist = np.linalg.norm(targ - sentenceEmbeddings[i, : ])

            dist = cosineSimilarity(targ, sentenceEmbeddings[i, : ])
            if dist > min_dist:
                min_dist = dist
                best_index = i

        return best_index



    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''

        print("sentence: ", sentence)

        with open(file_path, 'r', encoding = "utf-8") as fin:
            for line in fin:
                line = line.rstrip('\n')
                print(line)

        r1 = re.findall(r"\d,\d",sentence)
        if len(r1) > 0:
            return 'grid'

        r2 = re.findall(r"\d by \d",sentence)
        if len(r2) > 0:
            return 'grid'

        if ("light" in sentence and "What" not in sentence and "what" not in sentence) or "holoemit" in \
            sentence or "red" in sentence or "logic" in sentence:
            return 'light'

        if "speed" in sentence or "North" in sentence or "north"  in sentence \
            or "South" in sentence or "south" in sentence or "East" in sentence or "east" in sentence \
                or "West" in sentence or "west" in sentence or "drive" in sentence:
            return 'driving'

        if "position" in sentence or "grid" in sentence:
            return 'grid'

        if "name" in sentence or "What is" in sentence or "what is" in sentence:
            return 'state'

        if "Fall" in sentence or "scream" in sentence or "Laugh" in sentence or "Play an" in sentence or "noise" \
            in sentence or "speak" in sentence or "Speak" in sentence or "sing" in sentence or "Sing" in sentence:
            return 'animation'

        targ = self.calcSentenceEmbeddingBaseline(sentence)

        d = loadTrainingSentences(file_path)

        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(d)

        for i in range(sentenceEmbeddings.shape[0]):
            cur_vec = sentenceEmbeddings[i, : ]



        return "no"
        # pass

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
feedback_question_1 = 2

feedback_question_2 = """
na
"""

feedback_question_3 = """
na

"""


