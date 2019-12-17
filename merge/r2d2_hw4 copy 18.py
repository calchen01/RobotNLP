############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Rakesh Nagda"

############################################################
# Imports
############################################################

from pymagnitude import *
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import string
import re

############################################################
# Helper Functions
############################################################
both_lights = ["both the lights", "lights", "both lights", "both", "all", "LEDs"]
front_light = ["front", "one light"]
back_light = ["back", "rear"]
holoEmit = ["holoemitter", "holoEmit", "turn on holoemitter", "turn on your holoemitter", "set the holoemitter"]
logDisp = ["display", "logic display", "logDisp", "Lower"]
add = ["add", "increase", "strengthen"]
sub = ["sub", "decrease", "weaken", "dim"]
on = ["turn on", "turn your light on", "maximum", "lights on", "light on"]
off = ["turn off", "off", "minimum", "lights out"]

inc = ["increase", "fast", "run", "higher", "raise", "run away", "start rolling"]
dec = ["decrease", "slow", "lower", "stop"]
left = ["left", "west"]
right = ["right", "east"]
forward = ["forward", "front", "north"]
backward = ["backward", "back", "south"]

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
my_animation_sentences = ["fall over", "sing", "speak", "make a noise", "scream", "laugh", "play alarm", "dance", "jump", "play music"]
my_driving_sentences = ["turn east, and then go forward for 10 feet.", "drive forward", "go left", "head west", "go backward", "run!", "halt.", "darth vader is behind you! run away!", "lower your speed", "start rolling."]
my_grid_sentences = ["It's not possible to go to square 5,4.", "It's possible to go to square 5,5.", "There is a truck on 6,6.", "you are at position (1,1)", "You are on 10 by 9 grid.", "each rectangle is one foot long", "go to position (11,5)", "go to the right of the refrigerator", "there is a baby at position (3,4)", "there is a wall right behind you"]
my_head_sentences = ["Turn your head to the left", "Turn your head to the right", "Turn head to the east", "heading towads east", "head towards south", "turn head to west", "head to north", "look forward", "look ahead", "look back", "see at the back", "look down"]
my_light_sentences = ["set the holoemitter to minimum.", "set the holoemit to maximum.", "turn off your light", "turn on your red light", "turn off all the lights", "turn on all the lights", "change the colour of your light", "display something", "display something intresting", "display exciting", "weaken your all the lights" ]
my_stance_sentences = ["Set your stance to be biped.", "Put down your third wheel.", "put up your third wheel.", "stand on your tiptoes.", "can you stand on one leg?", "try to stand on your any one leg", "put up your both the wheels", "lift your third leg", "stand on your third wheel", "lift all the legs up", "can you remove your third leg?", "stand on two legs"]
my_state_sentences = ["What is the color on your front light?", "Are there other R2D2s nearby?", "tell me your name", "your name is dodo", "what is your battery status", "on how many legs you arestanding right now", "what is your current orientation", "tell me the colour of your all the lights", "In which direction your head is?", "is your holoemitter on?"]
my_connection_sentences = ["Connect D2-D4C0 to the server", "Connect D2-D4C0", "any other droid nearby", "disconnect droid", "reconnect droid again", "disconnect droid from server", "hay server! connect my droid", "don't disconnect r2d2", "name of nearby droids please", "establish connection", "permission granted to connect to D2-D4C0"]
############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    # regEx = re.compile("[^a-z0-9]+")
    # sentence = sentence.lower()
    # output = list((regEx.sub(' ',sentence).strip()).split(' '))
    # print(output)
    # return output
    result = []
    word = ""
    sentence = sentence.lower()
    for c in sentence:
        if c not in string.punctuation and c!=' ':
            word += c
        elif word!="":
            result.append(word)
            word = ""
        
    return result

def cosineSimilarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        word_vector = tokenize(sentence)

        if len(word_vector)==0 or len(word_vector[0])==0:
            sentence_vec = np.zeros((300,))
        else:  
            vec = np.array(self.vectors.query(word_vector))
            sentence_vec = np.sum(vec, axis=0)
        return sentence_vec

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
        sentenceEmbeddings = np.zeros((0,300))
        indexToSentence = {}
        i = 0
        for aCatagory in commandTypeToSentences.keys():
            for aSentence in commandTypeToSentences[aCatagory]:
                sentence_vec = self.calcSentenceEmbeddingBaseline(aSentence)
                try:
                    sentenceEmbeddings = np.vstack((sentenceEmbeddings,sentence_vec))
                except:
                    sentenceEmbeddings = sentence_vec

                indexToSentence[i] = (aSentence, aCatagory)
                i += 1

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
        testSentenceEmbedding = self.calcSentenceEmbeddingBaseline(sentence)
        max_dist = float('-inf')
        i = 0
        max_i = 0
        for anSentenceEmbedding in sentenceEmbeddings:
            dist = self.vectors.similarity(testSentenceEmbedding, anSentenceEmbedding)
            if dist > max_dist:
                max_i = i
                max_dist = dist
            i += 1

        return max_i

    def scikit_sentenceToEmbeddings(self, commandTypeToSentences):
        sentenceEmbeddings = np.zeros((0,300))
        documents_all = []
        indexToSentence = {}
        i = 0
        for aCatagory in commandTypeToSentences.keys():
            for aSentence in commandTypeToSentences[aCatagory]:
                aDoc = ' '.join(tokenize(aSentence))
                documents_all.append(aDoc)
                indexToSentence[i] = (aSentence, aCatagory)
                i += 1

        return (documents_all, indexToSentence)

    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        K = 5
        trainingSentences = loadTrainingSentences(file_path)
        (documents_all, indexToSentence) = self.scikit_sentenceToEmbeddings(trainingSentences)
        test_document = documents_all
        test_document.append(' '.join(tokenize(sentence)))

        sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True)
        sklearn_representation = sklearn_tfidf.fit_transform(test_document)
        vec_matrix = np.array(sklearn_representation.toarray().tolist())
        pairwise_similarity = np.dot(vec_matrix,vec_matrix.T) 
        arr = pairwise_similarity    
        np.fill_diagonal(arr, np.nan)
        # result_idx = np.nanargmax(arr[vec_matrix.shape[0]-1])
        nearest_neighbour_i = np.array(arr[vec_matrix.shape[0]-1][0:vec_matrix.shape[0]-1]).argsort()[-K:]
        nearest_catagory = []
        similarty = []
        for idx in nearest_neighbour_i:
            # print(arr[vec_matrix.shape[0]-1][idx],indexToSentence[idx])
            similarty.append(arr[vec_matrix.shape[0]-1][idx])
            nearest_catagory.append(indexToSentence[idx][1])
        similarty.reverse()
        nearest_catagory.reverse()

        if similarty[0] < 0.3:
            return 'no'
        else:
            return nearest_catagory[0]

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
        testSentences = loadTrainingSentences(dev_file_path)
        c = 0.0
        tot = 0.0
        for k, v in testSentences.items():
            for s in v:
                best_fit = self.getCategory(s, training_file_path)
                if(k == best_fit): c +=1.0
                tot += 1.0
        return c/tot

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''

        ### YOUR CODE HERE ###
        command = command.lower()

        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}
        is_light = True

        for pattern in holoEmit:
            if re.search(pattern, command):
                slots["holoEmit"] = True
                is_light = False

        if is_light == True:
            for pattern in logDisp:
                if re.search(pattern, command):
                    slots["logDisp"] = True

        if is_light == True:
            for pattern in both_lights:
                if re.search(pattern, command):
                    slots["lights"] = ['front', 'back']
            for pattern in front_light:
                if re.search(pattern, command):
                    slots["lights"] = ['front']
            for pattern in back_light:
                if re.search(pattern, command):
                    slots["lights"] = ['back']

            for pattern in add:
                if re.search(pattern, command):
                    slots["add"] = True
            for pattern in sub:
                if re.search(pattern, command):
                    slots["sub"] = True
            
            for pattern in on:
                if re.search(pattern, command):
                    slots["on"] = True
            for pattern in off:
                if re.search(pattern, command):
                    slots["off"] = True
        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###
        command = command.lower()

        for pattern in inc:
            if re.search(pattern, command):
                slots["increase"] = True
        for pattern in dec:
            if re.search(pattern, command):
                slots["decrease"] = True

        index = []
        for pattern in forward:
            index.extend([(m.span()[0],"forward") for m in re.finditer(pattern, command)])

        for pattern in backward:
            index.extend([(m.span()[0],"back") for m in re.finditer(pattern, command)])

        for pattern in left:
            index.extend([(m.span()[0],"left") for m in re.finditer(pattern, command)])

        for pattern in right:
            index.extend([(m.span()[0],"right") for m in re.finditer(pattern, command)])

        index.sort()

        for i, action in index:
            slots["directions"].append(action)

        return slots


############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 30

feedback_question_2 = """
(getCategory) part was really challenging.
"""

feedback_question_3 = """
I loved this exciting assignment. Next time it should be a main assignment, so that nobody misses it.
"""
