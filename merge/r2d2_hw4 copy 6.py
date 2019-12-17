############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Yuezhan Tao"

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
my_driving_sentences = ["Go forward, then turn right.",
"North is heading 50 degrees.",
"Go South.",
"Go West.",
"Go South-by-southeast",
"Run away please!",
"Turn to heading 90 degrees.",
"Reset your heading.",
"Turn to North.",
"Start rolling forward.",
"Increase your speed by 50%.",
"Turn to your right."
]

my_light_sentences = ["Change the intensity on the holoemitter to minimum.",
"Turn on the holoemitter.",
"Blink your logic display.",
"Change the front LED to green.",
"Turn your front light green.",
"Dim your lights holoemitter.",
"Turn on all your lights.",
"Set the RGB values on your lights to be 0,255,0.",
"Add 50 to the red value of your front LED.",
"Decrease the blue value of your back LED by 50%.",
]

my_head_sentences = ["Turn your head to face forward.",
"Look behind you.",
"Look forward",
"Turn your head to north",
"Look south",
"Turn right",
"Turn your head to face backward",
"See front",
"Turn left",
"Turn around"
]

my_state_sentences = ["What color is your back light?",
"Tell me what color your back light is set to.",
"Is your logic display off?",
"What direction are you facing now?",
"Are you standing on 5 feet or 6?",
"What is your current heading?",
"How much battery do you have?",
"Are you driving right now?",
"How fast are you going?",
"What is your current speed?",
"Is your back light blue?"
]

my_connection_sentences = [
"Connect to the server",
"Are there any other droids nearby?",
"You can stop.",
"Disconnect from the server.",
"Turn on.",
"System start.",
"Mission start.",
"Lets sleep.",
"Stop.",
"Shut down."
]

my_stance_sentences = [
"Set your stance to be biped.",
"Put down your third wheel.",
"Stand on your tiptoes.",
"Stand with one foot",
"Put down your second wheel",
"Put down your last wheel",
"Stand with your heel",
"sit down",
"lie down",
"jump"
]

my_animation_sentences = [
"Fall over",
"Scream",
"Make some noise",
"Laugh",
"Play an alarm",
"Smile",
"Sing a song",
"Turn around",
"Jump",
"hit me"
]

my_grid_sentences = [
"You are on a 3 by 3 grid.",
"Each square is 2 foot large.",
"You are at position (2,0).",
"Go to position (4,3).",
"There is an obstacle at position 1,1.",
"There is a chair at position 2,3",
"Go to the left of the chair.",
"It’s not possible to go from 2,5 to 2,5.",
"There is a table at position 3,4",
"Move to position 2,5"
]


############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    sentence = sentence.lower()
    sentence = sentence.strip()
    token = ""
    result = []
    special = ["\t", "\n", "\r", "\x0b", "\x0c"]
    for i in range(len(sentence)):
        ch = sentence[i]      
        if ch not in special:
            if ch == " ":
                if token != "":
                    result.append(token)
                    token = ""
                    continue
                else:
                    token = ""
                    continue
            elif ch in string.punctuation:
                if token != "":
                    result.append(token)
                # result.append(ch)
                token = ""
                continue
            else:
                token = token + ch
                if i == len(sentence) - 1:
                    result.append(token)
    return result  

def cosineSimilarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.sqrt(np.sum(vector1 ** 2))
    norm2 = np.sqrt(np.sum(vector2 ** 2))
    return dot_product/ (norm1 * norm2)

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path) 
        # print(self.Magnitude)
    def calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)
        # print(tokens)
        # print(self.Magnitude.query(tokens).shape)
        if len(tokens) == 0:
            return np.zeros(300)
        return np.sum(self.vectors.query(tokens), axis=0, dtype=float)


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

        indexToSentence = {}
        i = -1
        for category in commandTypeToSentences:
            category_sentence = commandTypeToSentences[category]
            # print(category_sentence)
            # print(len(category_sentence))
            for sentence in category_sentence:
                i += 1
                indexToSentence[i] = (sentence, category)
            
        # print(self.Magnitude.dim)
        sentenceEmbeddings = np.zeros((len(indexToSentence), self.vectors.dim))
        # print(len(indexToSentence))
        for j in range(len(indexToSentence)):
            sen = indexToSentence[j][0]
            tmp = self.calcSentenceEmbeddingBaseline(sen)
            sentenceEmbeddings[j, :] = tmp

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
        
        closest_val = -float("inf")
        closest_ind = -1
        
        sen_embed = self.calcSentenceEmbeddingBaseline(sentence)
        for i in range(len(sentenceEmbeddings)):
            tmp = cosineSimilarity(sen_embed, sentenceEmbeddings[i, :])
            if tmp > closest_val:
                closest_val = tmp
                closest_ind = i

        return closest_ind
    
    # def mytokenize(self, sentence):
    
    def new_calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)
        # print(tokens)
        # print(self.Magnitude.query(tokens).shape)
        if len(tokens) == 0:
            return np.zeros(300)
        # print(tokens) 
        remove = ["s", "is", "a", "an", "the", "to", "re"]
        new_tokens = []
        for token in tokens:
            if token not in remove:
                new_tokens.append(token)
        print(sentence)
        print(new_tokens)
        return np.sum(self.vectors.query(new_tokens), axis=0, dtype=float)
    
    def addTraindata(self, data):
        data["animation"].append("Sing")
        data["animation"].append("Speak")
        return data
    
    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        data = loadTrainingSentences(file_path)
        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(data)
        # print(data)
        data = self.addTraindata(data)
        
        closest_lst = []
        sen_embed = self.new_calcSentenceEmbeddingBaseline(sentence)
        for i in range(len(sentenceEmbeddings)):
            cosine = cosineSimilarity(sen_embed, sentenceEmbeddings[i, :])
            closest_lst.append((i, cosine))
        
        closest_lst.sort(key = lambda x: x[1], reverse = True)
        # print(closest_lst)
        if cosineSimilarity(sen_embed, sentenceEmbeddings[closest_lst[0][0], :]) < 0.6:
            return "no"

        k = 6
        vote = {}
        
        for j in range(k):
            if j == 0:
                weight = 1
            elif j == 1 or j == 2:
                weight = 0.4
            elif j == 3 or j == 4:
                weight = 0.15
            else:
                weight = 0.05
            if indexToSentence[closest_lst[j][0]][1] in vote.keys():
                vote[indexToSentence[closest_lst[j][0]][1]] += weight
            else:
                vote[indexToSentence[closest_lst[j][0]][1]] = weight
        
        most_likely = None
        most_likely_val = -float("inf")
        for key, val in vote.items():
            if val > most_likely_val:
                most_likely = key
                most_likely_val = val
            
        print(vote)
        print(most_likely)
        return most_likely
        

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
        dev = loadTrainingSentences(dev_file_path)
        for cate in dev:
            for sen in dev[cate]:
                s += 1
                if self.getCategory(sen, training_file_path) == cate:
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
        holo = ["holo emit", "holoemit"]
        log = ["logic display", "logicdisplay"]
        add = ["add", "increase", "speed up", "speedup"]
        sub = ["sub", "reduce", "devrese"]
        front = ["front", "forward", "head"]
        back = ["back", "backward"]
        
        command = command.strip()
        command = command.lower()
        
        for holo_param in holo:
            if holo_param in command:
                slots["holoEmit"] = True
        
        for log_param in log:
            if log_param in command:
                slots["logDisp"] = True
                
        for add_param in add:
            if add_param in command:
                slots["add"] = True
        
        for sub_param in sub:
            if sub_param in command:
                slots["sub"] = True
                
        for front_param in front:
            if front_param in command:
                if "front" not in slots["lights"]:
                    slots["lights"].append("front")
        
        for back_param in back:
            if back_param in command:
                if "back" not in slots["lights"]:
                    slots["lights"].append("back")
        
        if slots["lights"] == []:
            slots["lights"] = ["front", "back"]
        
        if "maximum" in command:
            slots["on"] = True
        
        if "minimum" in command:
            slots["off"] = True
        
        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###
        increase = ["increase", "speed up", "speedup", "faster"]
        decrease = ["decrease", "slow down", "slowdown", "slow", "slower"]
        north = ["north", "forward", "straight", "ahead"]
        south = ["south", "backward", "back"]
        west = ["west", "left"]
        east = ["east", "right"]
                
        command = command.strip()
        command = command.lower()
        tokens = tokenize(command)
        
        for token in tokens:
            if token in increase:
                slots["increase"] = True
            elif token in decrease:
                slots["decrease"] = True
            elif token in north:
                slots["directions"].append("forward")       
            elif token in south:
                slots["directions"].append("back")
            elif token in west:
                slots["directions"].append("left")
            elif token in east:
                slots["directions"].append("right")
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
