############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Michael Shur"

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

#need ten of each
my_driving_sentences = ["Turn Right.", "Stop moving.", "Move Forward.",
                        "CCB says roll South.", "Warp speed ahead.",
                        "Freeze mister.", "Reverse.", "Face South.",
                        "Walk North.", "Turn left then go forward."]
my_light_sentences = ["Turn lights on.", "Turn lights off.", "Make your lights blue.",
                      "Make your lights green.", "Make your back light a random color.",
                      "Blink one of your lights.", "Subtract 50 from the green value of your front LED.",
                      "Decrease your logic display's brightness by 50%.", "Blink all of your lights twice.",
                      "Turn your front light green and your back light red."]
my_grid_sentences = ["You are on a 4 by 5 grid.", "Which cell are you currently on?",
                     "There is obstacle to your left.", "Go to cell (0,0).",
                     "How many cells are in the grid?", "Are you at the edge of the grid?",
                     "Are you at your flag?", "What cell is your flag on?",
                     "How many cells have you visited?", "Is another robot in the cell in front of you?"]
my_state_sentences = ["Are you currently standing still?", "Are you currently moving north?",
                      "What are the RGB values of your back light?", "Is your off the holoemitter on?",
                       "How many of your lights are currently on?", "Which of your lights are currently on?",
                       "What is your current speed?", "How much time until you run out of power?",
                      "What is your id?", "Are you a D2 or a Q5?"]
my_connection_sentences = ["Connect a random robot to the server.", "Disconnect a random robot.",
                           "How long does connecting usually take?", "Disconnect one of the Q5 robots.",
                           "Connect one of the D2 robots.", "How many robots are currently connected to the server?",
                           "Have all Q5 robots been disconnected from the server?", "How much time until a robot disconnects from inactivity?",
                           "Have all D2 robots been disconnected from the server?", "Are 4 robots connected to the server?"]
my_animation_sentences = ["Roar.", "Wiggle.", "Beep please.", "Dance.", "Sing a song.",
                          "Walk funny.", "Stop dancing.", "Shake your head", "Shake your third wheel.", "Stop wiggling."]
my_head_sentences = ["About face.", "Look to your left.", "Look south.", "Look at another random robot.",
                     "Look at an obstacle, if you can.", "What is your heading?", "Set heading to 40 degrees.",
                     "Look North for 2 seconds then look East.","Which direction are you currently looking?",
                     "How many times have you looked at an obstacle?"]
my_stance_sentences = ["Retract your third wheel.", "Stand up straight.", "Look Alive!", "Attention!", "Put your wheel away.",
                       "How many of your wheels are touching the ground?", "Stand against the empire.",
                       "March!", "Pick a random stance.", "Please take out your third wheel."]
############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def checkAndResetCharList(tokens, charList):
    if len(charList) > 0:
        token = "".join(charList)
        tokens.append(token)

    return []

def tokenize(sentence):
    punctuationSet = set(string.punctuation) 
    i = 0
    while i<len(sentence) and sentence[i].isspace():
        i+=1

    tokens = []
    charList = []
    while i<len(sentence):
        if sentence[i] in punctuationSet or sentence[i].isspace():
            charList = checkAndResetCharList(tokens, charList)
        else:
            charList.append(sentence[i].lower())
        i+=1
    charList = checkAndResetCharList(tokens, charList)
    return tokens

def cosineSimilarity(vector1, vector2):
    dot_product = vector1.dot(vector2)
    magnitude_vec1 = np.linalg.norm(vector1)
    magnitude_vec2 = np.linalg.norm(vector2)

    return dot_product/(magnitude_vec1*magnitude_vec2)

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)


    def calcSentenceEmbeddingBaseline(self, sentence):
        sentenceTokens = tokenize(sentence)
        vectorSum = np.zeros(self.vectors.emb_dim)
        for token in sentenceTokens:
            vec = self.vectors.query(token)
            vectorSum = vec + vectorSum

        return vectorSum

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
        category_sentence_tuples = [(sentence, category) for category, sentence_list in commandTypeToSentences.items() for sentence in sentence_list]

        m = len(category_sentence_tuples)
        n = self.vectors.emb_dim
        matrix = np.zeros((m,n))
        indexToSentenceDict = {}
        for i in range(m):
            category_sentence_tuple = category_sentence_tuples[i]
            sentence = category_sentence_tuple[0]
            embeddings = self.calcSentenceEmbeddingBaseline(sentence)
            matrix[i] = embeddings
            
            indexToSentenceDict[i] = category_sentence_tuple

        
        return (matrix, indexToSentenceDict)




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


        sentenceVec = self.calcSentenceEmbeddingBaseline(sentence)
        sentenceEmbeddingVec = sentenceEmbeddings[0]

        closestIndex = 0
        maxCosValue = cosineSimilarity(sentenceVec, sentenceEmbeddingVec)

        for i in range(1,len(sentenceEmbeddings)):
            sentenceEmbeddingVec = sentenceEmbeddings[i]
            cosSim = cosineSimilarity(sentenceVec, sentenceEmbeddingVec)

            if cosSim>maxCosValue:
                maxCosValue = cosSim
                closestIndex = i

        return closestIndex

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
        """
        sentence_vec = self.calcSentenceEmbeddingBaseline(sentence)
        
        n = self.vectors.emb_dim

        #2.17-2.248
        #if the provided sentence is more than 2.2 away from every category it belongs to no category
        min_dist = 2.2
        predicted_category = 'no'
        
        for category, sentence_list in training_sentences.items():

            sum_of_sentence_baselines = np.zeros(n)
            for sentence in sentence_list:
                sum_of_sentence_baselines += self.calcSentenceEmbeddingBaseline(sentence)

            mean_of_sentence_baselines = np.true_divide(sum_of_sentence_baselines, len(sentence_list))   

            dist = np.linalg.norm(sentence_vec-mean_of_sentence_baselines)
            #print(dist)
            if dist < min_dist:
                predicted_category = category
                min_dist = dist
                
        return predicted_category
        """

        matrix, indexToSentenceDict = self.sentenceToEmbeddings(training_sentences)
        i = self.closestSentence(sentence, matrix)

        closest_sentence, predicted_category = indexToSentenceDict[i]

        closest_sentence_vec = self.calcSentenceEmbeddingBaseline(closest_sentence)
        sentence_vec = self.calcSentenceEmbeddingBaseline(sentence)
        
        dist = np.linalg.norm(closest_sentence_vec-sentence_vec)
        if dist>=2.8:
            return 'no'
        return predicted_category
        
    
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
        correct_guesses = 0
        total_tests = 0
        
        testing_sentences_dict = loadTrainingSentences(dev_file_path)
        for true_category, testing_sentences_list in testing_sentences_dict.items():
            for sentence in testing_sentences_list:
                predicted_category = self.getCategory(sentence, training_file_path)
                if predicted_category == true_category:
                    correct_guesses+=1
                total_tests+=1
        return correct_guesses/total_tests

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


        if "holoemitter" in command:
            slots["holoEmit"] = True
        if "logic display" in command:
            slots["logDisp"] = True
        if any(x in command for x in ['increase', 'add']):
            slots["add"] = True
        if any(x in command for x in ['decrease', 'subtract']):
            slots["sub"] = True
        if any(x in command for x in ['maximum', "turn on", "lights on"]):
            slots["on"] = True
        if any(x in command for x in ['minimum', "turn off", "lights off", 'lower']):
            slots["off"] = True

        if any(x in command for x in ['light', "led"]):
            if any(x in command for x in ['lights', "leds"]):
                slots["lights"] = ['front','back']
            elif 'front' in command:
                slots["lights"] = ['front']
            elif 'back' in command:
                slots["lights"] = ['back']
        
        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        
        ### YOUR CODE HERE ###
        direction_map = [(['north','straight'],'forward'),
                         (['south'],'back'),
                         (['east'],'right'),
                         (['west'],'left')]

        temp_direction_map = {}
        
        for alternative_commands, direction in direction_map:
            for alternative_command in alternative_commands:
                temp_direction_map[alternative_command] = direction
            temp_direction_map[direction] = direction

        
        direction_map = temp_direction_map
        #print(direction_map)
        command = command.lower()

        if any(x in command for x in ['increase', 'add']):
            slots["increase"] = True
        if any(x in command for x in ['decrease', 'subtract', 'lower']):
            slots["decrease"] = True

        possible_direction_tokens = tokenize(command)
        #print(possible_direction_tokens)
        direction_list = []
        for possible_direction_token in possible_direction_tokens:
            if possible_direction_token in direction_map:
                direction = direction_map[possible_direction_token]
                direction_list.append(direction)
        
        slots['directions'] = direction_list

        return slots


############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 30

feedback_question_2 = """
Implementing the k-nearest neighbor and coming up with original sentences were the hardest parts for me.
"""

feedback_question_3 = """
I liked everything.
"""
"""
g = cosineSimilarity(np.array([10, 1]), np.array([1, 10])) 

v1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
v2 = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])


g = cosineSimilarity(v1, v2)

g = tokenize("'Medium-rare,' she said.")
g = tokenize("  This is an example.  ")

X = WordEmbeddings("GoogleNews-vectors-negative300.magnitude") 

svec1 = X.calcSentenceEmbeddingBaseline("drive forward")
svec2 = X.calcSentenceEmbeddingBaseline("roll ahead")
svec3 = X.calcSentenceEmbeddingBaseline("set your lights to purple")
svec4 = X.calcSentenceEmbeddingBaseline("turn your lights to be blue")
g = cosineSimilarity(svec1, svec2)
print(g)
g = cosineSimilarity(svec1, svec3)
print(g)
g = cosineSimilarity(svec1, svec4)
print(g)
g = cosineSimilarity(svec2, svec3)
print(g)
g = cosineSimilarity(svec2, svec4)
print(g)
g = cosineSimilarity(svec3, svec4)
print(g)
"""

"""
trainingSentences = loadTrainingSentences("data/r2d2TrainingSentences.txt")
X = WordEmbeddings("GoogleNews-vectors-negative300.magnitude")  # Change this to where you downloaded the file.
#sentenceEmbeddings, indexToSentence = X.sentenceToEmbeddings(trainingSentences)
#g = sentenceEmbeddings[14:]
#print(g)
#g = indexToSentence[14]
#print(g)

#sentenceEmbeddings, k = X.sentenceToEmbeddings(loadTrainingSentences("data/r2d2TrainingSentences.txt"))
#print(X.closestSentence("Lights on.", sentenceEmbeddings))

g = X.getCategory("Turn your lights green.", "data/r2d2TrainingSentences.txt")
print(g)

g = X.getCategory("Drive forward for two feet.", "data/r2d2TrainingSentences.txt")
print(g)

g = X.getCategory("Do not laugh at me.", "data/r2d2TrainingSentences.txt")
print(g)

g = X.accuracy("data/r2d2TrainingSentences.txt", "data/r2d2DevelopmentSentences.txt")
print(g)

g = X.lightParser("Set your lights to maximum")
print(g)

g =  X.lightParser("Increase the red RGB value of your front light by 50.")
print(g)

g =  X.drivingParser("Increase your speed!")
print(g)
g = X.drivingParser("Go forward, left, right, and then East.")
print(g)
"""
