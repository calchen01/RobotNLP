############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Marty Rubin and Salomon Serfati"

############################################################
# Imports
############################################################

from pymagnitude import *
import numpy as np
import string
import re
import heapq

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
"Go forward for 2 meters, then turn right and then left.",
"North is at heading 20 degrees.",
"Go North and then south.",
"reverse yourself.",
" turn in the opposite direction",
"Go South-by-southeast and then turn around",
"Run away young bull",
"spin around 360 degrees",
"dance by spinning around",
"Turn to face West.",
"Start rolling backward.",
"Increase your speed by 60%." ]

my_light_sentences = [
"Change the intensity on the holoemitter to max.",
"Turn on the holoemitter.",
"Blink your logic display various times ",
"Change the back LED to blue.",
"Turn your back light blue.",
"Dim your lights holoemitter.",
"Turn on all your lights.",
"Lights on.",
"Set the RGB values on your lights to be 255, 255,0.",
"Add 110 to the red value of your front LED.",
"Increase the blue value of your back LED by 2%.",
"Display the following colors for 2 seconds each: red, orange, blue, green, blue, purple.",
"Change the color on both LEDs to be blue.", ]

my_head_sentences = [
"Turn your head to face forward.",
"Turn your head to face backward.", 
"Turn your head forward.",
"Turn your head backward.",
"look to the right",
"look to the left",
"turn your head around",
"turn your head North",
"Turn your head South",
"Look behind you.",]

my_state_sentences = [
"What color is your front light?",
"Tell me what color your front light is set to.",
"Is your logic display on?",
"What is your stance?",
"What is your orientation?",
"What direction are you facing?",
"Are you standing on 2 feet or 3?",
"What is your current heading?",
"How much battery do you have left?",
"What is your battery status?",
"Are you driving right now?",
"How fast are you going?",
"What is your current speed?",
"Is your back light red?",
"Are you awake?", ]

my_connection_sentences = [
"Connect D2-55A2 to the server",
"Are there any other droids nearby?",
"Disconnect.",
"Disconnect from the server.",
"make sure server is disconnected",
"find nearby droids?",
"Connect to server",
"Connect",
"Connect and then disconnect",
"connect to the server now",
]

my_stance_sentences = [
"Set your stance to be biped.",
"Put down your third wheel.",
"Stand on your tiptoes.",
"Get off tiptoes",
"raise your third whell",
"dont be biped."
"raise second wheel",
"raise first wheel",
"put down second wheel", 
"put down first wheel",
"get off of your tiptoes",
"lower your stance",
]

my_animation_sentences = [
"Fall over",
"Scream",
"Make some noise",
"Laugh",
"Play an alarm",
"Dance for me", 
"play dead", 
"it is my birthday", 
"slide", 
"go crazy"]

my_grid_sentences = [
"You are on a 4 by 5 grid.",
"Each square is 1 foot large.",
"You are at position (0,0).",
"Go to position (3,3).",
"There is an obstacle at position 2,1.",
"There is a chair at position 3,3",
"Go to the left of the chair.",
"It’s not possible to go from 2,2 to 2,3.",
"Go to the right of the chair.",
"It’s is possible to go from 2,2 to 2,3.",]


############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    tokens = [] 
    curr_word = ""
    sentence = sentence.lower()
    for c in sentence:
        if c in string.punctuation: 
            if len(curr_word) > 0: 
                tokens.append(curr_word);
                curr_word = ""
        elif(c.isspace()):
            if len(curr_word) > 0: 
                tokens.append(curr_word);
                curr_word = ""
        else:
            curr_word += c
    
    if len(curr_word) > 0: 
                tokens.append(curr_word);
                curr_word = ""
    return tokens

# print(tokenize("  This is an example.  "))
# print(tokenize("'Medium-rare,' she said."))

def cosineSimilarity(vector1, vector2):

    """
    dot_product = 0
    vector1_sum = 0; 
    vector2_sum = 0;
    for i in range(len(vector1)): 
        dot_product += (vector1[i] * vector2[i])
        vector1_sum += (vector1[i] ** 2) 
        vector2_sum += (vector2[i] ** 2) 
    
    vector1_length =  vector1_sum ** (1/2)
    vector2_length =  vector2_sum ** (1/2)

    return dot_product / (vector1_length * vector2_length)"""
    
    return np.dot(vector1, vector2) / np.linalg.norm(vector1) / np.linalg.norm(vector2)

    
    

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path) 

    def calcSentenceEmbeddingBaseline(self, sentence):
        vectors = []
        for word in tokenize(sentence): 
            vectors.append(self.vectors.query(word))
        
        if len(vectors) == 0:
            v = self.vectors.query("cat")
            return np.zeros(len(v))
        
        component_wise_addition_vector = [sum(v[i] for v in vectors) for i in range(len(vectors[0]))]
        
        return component_wise_addition_vector

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
        
        sentenceEmbeddings = []
        indexToSentence = {}
        i = 0
        for category, sentences in commandTypeToSentences.items(): 
            for sentence in sentences: 
                sentenceEmbeddings.append(self.calcSentenceEmbeddingBaseline(sentence))
                indexToSentence[i] = (sentence, category)
                i += 1
        sentenceEmbeddings = np.array(sentenceEmbeddings)
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
        max_similarity = -float("inf")
        max_index = -1;
        sentence = self.calcSentenceEmbeddingBaseline(sentence)
        
        for i, sentence2 in enumerate(sentenceEmbeddings):    
            cos_similarity = cosineSimilarity(sentence, sentence2)
            if cos_similarity > max_similarity: 
                max_index = i
                max_similarity = cos_similarity
        
        return max_index
    
    def _k_nearest(self, sentence, sentenceEmbeddings, k):
        sentence = self.calcSentenceEmbeddingBaseline(sentence)
        return heapq.nlargest(
            k,
            range(len(sentenceEmbeddings)),
            key=lambda i: cosineSimilarity(sentence, sentenceEmbeddings[i])
            )

    def _remove_stop_words(self, sentence):
        stops = r"\b(the|at|is|in|of|on|to|a|an|I|are|by|be|do|you|too|then)\b"
        return re.sub(stops, "", sentence)


    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        k = 5
        sentence = self._remove_stop_words(sentence)
        trainingSentences = loadTrainingSentences(file_path)
        for key in trainingSentences:
            trainingSentences[key] = [self._remove_stop_words(s) for s in trainingSentences[key]]
        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(trainingSentences)
        k_nearest = self._k_nearest(sentence, sentenceEmbeddings, k)
        scores = {}
        for index in k_nearest:
            category = indexToSentence[index][1]
            scores[category] = scores.get(category, 0) + 1 / len(trainingSentences[category])
        
        output = max(scores, key=lambda score: scores[score])

        if cosineSimilarity(self.calcSentenceEmbeddingBaseline(sentence), sentenceEmbeddings[k_nearest[0]]) > .8:
            return indexToSentence[k_nearest[0]][1]

        if cosineSimilarity(self.calcSentenceEmbeddingBaseline(sentence), sentenceEmbeddings[k_nearest[0]]) < .55:
            return "no"

        """
        if output != correct_category:
            print("Correct Answer: " + correct_category)
            sentence = self.calcSentenceEmbeddingBaseline(sentence)
            for i in k_nearest:
                category = indexToSentence[i][1]
                print("\t", category, cosineSimilarity(sentence, sentenceEmbeddings[i]))
          """
        
        return output

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

        dev_sentence_map = loadTrainingSentences(dev_file_path)

        total = 0
        correct = 0

        for correct_category, test_sentences in dev_sentence_map.items():
            for sentence in test_sentences:
                if correct_category == self.getCategory(sentence, training_file_path):
                    correct += 1
                total += 1
        return correct / total

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
