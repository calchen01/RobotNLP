############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Zhanpeng Wang"

############################################################
# Imports
############################################################

from pymagnitude import *
import numpy as np
import string
import re
import collections

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

my_driving_sentences = [
    "Turn to face West.",
    "Reset your orientation.",
    "Move forward for 3 seconds.",
    "Move in circle.",
    "It is a trap! Run!",
    "Move northward.",
    "Decrease your speed by 60%.",
    "Escape!",
    "Mission aborted! Run away!",
    "Turn clockwise by 180 degrees."
]
assert len(my_driving_sentences) == 10

my_light_sentences = [
    "Change your front light to blue.",
    "Turn your front LED green.",
    "Turn off both of your lights.",
    "Maximize your holoemitter intensity.",
    "Set the RBGs on your lights to be 0, 255, 0.",
    "Add 50 to your G value of your back LED.",
    "Flash your logic display.",
    "Blick both of your light.",
    "Brighten your lights holoemitter.",
    "Show the following colors for 3 seocnds in your lights each: purple, blue, green, yellow, orange, red."
]
assert len(my_light_sentences) == 10

my_head_sentences = [
    "Turn your head backward.",
    "Eyes at front!",
    "Look at your left.",
    "Rotate your head for a full circle.",
    "Spin your head 60 degrees.",
    "Rotate your head counterclockwise by 30 degrees.",
    "Turn your head clockwise by 45 degrees.",
    "Spin your head to face forward.",
    "Revolve your head for an angle bigger than 90 degrees.",
    "Swing around your head 180 degrees"
]
assert len(my_head_sentences) == 10

my_state_sentences = [
    "What is your current stance?",
    "Status report, please.",
    "How fast is your current move?",
    "Are you in driving mode?",
    "What is your front LED color?",
    "How much battery power do you have now?",
    "What is your current facing direction?",
    "Are your connected?",
    "Is your back light on?",
    "Are you waddling?"
]
assert len(my_state_sentences) == 10

my_connection_sentences = [
    "Connect D2-33CS to the server.",
    "Are there any droids in your scanning perimeter?",
    "Disconnect yourself.",
    "Exit the server.",
    "Scan your perimeter.",
    "Connect D2-55A2.",
    "Get rid of your connection.",
    "Say goodbye to the server.",
    "Bye bye!",
    "Have a great day!"
]
assert len(my_connection_sentences) == 10

my_stance_sentences = [
    "Tiptoes stand.",
    "Change your stance to be biped.",
    "Stand with your two legs.",
    "Start waddling.",
    "Put down your third wheel.",
    "Don't waddle.",
    "Don't walk like a duck.",
    "Imitate a duck's walking.",
    "Set your waddle to be false.",
    "Become a duck."
]
assert len(my_stance_sentences) == 10

my_animation_sentences = [
    "Fall!",
    "Scream as loud as you can!",
    "Play your alarm.",
    "Laugh out loud",
    "LOL",
    "Stay on the ground!",
    "Dance!",
    "Create some noise.",
    "Show off your dance.",
    "Yell it out!"
]
assert len(my_animation_sentences) == 10

my_grid_sentences = [
    "You are placed in a 5-by-5 grid.",
    "Each square is one foot in length.",
    "Your current position is (5,5).",
    "Move to position (3,3).",
    "There is an obstacle at position (2,2).",
    "There is no way to move from (3,3) to (2,2).",
    "Move to your goal (0,0) with shortest path",
    "Go to the right of the obstacle.",
    "Move along the path (0,0), (0,1), (0,2).",
    "Go east."
]
assert len(my_grid_sentences) == 10

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    # remove control characters \n\r\t etc., up to space (ASCII number 32)
    text = sentence.strip().translate(dict.fromkeys(range(32)))
    text = text.lower()
    tokens = []
    temp = []
    for i in range(len(text)):
        char = text[i]
        if char in string.punctuation or char == ' ':
            if len(temp) != 0:
                accumulated_string = "".join(temp)
                tokens.append(accumulated_string)
                temp = []
            if char != ' ':
                tokens.append(char)
        else:
            temp.append(char)
        if i == len(text) - 1:
            if len(temp) != 0:
                accumulated_string = "".join(temp)
                tokens.append(accumulated_string)
    return [token for token in tokens if token not in string.punctuation]

def cosineSimilarity(vector1, vector2):
    dot_prdocut = np.dot(vector1, vector2)
    v1_length = np.sqrt(np.sum(np.square(vector1)))
    v2_length = np.sqrt(np.sum(np.square(vector2)))
    return dot_prdocut / (v1_length * v2_length)

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)
        if len(tokens) == 0:
            return np.array([0] * self.vectors.dim)
        sentence_vector = self.vectors.query(tokens[0])
        for i in range(1, len(tokens)):
            sentence_vector = np.add(self.vectors.query(tokens[i]), sentence_vector)
        return sentence_vector

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
        sentence_embeddings = []
        index_to_sentence = {}
        if len(commandTypeToSentences) != 0:
            index = 0
            for command_type, sentences in commandTypeToSentences.items():
                for sentence in sentences:
                    sentence_embeddings.append(self.calcSentenceEmbeddingBaseline(sentence))
                    index_to_sentence[index] = (sentence, command_type)
                    index += 1
        return np.array(sentence_embeddings), index_to_sentence

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
        max_idx = 0
        maximum = -float('inf')
        input_sentence_embedding = self.calcSentenceEmbeddingBaseline(sentence)
        for i in range(len(sentenceEmbeddings)):
            temp = cosineSimilarity(input_sentence_embedding, sentenceEmbeddings[i])
            if temp > maximum:
                max_idx = i
                maximum = temp
        return max_idx

    def knn_classifier(self, k, training_data, labels, query):
        """
        Returns the category of the query (sentence embeddings)

        Inputs:
            k: the chosen number of neighbors
            training_data: a numpy array with dimension num_sentence x 300,
                           each row represents a sentence embeddings of a sentence.
            labels: the correct labels of the training data
            query: the sentence embeddings of a sentence
        
        Returns:
            the predicted category of this query
        """
        neighbor_similarity_and_indices = []

        # for each sample in the training data
        for index, sample in enumerate(training_data):
            
            # compute the cosine similarity between sample and query
            cos_sim = cosineSimilarity(sample, query)

            # append (cos_sim, index) to neighbor_similarity_and_indices
            neighbor_similarity_and_indices.append((cos_sim, index))
        
        # sort the neighbor_similarity_and_indices list in descending order
        # because we use cosine similarity
        neighbor_similarity_and_indices.sort(key=lambda x: x[0], reverse=True)

        # get the k nearest neighbors
        k_nearest_neighbors_and_indices = neighbor_similarity_and_indices[:k]
        
        # get the labels for k nearest neighbors
        k_nearest_neighbors_label = [labels[i] for _, i in k_nearest_neighbors_and_indices]

        # find the mode
        return collections.Counter(k_nearest_neighbors_label).most_common(1)[0][0]

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
        training_sentences_for_category_no = [
            "Do not laugh at me!",
            "What do you eat for dinner?",
            "Don't curse me.",
            "Robot is dumb!",
            "I like you very much!",
            "Howdy!",
            "Peter loves cooking.",
            "I came, I saw, I conquered.",
            "Rome is not built in one day.",
            "Cheese cake is too good."
        ]
        training_sentences["no"] = training_sentences_for_category_no
        training_data_for_knn, idx_to_sentence = self.sentenceToEmbeddings(training_sentences)
        labels = [idx_to_sentence[i][1] for i in range(training_data_for_knn.shape[0])]
        sentence_embedding = self.calcSentenceEmbeddingBaseline(sentence)
        return self.knn_classifier(2, training_data_for_knn, labels, sentence_embedding)

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
        dev_sentences = loadTrainingSentences(dev_file_path)
        c = 0
        s = 0
        for command_type, sentences in dev_sentences.items():
            for sentence in sentences:
                y_pred = self.getCategory(sentence, training_file_path)
                if y_pred == command_type:
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

        # holoemitter
        if re.search(r"\b(holoemitter|holo|emit)\b", command, re.I):
            slots["holoEmit"] = True
        
        #logic display
        if re.search(r"\b(logic|display)\b", command, re.I):
            slots["logDisp"] = True

        # lights
        if re.search(r"\b(lights|light)\b", command, re.I):
            if re.search(r"\b(front)\b", command, re.I):
                slots["lights"].append("front")
            elif re.search(r"\b(back|hind|rear)\b", command, re.I):
                slots["lights"].append("back")
            else:
                slots["lights"].append("front")
                slots["lights"].append("back")
        
        if re.search(r"\b(rgb|red|green|blue)\b", command, re.I):
            # add
            if re.search(r"\b(increase|increment|add|boost|enhance|raise|intensify)\b", command, re.I):
                slots["add"] = True
            # sub
            if re.search(r"\b(subtract|decrease|lower|reduce|drop|cut)\b", command, re.I):
                slots["sub"] = True

        if re.search(r"\b(max|maximum)\b", command, re.I):
            slots["on"] = True
        
        if re.search(r"\b(min|minimum)\b", command, re.I):
            slots["off"] = True

        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}
        
        tokens = tokenize(command)
        for token in tokens:
            # increase
            if re.search(r"\b(increase|increment|add|boost|enhance|raise|intensify)\b", token, re.I):
                slots["increase"] = True
            
            # decrease
            if re.search(r"\b(subtract|decrease|lower|reduce|drop|cut)\b", token, re.I):
                slots["decrease"] = True
            
            # forward
            if re.search(r"\b(forward|north|straight|ahead|onward|forth)\b", token, re.I):
                slots["directions"].append("forward")

            # back
            if re.search(r"\b(back|hind|rear|behind|south)\b", token, re.I):
                slots["directions"].append("back")

            # left
            if re.search(r"\b(left|west)\b", token, re.I):
                slots["directions"].append("left")

            # right
            if re.search(r"\b(right|east)\b", token, re.I):
                slots["directions"].append("right")

        return slots


############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 6

feedback_question_2 = """
The most challenging aspects of this assignment:
    1. Implement the getCategory function to have higher accuracy of
       predicting the correct category for a command sentence

The significant stumbling block: None for now
"""

feedback_question_3 = """
The aspects of this assignment I liked:
    1. Using voice to command R2D2 is very fun!

Something I would have changed: We can probably use RNN to perform 
the slot filling task instead of using re or word2vec.
"""

