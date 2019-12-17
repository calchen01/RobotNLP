############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Aaron Diamond-Reivich"

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
    "Drive North",
    "Slowly go forward",
    "Walk Left",
    "Stop Moving",
    "Make a U Turn",
    "Move faster",
    "decrease your speed by 30%",
    "Turn Right",
    "Go as fast as you can forward",
    "Run!"
]

my_light_sentences = [
    "Turn lights off",
    "Turn your lights as bright as possible!",
    "flash your lights one at a time",
    "Turn your back LED to red",
    "Increase green value of back LED",
    "Set the RBG value of your back LED to 200,200,200",
    "Change intensity of holoemitter to 50%",
    "dim your holoemitter",
    "change all of your lights to pink",
    "flash all of your lights red"
]

my_head_sentences = [
    "Look behind you",
    "keep facing forard",
    "turn your head to the left",
    "Spin your head all the way around",
    "turn your head to the right",
    "rotate head 90 degrees west",
    "rotate head 270 degrees west",
    "rotate head 90 degrees east",
    "rotate head 360 degrees",
    "look to the left"
]

my_state_sentences = [
    "How much battery do you have left?",
    "are you on?",
    "what color are you?",
    "what direction are you facing?",
    "what is your orientation?",
    "what is your speed?",
    "Are you sleeping?",
    "Are you off?",
    "What color is your back LED light?",
    "Are you moving to the north?"
]

my_connection_sentences = [
    "disconnect",
    "Are there any connections available?",
    "connect to server",
    "are there any robots nearby?",
    "connect to nearby robot",
    "connect to second closest robot",
    "Are any other droids connected?",
    "connect and then disconnect",
    "try to connect to the server",
    "exit the server"
]

my_stance_sentences = [
    "use only two wheels",
    "use all of your wheels",
    "try to use only two wheels",
    "waddle away",
    "stand on your toes",
    "waddle with two wheels",
    "put all of your wheels on the ground",
    "try to use only one wheel",
    "be biped",
    "be triped"
]

my_animation_sentences = [
    "lay on the ground",
    "yell",
    "make your loudest noise",
    "set off your alarm",
    "cry",
    "laugh",
    "exclaim",
    "get up",
    "play a quiet alarm",
    "play dead"
]

my_grid_sentences = [
    "You are on a 10 by 10 grid",
    "each square is 1 foot wide",
    "move to position (3,2)",
    "you are at position (5,5)",
    "you are at position (0,0)",
    "there is an obstacle at (5,7)",
    "there is another robot at 2,2",
    "you cannot move from 7,7 to 7,8 or 7,6",
    "you can't waddle from 1,1 to 2,1",
    "you can now move from 1,1 to 2,1"   
]




############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    tokens = []
    current_word = ""
    for char in sentence:
        if char.isspace():
            if current_word != "":
                current_word = current_word.lower()
                tokens.append(current_word)
                current_word = ""
        elif char in string.punctuation or char == "’":
            if current_word != "":
                current_word = current_word.lower()
                tokens.append(current_word)
                current_word = ""
        else:
            current_word += char
    if current_word != "":
        current_word = current_word.lower()
        tokens.append(current_word)
    return tokens

def cosineSimilarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    nomalization_vector1 = np.linalg.norm(vector1)
    nomalization_vector2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (nomalization_vector1 * nomalization_vector2)
    return cosine_similarity


class WordEmbeddings:

    def __init__(self, file_path):
        #self.vectors = Magnitude(file_path)
        self.vectors = Magnitude(file_path)
        self.vector_length = len(self.vectors.query('cat'))


    def calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)
        vector = []
        if len(tokens) == 0: 
            for i in range (0, self.vector_length):
                vector.append(0)
        else:
            for token in tokens:
                vector = self.addVectors(vector, self.vectors.query(token))
        return vector

    def addVectors(self,vec1, vec2):
        for i in range (0, self.vector_length):
            if len(vec1) <= i:
                vec1.append(vec2[i])
            else:
                vec1[i] = vec1[i] + vec2[i]
        return vec1


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
        count = 0

        for key in commandTypeToSentences.keys():
            for sentence in commandTypeToSentences[key]:
                current_sentence_embedding = self.calcSentenceEmbeddingBaseline(sentence)
                sentenceEmbeddings.append(current_sentence_embedding)
                indexToSentence[count] = ((sentence, key))
                count = count + 1

        sentenceEmbeddings = np.array(sentenceEmbeddings)
        sentenceEmbeddings = np.reshape(sentenceEmbeddings, (count, 300))
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
        vectorized_sentence = self.calcSentenceEmbeddingBaseline(sentence)
        closest_score = -10.0
        closet_sentence_index = -1
        m, n = sentenceEmbeddings.shape
        for i in range (m):
            cosine_sim = cosineSimilarity(vectorized_sentence, sentenceEmbeddings[i])
            if cosine_sim > closest_score:
                closest_score = cosine_sim
                closet_sentence_index = i
        return closet_sentence_index

    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        k = 8
        no_threshold = 2

        trainingSentences = loadTrainingSentences(file_path)
        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(trainingSentences)
        sentence_vector = self.calcSentenceEmbeddingBaseline(sentence)
        
        sims_and_idxs = []

        for idx, example in enumerate(sentenceEmbeddings):
            sim = cosineSimilarity(sentence_vector, example)
            sims_and_idxs.append((sim, idx))

        sorted_sims_and_idxs = sorted(sims_and_idxs)
        sorted_sims_and_idxs.reverse()
        
        k_nearest_sims_and_idxs = sorted_sims_and_idxs[:k]
        print("HIGHEST K: ", k_nearest_sims_and_idxs)
        
        highest_similarity_score = sorted_sims_and_idxs[0][0]
        highest_similarity_label = indexToSentence[sorted_sims_and_idxs[0][1]][1]

        second_highest_similarity_score = sorted_sims_and_idxs[1][0]
        second_highest_similarity_label = indexToSentence[sorted_sims_and_idxs[1][1]][1]
        print("HIGHEST: ", highest_similarity_score, highest_similarity_label)
        print("SECOND HIGdHEST: ", second_highest_similarity_score, second_highest_similarity_label)


        k_nearest_labels = {}
        for (_, idx) in k_nearest_sims_and_idxs:
            (_, label) = indexToSentence[idx]
            if label in k_nearest_labels:
                 k_nearest_labels[label] += 1
            else:
                k_nearest_labels[label] = 1
        print(k_nearest_labels)

        most_common_label = ""
        highest_occurence = -1
        second_highest_occurence = -1
        for key in k_nearest_labels.keys():
            if k_nearest_labels[key] > highest_occurence:
                most_common_label = key
                highest_occurence = k_nearest_labels[key]
        print("MOST COMMON: ", most_common_label)
        print("HOW MANY TIMES: ", highest_occurence)
        vs = list(k_nearest_labels.values())
        vs.sort(reverse=True)
        print("all counts: ", vs)
        print("HOW MANY TIMES (SECOND)", vs[1])
        print("difference: ",  (highest_similarity_score - second_highest_similarity_score))

        if (highest_similarity_score < .7): 
            return "no"
        if (highest_similarity_score > .8):
            return highest_similarity_label
     #   elif (highest_similarity_label == second_highest_similarity_label == 'state'):
      #      print("FIRST TWO HIGHEST ARE THE SAME")
       #     return highest_similarity_label
        
        elif (highest_similarity_label == 'state' and (highest_similarity_score - second_highest_similarity_score) < .1):
            print("TOO MANY STATES")
            return "no"
        elif (highest_similarity_score - second_highest_similarity_score > .15):
            print("HUGE DIFFERENCE FOR FIRST ONE")
            return highest_similarity_label
        elif (no_threshold >= highest_occurence):
            return "no"
       # elif (len(vs) > 0):
        #    if (vs[0] == vs[1]):
       #         print("SAME HIGHEST COUNTS")
        #        return "no"
      #  elif (highest_similarity_label == second_highest_similarity_label):
     #       print("FIRST TWO HIGHEST ARE THE SAME")
      #      return highest_similarity_label
        elif (k < highest_occurence):
            return most_common_label
        # elif(highest_similarity_label == 'state' and (highest_similarity_score - second_highest_similarity_score) > .05):
         #    print("TOO MANY STATES")
        #    return "no"
        return most_common_label

        # if highest_similarity_score > .85:
        #     #print("OPTION 1")
        #     return (highest_similarity_label, highest_similarity_score, highest_similarity_label)
        # if highest_occurence >= k:
        #     #print("OPTION 2")
        #     return (most_common_label, highest_similarity_score, highest_similarity_label)
        # if highest_occurence <= no_threshold and highest_similarity_score < .6:
        #     #print("OPTION 3")
        #     return ('no', highest_similarity_score, highest_similarity_label)
        # if highest_occurence < no_threshold:
        #     #print("OPTION 4")
        #     return ('no', highest_similarity_score, highest_similarity_label)
        # else:
        #     #print("OPTION 5")
        #     return (most_common_label, highest_similarity_score, highest_similarity_label)



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
        dev_labels_and_sentences = loadTrainingSentences(dev_file_path)

        correct = 0
        total = 0

        words_to_prune = ['your', "i", "the", "on", "it", "is"]

        for label in dev_labels_and_sentences.keys():
            for sentence in dev_labels_and_sentences[label]:
                prunedsentence = sentence
                for pruned_word in words_to_prune:
                    if pruned_word in prunedsentence:
                        prunedsentence.replace(pruned_word, '')
               # (calculated_category, highest_sim_score, highet_sim_label) = self.getCategory(prunedsentence, training_file_path)
                calculated_category = self.getCategory(prunedsentence, training_file_path)
                print("prediction: ", calculated_category)
                print("actual label: ", label)
                if calculated_category == label:
                    print('we got it right')
                    correct += 1
                else:
                    print("this case is wrong") 
                    #print("Y    Correct Label: " + label + " High Label: " + highet_sim_label)
                    #print(highest_sim_score)
                #else:
                    #print("N    Correct Label: " + label + " Guessed Label: " + calculated_category + " High Label: " + highet_sim_label)
                    #print(highest_sim_score)

                print('')
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
        threshold = .85

        lights = ('lights', ['front', 'back', 'lights'])
        logDisp = ('logDisp', ['logic', 'logDisp'])
        holoEmit = ('holoEmit', ['holoemitter'])
        add = ('add', ['increase', 'add'])
        sub = ('sub', ['decrease', 'reduce', 'substract', '0'])
        off = ('off', ['off'])
        on = ('on', ['maximum'])

        key_words_lists = [lights, logDisp, holoEmit, add, sub, off, on]

        label_embeddings = []

        for category, key_list in key_words_lists:
            #print(f"{category} : {key_list}")
            for word in key_list:
                #print(word)
                word_embedding = self.calcSentenceEmbeddingBaseline(word)
                label_embeddings.append((category, word_embedding))

        command_tokens = tokenize(command)
        #print(len(label_embeddings))

        front_embedding = self.calcSentenceEmbeddingBaseline('front')
        back_embedding = self.calcSentenceEmbeddingBaseline('back')

        for token in command_tokens:
            token_embedding = self.calcSentenceEmbeddingBaseline(token)
            for (label, embedding) in label_embeddings:
                #print(label)
                #print(embedding)
                #print(token_embedding)
                sim = cosineSimilarity(embedding, token_embedding)
                if sim > threshold:
                    if label == 'lights':
                        if token == 'lights':
                            slots['lights'].append('front')
                            slots['lights'].append('back')
                        elif cosineSimilarity(front_embedding, token_embedding) > cosineSimilarity(back_embedding, token_embedding):
                            #print("adding front")
                            slots['lights'].append('front')
                        else:
                            #print("adding back")
                            slots['lights'].append('back')
                    else:
                        #print("setting " + label + " to true")
                        slots[label] = True
        #print(slots)
        if slots["holoEmit"] or slots["logDisp"]:
            slots["on"] = False
            slots["off"] = False
        
        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left".
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###

        tokens = tokenize(command)
        for token in tokens:
            if token == 'increase':
                slots["increase"] = True
            elif token == 'decrease':
                slots["decrease"] = True
            elif token == 'forward' or token == 'north':
                slots["directions"].append('forward')
            elif token == 'backward':
                slots["directions"].append('backward')
            elif token == 'right' or token == 'east':
                slots["directions"].append('right')
            elif token == 'left' or token == 'west':
                slots["directions"].append('left')
            elif token == 'back' or token == 'south':
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
