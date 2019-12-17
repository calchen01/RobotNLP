############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Shuqi Zhang, Han Chen"

############################################################
# Imports
############################################################

from pymagnitude import *
import numpy as np
import string
import re

from sklearn.neighbors import NearestNeighbors
############################################################
# Helper Functions
############################################################

def loadTrainingSentences(file_path):
    commandTypeToSentences = {}

    with open(file_path, 'r', encoding = "utf-8") as fin: # add encoding part
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
"Go South.",
"Go South and then east.",
"Run!",
"speed zero"
"Set speed to be 70%",
"Turn 180 degrees",
"Start rolling back.",
"Rolling forward",
"Increase speed by 80%.",
"Turn to your left.",
"Turn to heading your north.",
"Go forward for 3 feet and turn left.",
"North is at heading 90 degrees.",
"Reset your heading to 90",
"Look at South.",
"Stop moving.",
"Rolling for 3 seconds",
"Stop rolling",
]

my_light_sentences = [
"Change the intensity on the holoemitter to maximum.",
"Turn off the holoemitter.",
"Blink your logic display.",
"Change the back LED to green.",
"Change the front LED to red.",
"Turn on all your lights.",
"Lights out.",
"Set the RGB values on your lights to be 255,0,0.",
"Add 100 to the red value of your front LED.",
"Increase the blue value of your back LED by 50%.",
"Turn your back light green.",
"Dim your lights holoemitter.",
"Turn off all your lights.",
"Display the following colors for 2 seconds each: red, orange, yellow, green, blue, purple.",
"Change the color on both LEDs to be green.", ]

my_head_sentences = [
"Turn your head to face forward.",
"Look behind you.",
"Rotate head by 90 degree.",
"Face South.",
"Look North.",
"Turn your head 50 degree.",
"Turn your head by 80 degree.",
"Reset.",
"Turn around your head.",
"Rotate to East.",
 ]

my_state_sentences = [
"How much battery do you have left?",
"Remain battery.",
"Current Speed.",
"Is your back light red?",
"Back light color",
"What color is your front light?",
"Tell me what color your front light is set to.",
"Is your logic display on?",
"What is your stance?",
"What is your orientation?",
"Your battery status",
"Are you driving?",
"How fast are you going?",
"Orientation",
"What direction are you facing?",
"Current direction.",
"Are you standing on 2 feet or 3?",
"What is your current heading?",
"Are you awake?", ]

my_connection_sentences = [
"Connect to the server",
"Connect",
"Are there any other droids nearby?",
"Disconnect.",
"Disconnect from the server.", 
"Connect to nearby droid",
"Disconnect from nearby droid",
"Scan nearby devices",
"Scan",
"Check nearby droid",
]

my_stance_sentences = [
"Put down first wheel.",
"Put on first wheel",
"Stand",
"Stand solid",
"Put down second wheel",
"Set your stance to be biped.",
"Put down your third wheel.",
"Stand on your tiptoes.",
"keep down",
"Stand up",
]

my_animation_sentences = [
"Fall",
"Where are you",
"Any sound",
"Make noise",
"Play music",
"Fall over",
"Scream",
"Make some noise",
"Laugh",
"Play an alarm",]

my_grid_sentences = [
"You are on a 5 by 5 grid.",
"Each square is 1 foot large.",
"You are at position (0,0).",
"Go to position (3,3).",
"Go to (1,2)",
"Object at (2,2)",
"Cannot go to 2,1",
"There is an obstacle at position 2,1.",
"There is a gift at position 3,3",
"Go get the gift",
"It’s not possible to go from 2,2 to 2,3.", 
]

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    st = sentence.split()
    result = []
    for s in st:
        if re.match(r'^\w+$', s):
            result.append(s.lower())
        else:
            a = re.split("[" + string.punctuation + "]+", s)
            for element in a:
                if element not in string.punctuation:
                    result.append(element.lower())
    return result


def cosineSimilarity(vector1, vector2):
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path) 


    def calcSentenceEmbeddingBaseline(self, sentence):
        token = tokenize(sentence)
        result = np.zeros(300)
        for word in token:
                result += self.vectors.query(word)
        return result


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
        values = 0
        for key in commandTypeToSentences.keys():
            for s in commandTypeToSentences[key]:
                values+=1

        indexToSentence = {}
        count = 0
        sentenceEmbed = np.zeros((values,300))
        for key in commandTypeToSentences.keys():
            for s in commandTypeToSentences[key]:
                se = self.calcSentenceEmbeddingBaseline(s)
                sentenceEmbed[count] = se
                indexToSentence[count] = (s,key)
                count+=1
        return (sentenceEmbed,indexToSentence)

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
        result = 0
        cosineSimMax = -999
        vect = self.calcSentenceEmbeddingBaseline(sentence)
        for i in range(len(sentenceEmbeddings)):
            vector = sentenceEmbeddings[i]
            cur_sim = cosineSimilarity(vector,vect)
            if cur_sim > cosineSimMax:
                cosineSimMax = cur_sim
                result = i
        # print("cosine similartiy: ", cosineSimMax)
        return result

    def kClosestSentence(self, sentence, sentenceEmbeddings): # return k indexes
        result = []
        cosineSimMax = -999
        cosSimTable = []
        lookup = {} # key: similarity      value: index
        vect = self.calcSentenceEmbeddingBaseline(sentence)
        for i in range(len(sentenceEmbeddings)):
            vector = sentenceEmbeddings[i]
            cur_sim = cosineSimilarity(vector,vect)
            lookup[cur_sim] = i
            cosSimTable.append(cur_sim)
        cosSimTable.sort(reverse=True)
        for i in range(3):
            result.append(lookup[cosSimTable[i]])
        #print("cosine similartiy: ", cosineSimMax)
        return result,lookup



    def preprocessSentence(self,sentence): 
        meaninglessWord =["turn","your","for","the","on","do","not","at","I","is"]
        # pre-processing sentence to remove meaningless words
        stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
                    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 
                    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
                    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 
                    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
                    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
                    'will', 'just', 'don', 'should', 'now']
        token = tokenize(sentence)
        processed_sentence = ""
        for t in token:
            if t not in meaninglessWord and t not in stop_words:
                processed_sentence += t + " "

        return processed_sentence
       


    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        # sentence = self.preprocessSentence(sentence) 
        # if len(sentence)==0:
        #     return "no"
        # trainingSentences = loadTrainingSentences(file_path)
        # sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(trainingSentences)
        # result,lookup = self.kClosestSentence(sentence, sentenceEmbeddings)
        # if result[0] > 0.5: 
        #     return indexToSentence[result[0]][1]

        #### many many pre-processing ######
        sentence = self.preprocessSentence(sentence) 
        if len(sentence)==0:
            return "no"
        state_key = ["What","orientation","battery","Can","name"]
        light_key = ["holoemitter","logic","minimum","maximum","lights"]
        driving_key = ["degrees","north","east","west","south"]
        grid_key = ["position"]
        token = tokenize(sentence)
        for t in token:
            if t in state_key:
                return "state"
            if t in light_key:
                return "light"
            if t in driving_key:
                return "driving"
            if t in grid_key:
                return "grid"
        if len(token) ==1:
            return "animation"

        ###### general processing #######
        trainingSentences = loadTrainingSentences(file_path)
        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(trainingSentences)
        neigh = NearestNeighbors(n_neighbors = 3)
        neigh.fit(sentenceEmbeddings)
        developEmbed = self.calcSentenceEmbeddingBaseline(sentence).reshape(1,300)
        possible_category_dist, possible_category_index = neigh.kneighbors(developEmbed.reshape(1,300))
        # calculate cosine similarity 
        cosMatrix = []
        for i in possible_category_index[0]:
            cosMatrix.append(cosineSimilarity(sentenceEmbeddings[i],self.calcSentenceEmbeddingBaseline(sentence)))
        # post processing cosine similarity 
        possible_cate = {}
        for i in range(len(cosMatrix)):
            if cosMatrix[i] < 0.3:
                continue
            if cosMatrix[i] > 0.6:
                return indexToSentence[possible_category_index[0][i]][1]
            cate = indexToSentence[possible_category_index[0][i]][1]
            if cate not in possible_cate.keys():
                possible_cate[cate] =1 
            else:
                possible_cate[cate] +=1 
        
        if len(possible_cate.keys()) == 0:
            return "no"
        elif len(possible_cate.keys())==1 and possible_cate.values()==3:
            return possible_cate.keys()
        else: 
            return "no"

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
        devSentences = loadTrainingSentences(dev_file_path)
        count = 0.0
        correct = 0.0
        for key in devSentences.keys():
            for s in devSentences[key]:
                category = self.getCategory(s,training_file_path)
                if category == key:
                    correct += 1.0
                # else:
                #     print(s," ",category, " ",key)
                count+=1.0
        return correct/count
        

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

        ### YOUR CODE HERE ###
        light = ["front","back"] # need extra light flag
        add = ["add","increase"]
        sub = ["reduce","decrease","dim"]
        color = ["rgb","red","green","blue","LED"]
        tokens = tokenize(command)
        light_flag = False
        increase_flag = False
        decrease_flag = False
        for word in tokens:
            if word =="holoemitter":
                slots["holoEmit"] = True
            if word =="logic":
                slots["logDisp"] = True
            if word == "front" or word == "back":
                slots["lights"].append(word)
            # if word == "lights":
            #     light_flag = True
            if word == "off" or word == "minimum" or word == "out":
                slots["off"] = True
                light_flag = True
            if word == "on" or word == "maximum":
                slots["on"] = True
                light_flag = True
            if word in add:
                increase_flag = True
            if word in sub:
                decrease_flag = True
            if word in color and increase_flag:
                slots["add"] = True
                light_flag = True
            if word in color and decrease_flag:
                slots["sub"] = True
                light_flag = True

        if light_flag == True and len(slots["lights"])==0:
            slots["lights"].append("front")
            slots["lights"].append("back")
        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###
        direction_forward = ["forward","north"]
        direction_back = ["back","south"]
        direction_left = ["left","west"]
        direction_right = ["right","east"]
        speed_fast = ["increase","fast"]
        speed_slow = ["decrease","slow"]
        tokens = tokenize(command)
        for word in tokens:
            if word in speed_fast:
                slots["increase"] = True
            if word in speed_slow:
                slots["decrease"] = True
            if word in direction_forward:
                slots["directions"].append("forward")
            if word in direction_back:
                slots["directions"].append("back") 
            if word in direction_left:
                slots["directions"].append("left")
            if word in direction_right:
                slots["directions"].append("right")
        return slots


############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 10

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

# X = WordEmbeddings("/Users/shuqi_zhang/Downloads/GoogleNews-vectors-negative300.magnitude") 
# print(X.lightParser("Increase the red RGB value of your front light by 50."))
# print(X.drivingParser("Go forward, left, right, and then East."))

# svec1 = X.calcSentenceEmbeddingBaseline("drive forward")
# # print(svec1)
# svec2 = X.calcSentenceEmbeddingBaseline("roll ahead")
# svec3 = X.calcSentenceEmbeddingBaseline("set your lights to purple")
# svec4 = X.calcSentenceEmbeddingBaseline("turn your lights to be blue")
# print(cosineSimilarity(svec1, svec2))
# print(cosineSimilarity(svec1, svec3))
# print(cosineSimilarity(svec1, svec4))
# print(cosineSimilarity(svec2, svec3))
# print(cosineSimilarity(svec2, svec4))
# print(cosineSimilarity(svec3, svec4))

# trainingSentences = loadTrainingSentences("data/r2d2TrainingSentences.txt")
# X = WordEmbeddings("/Users/shuqi_zhang/Downloads/GoogleNews-vectors-negative300.magnitude") 
# sentenceEmbeddings, indexToSentence = X.sentenceToEmbeddings(trainingSentences)
# print(sentenceEmbeddings[14:])
# print(indexToSentence[14])

# sentenceEmbeddings, _ = X.sentenceToEmbeddings(loadTrainingSentences("data/r2d2TrainingSentences.txt"))
# print(X.closestSentence("Lights on.", sentenceEmbeddings))


# print(X.getCategory("Drive forward for two feet.", "data/r2d2TrainingSentences.txt"))
# print(X.getCategory("Do not laugh at me.", "data/r2d2TrainingSentences.txt"))



# print(X.getCategory("Turn your lights green.", "data/r2d2TrainingSentences.txt"))
# print(X.getCategory("Do not laugh at me.", "data/r2d2TrainingSentences.txt"))


# print(X.accuracy("data/r2d2TrainingSentences.txt", "data/r2d2DevelopmentSentences.txt"))


# print(X.lightParser("Set your lights to maximum"))


# X = WordEmbeddings("/Users/shuqi_zhang/Downloads/GoogleNews-vectors-negative300.magnitude") 
# print(X.lightParser("Increase the red RGB value of your front light by 50."))
