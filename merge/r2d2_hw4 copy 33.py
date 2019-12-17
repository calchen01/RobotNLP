############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Aditya M. Kashyap"

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
"North-East is at heading 45 degrees",
"South-East is at heading 135 degrees",
"South-West is at heading 225 degrees",
"North-West is at heading 315 degrees",
"Increase speed to 30%",
"Slow down speed to 10%",
"Change direction to heading 180 degrees",
"Turn to your left",
"Halt",
"Turn to face heading 100 degrees"]


my_light_sentences = [
"Stop shining your light",
"Switch off your holoemitter",
"Set the Red,Green,Blue light values to 50,100,150",
"Switch the black lED to green and the front LED to blue",
"Brighten your lights holoemitter",
"Display blue for 2 seconds, followed by red for another 2 seconds",
"Put off your front light",
"Shut off your rear light",
"Halt your front and back lights",
"Switch the lights on both LEDs to red"]


my_head_sentences = [
"Look ahead",
"Look to your right",
"Rotate to see north which is at 40 degrees",
"Swivel your head to see behind you",
"Turn head to face 45 degrees",
"Turn your head to see what is in front of you",
"Turn your head around",
"Turn head to 90 degrees"
"Look northwest at 315 degrees",
"Turn your head right",
"Look immediately in front of you"]


my_state_sentences = [
"Are you conscious?",
"Are you waddling?",
"How intensely are you displaying logic?",
"Are you connected to a droid?",
"Is your continuous roll timer set?",
"What are the RGB values of your front light?",
"At what speed are you driving?",
"What is your current battery voltage?",
"What is your back light color?",
"What is the angle you are driving at?"]


my_connection_sentences = [
"Connect the Q5-4547 robot",
"Establish a connection with Q5-4547",
"Halt connection",
"Disconnect from Q5-4547",
"Link to Q5-4547",
"Stop the connection to Q5-4547",
"Are there droids next to me?",
"Stop connection of Q5-4547 with the server",
"Depart from the server",
"Leave the server"]


my_stance_sentences = [
"Walk clumsily",
"Roll like a biped human",
"Set stance to Triped",
"Set your stance to 2 limbs",
"Use 3 limbs to move",
"Start wobbling and tottering",
"Roll steadily",
"Raise your third wheel",
"Become a Biped",
"Roll on 3 wheels"]


my_animation_sentences = [
"Tip over",
"Shout!",
"Laugh out loud",
"Create a sound",
"Activate the alarm",
"Cry!",
"Giggle softly",
"Sound the alarm",
"Swivel head",
"Make a racket"]


my_grid_sentences = [
"The gird is 4 boxes by 2 boxes",
"The grid is a 4 by 2 rectangle",
"Advance to position (4,2)",
"Droid to (4,2)",
"Run to (4,2)",
"4,3 is blocked from 4,2",
"Position (4,2) has an obstacle",
"Go to the position in front of the table",
"You are standing on (4,2)",
"Proceed to position (4,2)"]


############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
    sentence = "".join([a if a not in string.punctuation + "’" else " " for a in sentence])
    
    tokens = [a.strip().lower() for a in sentence.split(" ") if a.strip() != ""]
    return tokens

def cosineSimilarity(vector1, vector2):
    num = np.sum(np.multiply(vector1,vector2))
    den = (np.sum(np.square(vector1))**0.5)*(np.sum(np.square(vector2))**0.5)
    return round(num/den,8)

def most_frequent(Dict):
    max_value = -100
    for key in Dict.keys():
        if Dict[key] > max_value:
            return_value = key
            max_value = Dict[key]
    return return_value
#    return max(set(Dict), key = Dict.count) 
    



class WordEmbeddings:

    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        sentence_embeddings = np.zeros((300,))
        for token in tokenize(sentence):
            sentence_embeddings += self.vectors.query(token)
        return sentence_embeddings
        
        

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
        counter = 0
        for category in commandTypeToSentences.keys():
            for sentence in commandTypeToSentences[category]:
                current_emb = self.calcSentenceEmbeddingBaseline(sentence)
                sentenceEmbeddings.append(current_emb.reshape((1,current_emb.shape[0])))
                indexToSentence[counter] = (sentence,category)
                counter += 1
        if (len(sentenceEmbeddings) > 0):
            sentenceEmbeddings = np.concatenate(sentenceEmbeddings,axis=0)
        else:
            sentenceEmbeddings = np.array([[] for a in range(300)]).reshape((0,300))
        return (sentenceEmbeddings,indexToSentence)
    

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
        
        currentSentEmb = self.calcSentenceEmbeddingBaseline(sentence)
        sim_scores = []
        for i in range(sentenceEmbeddings.shape[0]):
            sim_scores.append(cosineSimilarity(currentSentEmb,sentenceEmbeddings[i]))
        return np.argmax(sim_scores)
        
        
        

    def getCategory(self, sentence, file_path):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        
        if "?" in sentence:
            return "state"
        
        k = 2
                
        trainingSentences = loadTrainingSentences(file_path)
          
        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(trainingSentences)
        
        currentSentEmb = self.calcSentenceEmbeddingBaseline(sentence)
        
       
       
        sim_scores = []
        categories = []
        for i in range(sentenceEmbeddings.shape[0]):
            sim_scores.append(cosineSimilarity(currentSentEmb,sentenceEmbeddings[i]))
            categories.append(indexToSentence[i][1])
        
        # ARRANGING NEIGHBOURS IN ORDER OF THEIR SIMILARITY
        sorted_ind = np.argsort(-np.array(sim_scores))
        closestCategories = [categories[sorted_ind[a]] for a in range(k)]
        closestSimScores = [sim_scores[sorted_ind[a]] for a in range(k)]
                
        
        
        
        categorySimScore = {}
        categoryCounter = {}
        
        for i in range(len(closestCategories)):
            currentCateg = closestCategories[i]
            currentSimScore = closestSimScores[i]
            
            try:
                categorySimScore[currentCateg] += currentSimScore
                categoryCounter[currentCateg] += 1
            except KeyError:
                categorySimScore[currentCateg] = currentSimScore
                categoryCounter[currentCateg] = 1
        
        for key in categorySimScore.keys():
            categorySimScore[key] = categorySimScore[key]/k
        
        
        final_categories = np.array([a for a in categorySimScore.keys()])
        final_simscores = np.array([categorySimScore[a] for a in categorySimScore.keys()])
        
        final_ind = np.argsort(-final_simscores)
        final_simscores = final_simscores[final_ind]
        final_categories = final_categories[final_ind]
        

        
        
        if len(final_simscores) > 1 and final_simscores[0] - final_simscores[1] < 0.05:
            return 'no'
        elif final_categories[0] == 'state' and "?" not in sentence:
            return 'no'

        
        returnValue = most_frequent(categorySimScore)
            
        
        return returnValue
        
        
        
        

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
        
        c = 0
        s = 0
        
        devSentences = loadTrainingSentences(dev_file_path)
        for category in devSentences.keys():
            for sentence in devSentences[category]:
                pred_category = self.getCategory(sentence, training_file_path)
                if pred_category == category:
                    c += 1
                else:
                    pass

                s += 1
        return c/s
    

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        '''Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        '''
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

        if re.findall("holoemitter",command,re.IGNORECASE):
            slots["holoEmit"] = True
        if re.findall("logic display",command,re.IGNORECASE):
            slots["logDisp"] = True
            
        if re.findall("R[,]?G[,]?B",command,re.IGNORECASE) or re.findall("red|orange|indigo|violet|aqua|yellow|green|blue|purple",command,re.IGNORECASE):
            if (re.findall("front",command,re.IGNORECASE)):
                slots["lights"].append("front")
            if (re.findall("back",command,re.IGNORECASE)):
                slots["lights"].append("back") 
            if (len(slots["lights"]) == 0):
                slots["lights"] += ["front","back"]
                
            
        if (re.findall("light",command,re.IGNORECASE)) and len(slots["lights"]) == 0:
            slots["lights"] += ["front","back"]
        
        if (re.findall("(^|\W)on($|\W)",command,re.IGNORECASE)) or (re.findall("maximum",command,re.IGNORECASE)):
            slots["on"] = True
        if (re.findall("(^|\W)off($|\W)",command,re.IGNORECASE)) or (re.findall("minimum",command,re.IGNORECASE)):
            slots["off"] = True
        
        if (re.findall("increase|add",command,re.IGNORECASE)):
            slots["add"] = True
            
        if (re.findall("decrease|subtract|reduce",command,re.IGNORECASE)):
            slots["sub"] = True
        
        
        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        instruction_dict = {}
        instruction_dict["front"] = "forward"
        instruction_dict["forward"] = "forward"
        instruction_dict["left"] = "left"
        instruction_dict["right"] = "right"
        instruction_dict["north"] = "forward"
        instruction_dict["east"] = "right"
        instruction_dict["west"] = "left"

        tokens = tokenize(command)
        directions = ["front","forward","left","right","north","east","west"]
        given_directions = [instruction_dict[a] for a in tokens if a in directions]
        
        slots["directions"] += given_directions
        
        if (re.findall("increase",command,re.IGNORECASE)):
            slots["increase"] = True
            
        if (re.findall("decrease|reduce",command,re.IGNORECASE)):
            slots["decrease"] = True
        


        return slots


############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 15

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


#%%



#trainingSentences = loadTrainingSentences("/Users/aditya/Classes/CIS521/r2d2_hw4/data/r2d2TrainingSentences.txt")
#X = WordEmbeddings("/Users/aditya/Downloads/GoogleNews-vectors-negative300.magnitude") # Change this to where you downloaded the file.

#
#X.getCategory("Turn your lights green.", "/Users/aditya/Classes/CIS521/r2d2_hw4/data/r2d2TrainingSentences.txt")
#X.getCategory("Drive forward for two feet.", "data/r2d2TrainingSentences.txt")
#X.getCategory("Do not laugh at me.", "data/r2d2TrainingSentences.txt")
#
#acc =X.accuracy("/Users/aditya/Classes/CIS521/r2d2_hw4/data/r2d2TrainingSentences.txt", "/Users/aditya/Classes/CIS521/r2d2_hw4/data/r2d2DevelopmentSentences.txt")

#print(X.lightParser("Set your lights to maximum"))
#print(X.lightParser("Increase the red RGB value of your front light by 50."))
#print(X.drivingParser("Increase your speed!"))
#print(X.drivingParser("Go forward, left, right, and then East."))





#print(X.lightParser("Set your lights to maximum"))
#print(X.lightParser("Increase the red RGB value of your front light by 50."))
#print(X.lightParser("Turn your lights off."))
#print(X.lightParser("Set the holoemitter to maximum."))
#print(X.lightParser("Change your back light to aqua."))
#print(X.lightParser("Turn off your logic display."))
#print(X.lightParser("Set the green value on your back light to 0."))
#print(X.lightParser("Change your forward light to red."))
#print(X.lightParser("Reduce the green value on your lights by 50."))








