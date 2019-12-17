############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Steven Bursztyn"

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

    with open(file_path, "r") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if len(line.strip()) == 0 or "##" == line.strip()[0:2]:
                continue
            commandType, command = line.split(" :: ")
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
    "Move forward 2 units, and turn left.",
    "Head East.",
    "Go North.",
    "Turn 30 degrees.",
    "Increase speed to max speed.",
    "Stop moving!",
    "Go fast!",
    "Go backwards",
    "Go forward for one second",
    "Go East.",
]

my_light_sentences = [
    "Max brightness.",
    "Turn off your holoemitter.",
    "Turn off your lights.",
    "Low brightness.",
    "Change color to purple.",
    "Change color to blue.",
    "Change color to yellow.",
    "Change color to green.",
    "Change color to orange.",
    "Blink your lights.",
]

my_head_sentences = [
    "Turn head to face east.",
    "Turn head to face south.",
    "Turn head to face north.",
    "Turn head to face west.",
    "Look behind you.",
    "Look in front of you.",
    "Look to your right.",
    "Look to your left.",
    "Do nothing.",
    "Turn head 30 degrees.",
]

my_state_sentences = [
    "Are you on?",
    "Is your light on?",
    "What is your speed?",
    "What direction are you facing?",
    "Battery status?",
    "How fast are you moving?",
    "What is your battery level?",
    "What is your current direction?",
    "What color is your light?",
    "Is your light off?",
]

my_connection_sentences = [
    "Connect D2-55A2 to the server",
    "Connect D2-59A1 to the server",
    "Connect D2-82S1 to the server",
    "How many droids are nearby?",
    "Are there droid nearby?",
    "Disconnect.",
    "Disconnect me from server.",
    "Disconnect me from the server.",
    "Disconnect D2-55A2 from the server.",
    "Disconnect me now.",
]

my_stance_sentences = [
    "Set stance to two",
    "Set stance to three",
    "Stand on heels." "Spin.",
    "Jump.",
    "Move on three.",
    "Put down third wheel, please.",
    "Stand on two feet.",
    "Stand on three feet.",
    "Move on two feet.",
    "Change your stance.",
]

my_animation_sentences = [
    "Make noise.",
    "Yell!",
    "Tweet.",
    "Waddle.",
    "Todder.",
    "Begin to waddle.",
    "Walk like a duck.",
    "Move like a duck.",
    "Make some noise.",
    "Make as much noise as possible.",
]

my_grid_sentences = [
    "Your world is 4 by 5",
    "Each slot is a 1 foot.",
    "The grid is 4x4.",
    "The grid is 4 by 4.",
    "Go to slot (1, 1)",
    "Go to (1, 1)",
    "Move to (1, 1)",
    "Move to start.",
    "Move around the 4x4 grid.",
    "Please move around the grid.",
]


############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"


def tokenize(sentence):
    return [word.lower() for word in re.findall(r"[\w]+", sentence)]


def cosineSimilarity(vector1, vector2):
    numerator = np.dot(vector1, vector2)
    denom = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    return numerator / denom


class WordEmbeddings:
    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        tokens = tokenize(sentence)
        if len(tokens) == 0:
            return np.zeros(self.vectors.dim)

        query = self.vectors.query(tokens)

        return query.sum(axis=0)

    def sentenceToEmbeddings(self, commandTypeToSentences):
        """Returns a tuple of sentence embeddings and an index-to-(category, sentence)
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
        """
        indexToSentence = dict()

        count = 0
        for command_type, sentence_list in commandTypeToSentences.items():
            for sentence in sentence_list:
                indexToSentence[count] = (sentence, command_type)
                count += 1

        sentenceEmbeddings = np.empty((count, self.vectors.dim))

        for i in range(count):
            sentenceEmbeddings[i:] = self.calcSentenceEmbeddingBaseline(
                indexToSentence[i][0]
            )

        return (sentenceEmbeddings, indexToSentence)

    def closestSentence(self, sentence, sentenceEmbeddings):
        """Returns the index of the closest sentence to the input, 'sentence'.

        Inputs:
            sentence: A sentence

            sentenceEmbeddings: An mxn numpy array, where m is the total number
            of sentences and n is the dimension of the vectors.

        Returns:
            an integer i, where i is the row index in sentenceEmbeddings 
            that contains the closest sentence to the input
        """
        score = self.calcSentenceEmbeddingBaseline(sentence)

        top = -np.inf
        index = None

        num_sentences, _ = sentenceEmbeddings.shape

        for i in range(num_sentences):
            similarity_score = cosineSimilarity(sentenceEmbeddings[i, :], score)
            if top < similarity_score:
                top = similarity_score
                index = i

        return index

    def getCategory(self, sentence, file_path):
        """Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        """
        sentence = sentence.lower()

        if (
            "holoemitter" in sentence
            or "light" in sentence
            or "logic display" in sentence
            or "red" in sentence
            or "blue" in sentence
            or "green" in sentence
        ):
            return "light"

        if (
            "drive" in sentence
            or "go" in sentence
            or "roll" in sentence
            or "turn" in sentence
            or "halt" in sentence
            or "run" in sentence
            or "head" in sentence
        ):
            return "driving"

        if (
            "?" in sentence
            or "what" in sentence
            or "how" in sentence
            or "are" in sentence
        ):
            return "state"

        if (
            "grid" in sentence
            or "square" in sentence
            or "(" in sentence
            or "position" in sentence
        ):
            return "grid"

        if (
            "fall" in sentence
            or "roll" in sentence
            or "scream" in sentence
            or "noise" in sentence
            or "speak" in sentence
            or "sing" in sentence
        ):
            return "animation"

        return "no"

    def accuracy(self, training_file_path, dev_file_path):
        """Returns the accuracy of your implementation of getCategory

        Inputs:
            training_file_path: path to training set

            dev_file_path: path to development set

        Let c = number of correctly categorized sentences in the development set.
        Let s = total number of sentences in the development set.

        Returns:
            A float equal to c/s.
        """
        c = 0
        s = 0

        dev_sentences = loadTrainingSentences(dev_file_path)

        for command_type, sentence_list in dev_sentences.items():
            s += len(sentence_list)
            for sentence in sentence_list:
                if command_type == self.getCategory(sentence, training_file_path):
                    c += 1

        return c / s

    ############################################################
    # Section 3: Slot filling
    ############################################################

    def lightParser(self, command):
        """Slots for light command
        The slot "lights" can have any combination of "front"/"back"
        """
        slots = {
            "holoEmit": False,
            "logDisp": False,
            "lights": [],
            "add": False,
            "sub": False,
            "off": False,
            "on": False,
        }

        ### YOUR CODE HERE ###

        command = command.lower()

        if "holoemitter" in command:
            slots["holoEmit"] = True
            slots["lights"].append("front")
            slots["lights"].append("back")

        if "logic display" in command:
            slots["logDisp"] = True

        if "front" in command:
            slots["lights"].append("front")
        if "back" in command:
            slots["lights"].append("back")
        if "lights" in command:
            slots["lights"].append("front")
            slots["lights"].append("back")

        if "add" in command or "increase" in command or "gain" in command:
            slots["add"] = True
        if "sub" in command or "decrease" in command or "reduce" in command:
            slots["sub"] = True

        if "turn" in command and "off" in command:
            slots["off"] = True
        if "turn" in command and ("on" in command or "max" in command):
            slots["on"] = True

        return slots

    def drivingParser(self, command):
        """Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        """
        slots = {"increase": False, "decrease": False, "directions": []}

        ### YOUR CODE HERE ###

        command = command.lower()

        if "increase" in command or "fast" in command:
            slots["increase"] = True

        if "decrease" in command or "slow" in command:
            slots["decrease"] = True

        # Insns: forward, back, left, or right. Cardinal directions like "South"
        # should map onto back, and "East" should map onto right, etc.
        for word in command.split():
            if "forward" in word or "up" in word or "north" in word:
                slots["directions"].append("forward")
            elif "back" in word or "down" in word or "south" in word:
                slots["directions"].append("back")
            elif "left" in word or "west" in word:
                slots["directions"].append("left")
            elif "right" in word or "east" in word:
                slots["directions"].append("right")

        return slots


############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 6

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
