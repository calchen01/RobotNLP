from pymagnitude import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random
import time
import csv
import re
import numpy as np

from client import DroidClient

path = "/Volumes/SD/hw4_2019/"
vectors = Magnitude(path + "vectors/word2vecRetrofitted.magnitude")

def loadTrainingSentences(file_path):
    commandTypeDict = {}

    with open(file_path, 'r') as fin:
        for line in fin:
            line = line.rstrip('\n')
            if len(line.strip()) == 0 or "##" == line.strip()[0:2]:
                continue
            commandType, command = line.split(' :: ')
            if commandType not in commandTypeDict:
                commandTypeDict[commandType] = [command]
            else:
                commandTypeDict[commandType].append(command)

    return commandTypeDict

def cosineSim(vector1, vector2):
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2))

def calcPhraseEmbedding(sentence):
    words = re.split('\W+', sentence)
    words = [x.lower() for x in words if x.lower() not in ["", "a", "an", "the", "is"]]
    if "?" in sentence:
        words.append("?")
    return vectors.query(words).sum(axis = 0) / len(words)

def rankSentences(commandEmbedding, sentenceEmbeddings):
    sortList = []

    for i in range(sentenceEmbeddings.shape[0]):
        similarity = cosineSim(commandEmbedding, sentenceEmbeddings[i, :])
        sortList.append((i, similarity))

    similarSentences = sorted(sortList, key = lambda x: x[1], reverse = True)

    return [x[0] for x in similarSentences]

def getCommandType(categories, closestSentences, indexToTrainingSentence):
    commandDict = {}
    for category in categories:
        commandDict[category] = 0

    commandDict[indexToTrainingSentence[closestSentences[0]][1]] += 1
    commandDict[indexToTrainingSentence[closestSentences[1]][1]] += 0.5
    commandDict[indexToTrainingSentence[closestSentences[2]][1]] += 0.5
    commandDict[indexToTrainingSentence[closestSentences[3]][1]] += 0.2
    commandDict[indexToTrainingSentence[closestSentences[4]][1]] += 0.2
    print(commandDict)

    return max(commandDict, key=commandDict.get)

# R2D2 class
class AnimateR2:
    def __init__(self, droidID, wordSimilarityCutoff, voice):
        self.createSentenceEmbeddings()

        self.droid = DroidClient()
        self.droid.connect_to_droid(droidID)
        self.name = "R2"

        self.wordSimilarityCutoff = wordSimilarityCutoff
        self.voice = voice

        # color variables
        self.holoProjectorIntensity = 0
        self.logicDisplayIntensity = 0
        self.frontRGB = (0, 0, 0)
        self.backRGB = (0, 0, 0)

        self.colorToRGB = {}
        with open(path + 'data/colors.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                self.colorToRGB[row[0]] = (int(row[2]), int(row[3]), int(row[4]))

        # direction variables
        self.dirMap = {
            "up": 0,
            "right": 90,
            "down": 180,
            "left": 270
        }


    def createSentenceEmbeddings(self):
        self.categories = ["state", "direction", "light", "animation", "head", "grid"]
        trainingSentences = loadTrainingSentences(path + "data/r2d2TrainingSentences.txt")

        self.indexToTrainingSentence = {}

        i = 0
        for category in self.categories:
            sentences = trainingSentences[category + "Sentences"]
            for sentence in sentences:
                self.indexToTrainingSentence[i] = (sentence, category)
                i += 1

        self.sentenceEmbeddings = np.zeros((len(self.indexToTrainingSentence), vectors.dim))

        for i in range(len(self.indexToTrainingSentence)):
            sentence = self.indexToTrainingSentence[i][0]
            sentenceEmbedding = calcPhraseEmbedding(sentence)

            self.sentenceEmbeddings[i, :] = sentenceEmbedding

    def inputCommand(self, command):
        commandEmbedding = calcPhraseEmbedding(command)

        closestSentences = rankSentences(commandEmbedding, self.sentenceEmbeddings)

        # print(self.indexToTrainingSentence[closestSentences[0]][0])
        # print(self.indexToTrainingSentence[closestSentences[1]][0])
        # print(self.indexToTrainingSentence[closestSentences[2]][0])
        # print(self.indexToTrainingSentence[closestSentences[3]][0])
        # print(self.indexToTrainingSentence[closestSentences[4]][0])

        if not self.voice:
            print("Closet sentence was: " + self.indexToTrainingSentence[closestSentences[0]][0])
            print("Its cosine similarity to the command was: " + str(cosineSim(commandEmbedding, self.sentenceEmbeddings[closestSentences[0], :])))

        if cosineSim(commandEmbedding, self.sentenceEmbeddings[closestSentences[0], :]) < 0.84:
            if not self.voice:
                subcommand = input(self.name + ": I could not understand your command. Do you want to add this command to the training set? (yes/no): ")
                if "yes" in subcommand.lower():
                    subcommand = input("What category do you want to add it to? Choices are state, direction, light, animation, head, or grid: ")
                    subcommand = subcommand.lower()
                    if subcommand in self.categories:
                        with open(path + "data/r2d2TrainingSentences.txt", 'a') as the_file:
                            the_file.write(subcommand + 'Sentences :: ' + command + '\n')
                        print("Command added. Changes will be present on restart.")
                    else:
                        print(subcommand + " not a valid category.")
            else:
                superDumbVariable = True
                # want to make it so the R2D2 goes like nuh huh, can't understand
            return

        commandType = getCommandType(self.categories, closestSentences, self.indexToTrainingSentence)
        result = getattr(self, commandType + "Parser")(command.lower())
        if not self.voice:
            if result:
                print(self.name + ": Done executing "  + commandType + " command.")
            else:
                print(self.name + ": I could not understand your " + commandType + " command.")







    def roll(self, heading):
        self.droid.roll(0, self.dirMap.get(heading), 0)
        time.sleep(0.35)
        self.droid.roll(1, self.dirMap.get(heading), 0.62)

    def animate(self):
        # 55 animations available
        self.droid.animate(random.randint(1, 55))

    def play_sound(self):
        # 49 sounds available
        self.droid.play_sound(random.randint(1, 49))

    def reset(self):
        self.droid.roll(0, 0, 0)

    def disconnect(self):
        self.droid.disconnect()

    def flash_colors(self, colors, seconds = 1, front = True):
        if front:
            for color in colors:
                self.droid.set_front_LED_color(*color)
                time.sleep(seconds)
        else:
            for color in colors:
                self.droid.set_back_LED_color(*color)
                time.sleep(seconds)

    def askForColor(self, lightPosition = "both"):
        if self.voice:
            return False
        
        if lightPosition != "both":
            print("We detect that you want to change your " + lightPosition + " light, but could not find a color.")
        else:
            print("We parsed this as a light command, but could not find a color.")
        command = input("Do you want to input a color? (yes/no): ")
        color = False
        if "yes" in command.lower():
            print("You may have inputted a color, but it is not in our database or is mispelled. Please input a color or rgb tuple.")
            command = input("If you want to add the color to the database, input color_name (one string) :: rgb tuple: ")

            words = re.split('\W+', command)
            words = [x for x in words if x != ""]
            for word in words:
                if word in self.colorToRGB: color = self.colorToRGB[word]
            if len(words) == 4:
                try:
                    color = (int(words[1]), int(words[2]), int(words[3]))
                    colorName = words[0]
                    with open(path + 'data/colors.csv', 'a') as csvStorer:
                        csvStorer.write('\n' + colorName + ',R2D2 ' + colorName + ',' + words[1] + ',' + words[2] + ',' + words[3])
                    print(colorName + " added to database. It will be available on the next restart.")
                except ValueError:
                    superDumbVariable = 1
            elif len(words) == 3:
                try:
                    color = (int(words[0]), int(words[1]), int(words[2]))
                except ValueError:
                    superDumbVariable = 1

        return color
















    def stateParser(self, command):
        print("stateParser has not yet been initialized.")

    def directionParser(self, command):
        tokens = re.split("[^a-zA-Z]", command)
        for token in tokens:
            if token in {"up", "forward", "ahead", "straight"}:
                self.roll("up")
                return True
            elif token in {"down", "back"}:
                self.roll("down")
                return True
            elif token in {"left", "right"}:
                self.roll(token)
                return True

    def lightParser(self, command):
        # slot filler for lights
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False,
        "percent": False, "whichRGB": [], "colors": [], "intensities": [], "rgb": False, "increment/seconds": False}

        if "holoemitter" in command or "holo emitter" in command:
            slots["holoEmit"] = True
        if "logic display" in command:
            slots["logDisp"] = True

        if "dim" in command:
            slots["intensities"].append("dim")
        if "blink" in command:
            slots["intensities"].append("blink")

        if "%" in command:
            slots["percent"] = True

        # WANT TO MAKE INCREASE BETTER
        if "increase" in command or "add" in command:
            slots["add"] = True
        if "decrease" in command or "reduce" in command or "subtract" in command:
            slots["sub"] = True

        # front back too similar
        if "back" in command:
            slots["lights"].append("back")
        if "front" in command:
            slots["lights"].append("front")
        if slots["lights"] == []:
            slots["lights"] = ["front", "back"]

        if "red" in command:
            slots["whichRGB"].append("red")
        if "green" in command:
            slots["whichRGB"].append("green")
        if "blue" in command:
            slots["whichRGB"].append("blue")

        words = re.split('\W+', command)
        words = [x for x in words if x != ""]

        i = 0
        for word in words:
            if i < len(words) - 2:
                try:
                    slots["rgb"] = (int(words[i]), int(words[i+1]), int(words[i+2]))
                except ValueError:
                    superDumbVariable = True
            if vectors.similarity("off", word) > self.wordSimilarityCutoff or "minimum" in command:
                slots["intensities"].append("off")
            elif vectors.similarity("on", word) > self.wordSimilarityCutoff or vectors.similarity("maximum", word) > self.wordSimilarityCutoff:
                slots["intensities"].append("on")
            if vectors.similarity("percent", word) > self.wordSimilarityCutoff:
                slots["percent"] = True
            if word in self.colorToRGB:
                slots["colors"].append(self.colorToRGB[word])
            i += 1

            try:
                increment = int(word)
                slots["increment/seconds"] = increment
            except ValueError:
                continue

        return self.lightSlotsToActions(slots)

    def lightSlotsToActions(self, slots):
        if slots["holoEmit"]:
            if "off" in slots["intensities"]:
                self.holoProjectorIntensity = 0
                self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            elif "dim" in slots["intensities"]:
                self.holoProjectorIntensity = self.holoProjectorIntensity / 2
                self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            elif "on" in slots["intensities"]:
                self.holoProjectorIntensity = 1
                self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            elif "blink" in slots["intensities"]:
                self.droid.set_holo_projector_intensity((self.holoProjectorIntensity + 1)%2)
                time.sleep(0.3)
                self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            else:
                return False
            return True

        if slots["logDisp"]:
            if "off" in slots["intensities"]:
                self.logicDisplayIntensity = 0
                self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            elif "dim" in slots["intensities"]:
                self.logicDisplayIntensity = self.logicDisplayIntensity / 2
                self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            elif "on" in slots["intensities"]:
                self.logicDisplayIntensity = 1
                self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            elif "blink" in slots["intensities"]:
                self.droid.set_logic_display_intensity((self.logicDisplayIntensity + 1)%2)
                time.sleep(0.3)
                self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            else:
                return False
            return True

        if (slots["add"] or slots["sub"]) and (slots["percent"]):
            lights = slots["lights"]

            if not slots["increment/seconds"]:
                command = input("Percent not found in command, please input percent to change by here: ")
                try:
                    command = command.replace("%", "")
                    slots["increment/seconds"] = int(command)
                except ValueError:
                    return False

            if slots["sub"]: slots["increment/seconds"] = -slots["increment/seconds"]

            percent = slots["increment/seconds"]

            if len(slots["whichRGB"]) == 0:
                command = input("Did not find what values (red/blue/green) to change, input what values to change: ")
                if "red" in command: slots["whichRGB"].append("red")
                if "green" in command: slots["whichRGB"].append("green")
                if "blue" in command: slots["whichRGB"].append("blue")

            if len(slots["whichRGB"]) == 0: return False

            if "red" in slots["whichRGB"]:
                for light in lights:
                    rgb = getattr(self, light+"RGB")
                    setattr(self, light+"RGB", (max(0, min(rgb[0] + rgb[0]*percent/100, 255)), rgb[1], rgb[2]))
                    getattr(self.droid, "set_"+light+"_LED_color")(*getattr(self, light+"RGB"))
            if "green" in slots["whichRGB"]:
                for light in lights:
                    rgb = getattr(self, light+"RGB")
                    setattr(self, light+"RGB", (rgb[0], max(0, min(rgb[1] + rgb[1]*percent/100, 255)), rgb[2]))
                    getattr(self.droid, "set_"+light+"_LED_color")(*getattr(self, light+"RGB"))
            if "blue" in slots["whichRGB"]:
                for light in lights:
                    rgb = getattr(self, light+"RGB")
                    setattr(self, light+"RGB", (rgb[0], rgb[1], max(0, min(rgb[2] + rgb[2]*percent/100, 255))))
                    getattr(self.droid, "set_"+light+"_LED_color")(*getattr(self, light+"RGB"))

            return True

        if slots["add"] or slots["sub"]:
            lights = slots["lights"]

            if not slots["increment/seconds"]:
                command = input("Increment not found in command, please input amount to change by here: ")
                try:
                    slots["increment/seconds"] = int(command)
                except ValueError:
                    return False

            if slots["sub"]: slots["increment/seconds"] = -slots["increment/seconds"]

            increaseValue = slots["increment/seconds"]

            if len(slots["whichRGB"]) == 0:
                command = input("Did not find what values (red/blue/green) to change, input what values to change: ")
                if "red" in command: slots["whichRGB"].append("red")
                if "green" in command: slots["whichRGB"].append("green")
                if "blue" in command: slots["whichRGB"].append("blue")

            if len(slots["whichRGB"]) == 0: return False

            if "red" in slots["whichRGB"]:
                for light in lights:
                    rgb = getattr(self, light+"RGB")
                    setattr(self, light+"RGB", (max(0, min(rgb[0] + increaseValue, 255)), rgb[1], rgb[2]))
                    getattr(self.droid, "set_"+light+"_LED_color")(*getattr(self, light+"RGB"))
            if "green" in slots["whichRGB"]:
                for light in lights:
                    rgb = getattr(self, light+"RGB")
                    setattr(self, light+"RGB", (rgb[0], max(0, min(rgb[1] + increaseValue, 255)), rgb[2]))
                    getattr(self.droid, "set_"+light+"_LED_color")(*getattr(self, light+"RGB"))
            if "blue" in slots["whichRGB"]:
                for light in lights:
                    rgb = getattr(self, light+"RGB")
                    setattr(self, light+"RGB", (rgb[0], rgb[1], max(0, min(rgb[2] + increaseValue, 255))))
                    getattr(self.droid, "set_"+light+"_LED_color")(*getattr(self, light+"RGB"))

            return True

        askedForColor = False

        if "back" in slots["lights"] and len(slots["lights"]) == 1:
            if len(slots["colors"]) > 1:
                seconds = slots["increment/seconds"]
                if not seconds: seconds = 1
                self.flash_colors(slots["colors"], seconds, False)
            elif len(slots["colors"]) == 1:
                self.backRGB = slots["colors"][0]
            else:
                if not slots["rgb"]:
                    color = self.askForColor("back")
                    askedForColor = True
                    if not color: return False
                    self.backRGB = color
                else:
                    self.backRGB = slots["rgb"]

            self.droid.set_back_LED_color(*self.backRGB)
            return True

        if ("front" in slots["lights"] and len(slots["lights"]) == 1) or len(slots["colors"]) > 1:
            if len(slots["colors"]) > 1:
                seconds = slots["increment/seconds"]
                if not seconds: seconds = 1
                self.flash_colors(slots["colors"], seconds)
            elif len(slots["colors"]) == 1:
                self.frontRGB = slots["colors"][0]
            else:
                if not slots["rgb"]:
                    color = self.askForColor("front")
                    askedForColor = True
                    if not color: return False
                    self.frontRGB = color
                else:
                    self.frontRGB = slots["rgb"]

            self.droid.set_front_LED_color(*self.frontRGB)
            return True

        if len(slots["colors"]) == 1:
            self.backRGB = slots["colors"][0]
            self.frontRGB = slots["colors"][0]
            self.droid.set_back_LED_color(*self.backRGB)
            self.droid.set_front_LED_color(*self.frontRGB)
            return True

        if len(slots["colors"]) == 0:
            if slots["rgb"]:
                self.backRGB = slots["rgb"]
                self.frontRGB = slots["rgb"]
                self.droid.set_back_LED_color(*self.backRGB)
                self.droid.set_front_LED_color(*self.frontRGB)
                return True

        if "off" in slots["intensities"]:
            self.holoProjectorIntensity = 0
            self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            self.logicDisplayIntensity = 0
            self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            self.backRGB = (0, 0, 0)
            self.frontRGB = (0, 0, 0)
            self.droid.set_back_LED_color(*self.backRGB)
            self.droid.set_front_LED_color(*self.frontRGB)
            return True
        elif "dim" in slots["intensities"]:
            self.holoProjectorIntensity = 0
            self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            self.logicDisplayIntensity = 0
            self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            self.backRGB = tuple(x/2 for x in self.backRGB)
            self.frontRGB = tuple(x/2 for x in self.frontRGB)
            self.droid.set_back_LED_color(*self.backRGB)
            self.droid.set_front_LED_color(*self.frontRGB)
            return True
        elif "on" in slots["intensities"]:
            self.holoProjectorIntensity = 1
            self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            self.logicDisplayIntensity = 1
            self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            return True
        elif "blink" in slots["intensities"]:
            self.droid.set_holo_projector_intensity((self.holoProjectorIntensity + 1)%2)
            self.droid.set_logic_display_intensity((self.holoProjectorIntensity + 1)%2)
            time.sleep(0.3)
            self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            self.droid.set_logic_display_intensity(self.logicDisplayIntensity)  
            return True

        if not slots["rgb"] and not askedForColor:
            color = self.askForColor()
            if color:
                self.backRGB = color
                self.frontRGB = color
                self.droid.set_back_LED_color(*self.backRGB)
                self.droid.set_front_LED_color(*self.frontRGB)
                return True

        return False

    def animationParser(self, command):
        print("animationParser has not yet been initialized.")

    def headParser(self, command):
        print("headParser has not yet been initialized.")

    def gridParser(self, command):
        print("gridParser has not yet been initialized.")