from pymagnitude import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
import csv
import re
import numpy as np

from client import DroidClient

path = "/Volumes/SD/hw4_2019/"

vectors = Magnitude(path + "vectors/word2vecRetrofitted.magnitude")

stateSentences = [
# Questions about the state of the droid
"What color is your front light?",
"Are there any other droids nearby?", 
"What is your name?", 
"I want to call you 'Artoo'.", 
"What is your orientation?", 
"What is your current heading?", 
"How much battery do you have left?", 
"What is your battery status?"
]

directionSentences = [
# Direction commands:
"North is at heading 50 degrees.", 
"Go North.", 
"Go East.", 
"Go South-by-southeast", 
"Run away!",
"There’s a storm trooper to your left!  Run away from the storm trooper!", 
"Turn to heading 30 degrees.", 
"Reset your heading to 0", 
"Turn to face North.", 
"Start rolling forward.", 
"Increase your speed by 50%.", 
"Turn to your right.", 
"Stop.", 
"Set speed to be 0.", 
"Set speed to be 20%", 
"Go forward for 2 feet, then turn right.", 
"Turn around"
]

lightSentences = [
# Light commands
"Change the intensity on the holoemitter to maximum.", 
"Turn off the holoemitter.", 
"Blink your logic display.", 
"Change the back LED to green.", 
"Dim your lights holoemitter.", 
"Turn off all your lights.", 
"Lights out.", 
"Set the RGB values on your lights to be 255,0,0.", 
"Add 100 to the red value of your front LED.", 
"Increase the blue value of your back LED by 50%.", 
"Display the following colors for 2 seconds each: red, orange, yellow, green, blue, purple.", 
"Change the color on both LEDs to be green."
]

animationSentences = [
# Animations/sound
"Fall over", 
"Scream", 
"Make some noise", 
"Laugh", 
"Play an alarm"
]

headSentences = [
# Head rotation
"turn your head to face forward", 
"look behind you"
]

gridSentences = [
# Relationships on a grid.
"You are on a 4 by 5 grid.", 
"Each square is 1 foot large.", 
"You are at position (0,0).", 
"Go to position (3,3).", 
"There is an obstacle at position 2,1.", 
"There is a chair at position 3,3", 
"Go to the left of the chair.", 
"It’s not possible to go from 2,2 to 2,3."
]

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

    return max(commandDict, key=commandDict.get)

class AnimateR2:
    def __init__(self, droidID):
        self.droid = DroidClient()
        self.droid.connect_to_droid(droidID)
        self.name = "R2"
        self.holoProjectorIntensity = 0
        self.logicDisplayIntensity = 0
        self.frontRGB = (0, 0, 0)
        self.backRGB = (0, 0, 0)

        self.colorToRGB = {}

        with open(path + 'data/colors.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                self.colorToRGB[row[0]] = (int(row[2]), int(row[3]), int(row[4]))

    def stateParser(self, command):
        print("stateParser has not yet been initialized.")

    def directionParser(self, command):
        print("directionParser has not yet been initialized.")

    def lightParser(self, command):
        if "holoemitter" in command or "holo emitter" in command:
            if "off" in command or "out" in command or "minimum" in command:
                self.holoProjectorIntensity = 0
                self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            if "dim" in command:
                self.holoProjectorIntensity = self.holoProjectorIntensity / 2
                self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            if "on" in command or "maximum" in command:
                self.holoProjectorIntensity = 1
                self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            if "blink" in command:
                self.droid.set_holo_projector_intensity((self.holoProjectorIntensity + 1)%2)
                time.sleep(0.3)
                self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            return True

        if "logic display" in command:
            if "off" in command or "out" in command or "minimum" in command:
                self.logicDisplayIntensity = 0
                self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            if "dim" in command:
                self.logicDisplayIntensity = self.logicDisplayIntensity / 2
                self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            if "on" in command or "maximum" in command:
                self.logicDisplayIntensity = 1
                self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            if "blink" in command:
                self.droid.set_logic_display_intensity((self.holoProjectorIntensity + 1)%2)
                time.sleep(0.3)
                self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            return True

        words = re.split('\W+', command)
        words = [x for x in words if x != ""]

        if ("increase" in command or "decrease" in command or "reduce" in command) and ("%" in command or "percent" in command):
            lights = []
            if "back" in command:
                lights.append("back")
            if "front" in command:
                lights.append("front")
            if lights == []:
                lights = ["front", "back"]

            foundNum = False
            percent = 0

            for word in words:
                try:
                    percent = int(word)
                    foundNum = True
                except ValueError:
                    continue

            if not foundNum:
                return False

            if "decrease" in command or "reduce" in command:
                percent = -percent

            changed = False

            if "red" in command:
                for light in lights:
                    rgb = getattr(self, light+"RGB")
                    setattr(self, light+"RGB", (max(0, min(rgb[0] + rgb[0]*percent/100, 255)), rgb[1], rgb[2]))
                    getattr(self.droid, "set_"+light+"_LED_color")(*getattr(self, light+"RGB"))
                changed = True
            if "green" in command:
                for light in lights:
                    rgb = getattr(self, light+"RGB")
                    setattr(self, light+"RGB", (rgb[0], max(0, min(rgb[1] + rgb[1]*percent/100, 255)), rgb[2]))
                    getattr(self.droid, "set_"+light+"_LED_color")(*getattr(self, light+"RGB"))
                changed = True
            if "blue" in command:
                for light in lights:
                    rgb = getattr(self, light+"RGB")
                    setattr(self, light+"RGB", (rgb[0], rgb[1], max(0, min(rgb[2] + rgb[2]*percent/100, 255))))
                    getattr(self.droid, "set_"+light+"_LED_color")(*getattr(self, light+"RGB"))
                changed = True

            return changed

        if "increase" in command or "decrease" in command or "add" in command or "subtract" in command or "reduce" in command:
            lights = []
            if "back" in command:
                lights.append("back")
            if "front" in command:
                lights.append("front")
            if lights == []:
                lights = ["front", "back"]

            foundNum = False
            increaseValue = 0

            for word in words:
                try:
                    increaseValue = int(word)
                    foundNum = True
                except ValueError:
                    continue

            if not foundNum:
                return False

            if "decrease" in command or "reduce" in command or "subtract" in command:
                increaseValue = -increaseValue

            changed = False

            if "red" in command:
                for light in lights:
                    rgb = getattr(self, light+"RGB")
                    setattr(self, light+"RGB", (max(0, min(rgb[0] + increaseValue, 255)), rgb[1], rgb[2]))
                    getattr(self.droid, "set_"+light+"_LED_color")(*getattr(self, light+"RGB"))
                changed = True
            if "green" in command:
                for light in lights:
                    rgb = getattr(self, light+"RGB")
                    setattr(self, light+"RGB", (rgb[0], max(0, min(rgb[1] + increaseValue, 255)), rgb[2]))
                    getattr(self.droid, "set_"+light+"_LED_color")(*getattr(self, light+"RGB"))
                changed = True
            if "blue" in command:
                for light in lights:
                    rgb = getattr(self, light+"RGB")
                    setattr(self, light+"RGB", (rgb[0], rgb[1], max(0, min(rgb[2] + increaseValue, 255))))
                    getattr(self.droid, "set_"+light+"_LED_color")(*getattr(self, light+"RGB"))
                changed = True

            return changed

        colors = []
        for word in words:
            if word in self.colorToRGB:
                colors.append(self.colorToRGB[word])

        if "back" in command:
            if len(colors) > 1:
                seconds = 1
                for word in words:
                    try:
                        seconds = int(word)
                    except ValueError:
                        continue
                self.flash_colors(colors, seconds, False)
            elif len(colors) == 1:
                self.backRGB = colors[0]
            else:
                changed = False
                for i in range(len(words) - 2):
                    try:
                        self.backRGB = (int(words[i]), int(words[i+1]), int(words[i+2]))
                        changed = True
                    except ValueError:
                        continue
                if not changed:
                    return False

            self.droid.set_back_LED_color(*self.backRGB)
            return True

        if "front" in command or "forward" in command or len(colors) > 1:
            if len(colors) > 1:
                seconds = 1
                for word in words:
                    try:
                        seconds = int(word)
                    except ValueError:
                        continue
                self.flash_colors(colors, seconds)
            elif len(colors) == 1:
                self.frontRGB = colors[0]
            else:
                changed = False
                for i in range(len(words) - 2):
                    try:
                        self.frontRGB = (int(words[i]), int(words[i+1]), int(words[i+2]))
                        changed = True
                    except ValueError:
                        continue
                if not changed:
                    return False

            self.droid.set_front_LED_color(*self.frontRGB)
            return True

        if len(colors) == 1:
            self.backRGB = colors[0]
            self.frontRGB = colors[0]
            self.droid.set_back_LED_color(*self.backRGB)
            self.droid.set_front_LED_color(*self.frontRGB)
            return True

        if len(colors) == 0:
            changed = False
            for i in range(len(words) - 2):
                try:
                    rgb = (int(words[i]), int(words[i+1]), int(words[i+2]))
                    self.backRGB = rgb
                    self.frontRGB = rgb
                    self.droid.set_back_LED_color(*self.backRGB)
                    self.droid.set_front_LED_color(*self.frontRGB)
                    changed = True
                except ValueError:
                    continue
            if changed:
                return True

        if "off" in command or "out" in command or "minimum" in command:
            self.holoProjectorIntensity = 0
            self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            self.logicDisplayIntensity = 0
            self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            self.backRGB = (0, 0, 0)
            self.frontRGB = (0, 0, 0)
            self.droid.set_back_LED_color(*self.backRGB)
            self.droid.set_front_LED_color(*self.frontRGB)
            return True
        if "dim" in command:
            self.holoProjectorIntensity = self.holoProjectorIntensity / 2
            self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            self.logicDisplayIntensity = self.logicDisplayIntensity / 2
            self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            return True
        if "on" in command or "maximum" in command:
            self.holoProjectorIntensity = 1
            self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            self.logicDisplayIntensity = 1
            self.droid.set_logic_display_intensity(self.logicDisplayIntensity)
            return True
        if "blink" in command:
            self.droid.set_holo_projector_intensity((self.holoProjectorIntensity + 1)%2)
            self.droid.set_logic_display_intensity((self.holoProjectorIntensity + 1)%2)
            time.sleep(0.3)
            self.droid.set_holo_projector_intensity(self.holoProjectorIntensity)
            self.droid.set_logic_display_intensity(self.logicDisplayIntensity)  
            return True

        return False      

    def animationParser(self, command):
        print("animationParser has not yet been initialized.")

    def headParser(self, command):
        print("headParser has not yet been initialized.")

    def gridParser(self, command):
        print("gridParser has not yet been initialized.")

    def flash_colors(self, colors, seconds = 1, front = True):
        if front:
            for color in colors:
                self.droid.set_front_LED_color(*color)
                time.sleep(seconds)
        else:
            for color in colors:
                self.droid.set_back_LED_color(*color)
                time.sleep(seconds)











categories = ["state", "direction", "light", "animation", "head", "grid"]

indexToTrainingSentence = {}

i = 0
for category in categories:
    sentenceType = globals()[category + "Sentences"]
    for sentence in sentenceType:
        indexToTrainingSentence[i] = (sentence, category)
        i += 1

sentenceEmbeddings = np.zeros((len(indexToTrainingSentence), vectors.dim))

for i in range(len(indexToTrainingSentence)):
    sentence = indexToTrainingSentence[i][0]
    sentenceEmbedding = calcPhraseEmbedding(sentence)

    sentenceEmbeddings[i, :] = sentenceEmbedding

print("Welcome to CommandDroid, where we will try our best to understand what you want our R2D2 to do.")
print("In this environment, type 'exit()' to quit.")
print("***********************************************")

animateR2 = AnimateR2('D2-152E')

while(True):
    command = input("You: ")
    if command.strip() == "exit()":
        print("Have a nice day!")
        break

    commandEmbedding = calcPhraseEmbedding(command)

    closestSentences = rankSentences(commandEmbedding, sentenceEmbeddings)
    if cosineSim(commandEmbedding, sentenceEmbeddings[closestSentences[0], :]) < 0.85:
        print(animateR2.name + ": I could not understand your command.")
        continue

    commandType = getCommandType(categories, closestSentences, indexToTrainingSentence)
    result = getattr(animateR2, commandType + "Parser")(command.lower())
    if result:
        print(animateR2.name + ": Done executing "  + commandType + " command.")
    else:
        print(animateR2.name + ": I could not understand your " + commandType + " command.")