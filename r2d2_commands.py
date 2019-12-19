from pymagnitude import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from graph import *
from a_star import *
from client import DroidClient
from word2number import w2n
import random, time, csv, re
import numpy as np

# Change this path to where you put your project folder
path = "/Users/calchen/Desktop/sphero-project/"
# Change this path to / name of your .magnitude file.
# Do not merge this line with the previous line.
vectors = Magnitude(path + "vectors/word2vecRetrofitted.magnitude")

############################################################
# Helper Function
############################################################
# Load and process training sentences
def loadTrainingSentences(file_path):
    commandTypeDict = {}
    with open(file_path, "r") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if len(line.strip()) == 0 or "##" == line.strip()[0:2]:
                continue
            commandType, command = line.split(" :: ")
            if commandType not in commandTypeDict:
                commandTypeDict[commandType] = [command]
            else:
                commandTypeDict[commandType].append(command)
    return commandTypeDict

############################################################
# Classification / Intent Detection
############################################################
# Calculate the cosine similarity between two vectors
def cosineSim(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

# Given a sentence, calculate its embedding
def calcPhraseEmbedding(sentence):
    words = re.split("\W+", sentence)
    words = [x.lower() for x in words if x.lower() not in ["", "a", "an", "the", "is"]]
    if "?" in sentence:
        words.append("?")
    return vectors.query(words).sum(axis = 0) / len(words)

# Given a sentence embedding, rank training sentence embeddings
# according to their similarity to the given sentence embedding
def rankSentences(commandEmbedding, sentenceEmbeddings):
    sortList = []
    for i in range(sentenceEmbeddings.shape[0]):
        similarity = cosineSim(commandEmbedding, sentenceEmbeddings[i, :])
        sortList.append((i, similarity))
    similarSentences = sorted(sortList, key = lambda x: x[1], reverse = True)
    return [x[0] for x in similarSentences]

# Classify a given sentence
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
    return max(commandDict, key = commandDict.get)

############################################################
# Parsers / Slot Filling
############################################################
class Robot:
    def __init__(self, droidID, wordSimilarityCutoff, voice):
        # Initialize instance variables
        self.createSentenceEmbeddings()
        self.droid = DroidClient()
        self.name = "R2"
        self.wordSimilarityCutoff = wordSimilarityCutoff
        self.holoProjectorIntensity = 0
        self.logicDisplayIntensity = 0
        self.frontRGB = (0, 0, 0)
        self.backRGB = (0, 0, 0)
        self.voice = voice
        self.grid = [[]]
        self.speed = 0.5
        self.pos = (-1, -1)
        self.objCoord = dict()

        # Load and process color data
        self.colorToRGB = {}
        with open(path + "data/colors.csv") as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ",")
            for row in readCSV:
                self.colorToRGB[row[0]] = (int(row[2]), int(row[3]), int(row[4]))

        # Connect to robot
        connected = self.droid.connect_to_droid(droidID)
        while not connected:
            connected = self.droid.connect_to_droid(droidID)

    # Create training sentence embeddings
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

    # Process user's input command
    def inputCommand(self, command):
        commandEmbedding = calcPhraseEmbedding(command)
        closestSentences = rankSentences(commandEmbedding, self.sentenceEmbeddings)

        # print(self.indexToTrainingSentence[closestSentences[0]][0])
        # print(self.indexToTrainingSentence[closestSentences[1]][0])
        # print(self.indexToTrainingSentence[closestSentences[2]][0])
        # print(self.indexToTrainingSentence[closestSentences[3]][0])
        # print(self.indexToTrainingSentence[closestSentences[4]][0])

        print("Closet sentence was: " + self.indexToTrainingSentence[closestSentences[0]][0])
        print("Its cosine similarity to the command was: " + str(cosineSim(commandEmbedding, self.sentenceEmbeddings[closestSentences[0], :])))

        if cosineSim(commandEmbedding, self.sentenceEmbeddings[closestSentences[0], :]) < 0.84 and not self.voice:
            subcommand = input(self.name + ": I could not understand your command. Do you want to add this command to the training set? (yes/no): ")
            if "yes" in subcommand.lower():
                subcommand = input("What category do you want to add it to? Choices are state, direction, light, animation, head, or grid: ")
                subcommand = subcommand.lower()
                if subcommand in self.categories:
                    with open(path + "data/r2d2TrainingSentences.txt", "a") as the_file:
                        the_file.write(subcommand + "Sentences :: " + command + "\n")
                    print("Command added. Changes will be present on restart.")
                else:
                    print(subcommand + " not a valid category.")
            return

        commandType = getCommandType(self.categories, closestSentences, self.indexToTrainingSentence)
        result = getattr(self, commandType + "Parser")(command.lower())
        if result:
            print(self.name + ": Done executing "  + commandType + " command.")
        else:
            print(self.name + ": I could not understand your " + commandType + " command.")

    # Reset the robot to its initial orientation
    def reset(self):
        self.droid.roll(0, 0, 0)

    # Disconnect from the robot
    def disconnect(self):
        self.droid.disconnect()

    # Flash given light(s) in given color(s) for a given amount of time
    def flash_colors(self, colors, seconds = 1, front = True):
        if front:
            for color in colors:
                self.droid.set_front_LED_color(*color)
                time.sleep(seconds)
        else:
            for color in colors:
                self.droid.set_back_LED_color(*color)
                time.sleep(seconds)

    # If we can't detect the color for the light(s), ask the user for more info
    def askForColor(self, lightPosition = "both"):
        if lightPosition != "both":
            print("We detect that you want to change your " + lightPosition + " light, but could not find a color.")
        else:
            print("We parsed this as a light command, but could not find a color.")
        command = input("Do you want to input a color? (yes/no): ")
        color = False
        if "yes" in command.lower():
            print("You may have inputted a color, but it is not in our database or is mispelled. Please input a color or rgb tuple.")
            command = input("If you want to add the color to the database, input color_name (one string) :: rgb tuple: ")

            words = re.split("\W+", command)
            words = [x for x in words if x != ""]
            for word in words:
                if word in self.colorToRGB: color = self.colorToRGB[word]
            if len(words) == 4:
                try:
                    color = (int(words[1]), int(words[2]), int(words[3]))
                    colorName = words[0]
                    with open(path + "data/colors.csv", "a") as csvStorer:
                        csvStorer.write("\n" + colorName + ",R2D2 " + colorName + "," + words[1] + "," + words[2] + "," + words[3])
                    print(colorName + " added to database. It will be available on the next restart.")
                except ValueError:
                    superDumbVariable = 1
            elif len(words) == 3:
                try:
                    color = (int(words[0]), int(words[1]), int(words[2]))
                except ValueError:
                    superDumbVariable = 1
        return color

    # Parser for a light command
    def lightParser(self, command):
        # Slots for a light command
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

        # Front / back too similar
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

        words = re.split("\W+", command)
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

    # Execute a light command given its slots
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
                self.droid.set_holo_projector_intensity((self.holoProjectorIntensity + 1) % 2)
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
                self.droid.set_logic_display_intensity((self.logicDisplayIntensity + 1) % 2)
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
                if "red" in command:
                    slots["whichRGB"].append("red")
                if "green" in command:
                    slots["whichRGB"].append("green")
                if "blue" in command:
                    slots["whichRGB"].append("blue")

            if len(slots["whichRGB"]) == 0:
                return False

            if "red" in slots["whichRGB"]:
                for light in lights:
                    rgb = getattr(self, light + "RGB")
                    setattr(self, light + "RGB", (max(0, min(rgb[0] + rgb[0] * percent / 100, 255)), rgb[1], rgb[2]))
                    getattr(self.droid, "set_" + light + "_LED_color")(*getattr(self, light + "RGB"))
            if "green" in slots["whichRGB"]:
                for light in lights:
                    rgb = getattr(self, light + "RGB")
                    setattr(self, light + "RGB", (rgb[0], max(0, min(rgb[1] + rgb[1] * percent / 100, 255)), rgb[2]))
                    getattr(self.droid, "set_" + light + "_LED_color")(*getattr(self, light + "RGB"))
            if "blue" in slots["whichRGB"]:
                for light in lights:
                    rgb = getattr(self, light + "RGB")
                    setattr(self, light + "RGB", (rgb[0], rgb[1], max(0, min(rgb[2] + rgb[2] * percent / 100, 255))))
                    getattr(self.droid, "set_" + light + "_LED_color")(*getattr(self, light + "RGB"))

            return True

        if slots["add"] or slots["sub"]:
            lights = slots["lights"]

            if not slots["increment/seconds"]:
                command = input("Increment not found in command, please input amount to change by here: ")
                try:
                    slots["increment/seconds"] = int(command)
                except ValueError:
                    return False

            if slots["sub"]:
                slots["increment/seconds"] = -slots["increment/seconds"]

            increaseValue = slots["increment/seconds"]

            if len(slots["whichRGB"]) == 0:
                command = input("Did not find what values (red/blue/green) to change, input what values to change: ")
                if "red" in command:
                    slots["whichRGB"].append("red")
                if "green" in command:
                    slots["whichRGB"].append("green")
                if "blue" in command:
                    slots["whichRGB"].append("blue")

            if len(slots["whichRGB"]) == 0:
                return False

            if "red" in slots["whichRGB"]:
                for light in lights:
                    rgb = getattr(self, light + "RGB")
                    setattr(self, light + "RGB", (max(0, min(rgb[0] + increaseValue, 255)), rgb[1], rgb[2]))
                    getattr(self.droid, "set_" + light + "_LED_color")(*getattr(self, light + "RGB"))
            if "green" in slots["whichRGB"]:
                for light in lights:
                    rgb = getattr(self, light + "RGB")
                    setattr(self, light + "RGB", (rgb[0], max(0, min(rgb[1] + increaseValue, 255)), rgb[2]))
                    getattr(self.droid, "set_" + light + "_LED_color")(*getattr(self, light + "RGB"))
            if "blue" in slots["whichRGB"]:
                for light in lights:
                    rgb = getattr(self, light + "RGB")
                    setattr(self, light + "RGB", (rgb[0], rgb[1], max(0, min(rgb[2] + increaseValue, 255))))
                    getattr(self.droid, "set_" + light + "_LED_color")(*getattr(self, light + "RGB"))

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
                    if not color:
                        return False
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
            self.backRGB = tuple(x / 2 for x in self.backRGB)
            self.frontRGB = tuple(x / 2 for x in self.frontRGB)
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
            self.droid.set_holo_projector_intensity((self.holoProjectorIntensity + 1) % 2)
            self.droid.set_logic_display_intensity((self.holoProjectorIntensity + 1) % 2)
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

    # Parser for a directional command
    def directionParser(self, command):
        if re.search("(circle|donut)", command):
            if re.search("counter", command):
                for heading in range(360, 0, -30):
                    self.droid.roll(self.speed, heading % 360, 0.6)
            else:
                for heading in range(0, 360, 30):
                    self.droid.roll(self.speed, heading, 0.6)
            self.droid.roll(0, 0, 0)
            return True
        elif re.search("square", command):
            if re.search("counter", command):
                for heading in range(360, 0, -90):
                    self.droid.roll(0, heading % 360, 0)
                    time.sleep(0.35)
                    self.droid.roll(self.speed, heading % 360, 0.6)
            else:
                for heading in range(0, 360, 90):
                    self.droid.roll(0, heading, 0)
                    time.sleep(0.35)
                    self.droid.roll(self.speed, heading, 0.6)
            self.droid.roll(0, 0, 0)
            return True
        elif re.search("(speed|slow|faster|slower)", command):
            if re.search("(increase|faster|up)", command):
                if self.speed <= 0.75:
                    self.speed += 0.25
            else:
                if self.speed >= 0.5:
                    self.speed -= 0.25
            self.droid.animate(1)
            return True
        else:
            flag = False
            tokens = re.split("[^a-zA-Z]", command)
            for token in tokens:
                if token in {"up", "forward", "ahead", "straight", "north"}:
                    self.droid.roll(0, 0, 0)
                    time.sleep(0.35)
                    self.droid.roll(self.speed, 0, 0.6)
                    flag = True
                elif token in {"down", "back", "south"}:
                    self.droid.roll(0, 180, 0)
                    time.sleep(0.35)
                    self.droid.roll(self.speed, 180, 0.6)
                    flag = True
                elif token in {"left", "west"}:
                    self.droid.roll(0, 270, 0)
                    time.sleep(0.35)
                    self.droid.roll(self.speed, 270, 0.6)
                    flag = True
                elif token in {"right", "east"}:
                    self.droid.roll(0, 90, 0)
                    time.sleep(0.35)
                    self.droid.roll(self.speed, 90, 0.6)
                    flag = True
            self.droid.roll(0, 0, 0)
            return flag

    # Parser for a animation command
    def animationParser(self, command):
        if re.search("fall", command):
            self.droid.animate(14)
            return True
        elif re.search("run away", command):
            self.droid.animate(19)
            return True
        elif re.search("(dance|move)", command):
            self.droid.animate(20)
            return True

        elif re.search("(sing|sound|noise)", command):
            self.droid.play_sound(3)
            return True
        elif re.search("scream", command):
            self.droid.play_sound(7)
            return True

        return False

    # Parser for a head command
    def headParser(self, command):
        if re.search("(forward|ahead|straight|front)", command):
            self.droid.rotate_head(0)
            return True
        elif re.search("left", command):
            self.droid.rotate_head(-90)
            return True
        elif re.search("right", command):
            self.droid.rotate_head(90)
            return True
        elif re.search("(behind|back)", command):
            self.droid.rotate_head(180)
            return True
        return False

    # Parser for a state command
    def stateParser(self, command):
        if re.search("color", command):
            if re.search("(front|forward)", command):
                print("****************************************")
                print(self.frontRGB)
                print("****************************************")
            elif re.search("(back|rear|rare)", command):
                print("****************************************")
                print(self.backRGB)
                print("****************************************")
            else:
                print("****************************************")
                print(self.frontRGB)
                print(self.backRGB)
                print("****************************************")
            return True
        elif re.search("(name|call)", command):
            if re.search("(want|wanna).*you", command):
                if self.extractName(command) == "":
                    print("****************************************")
                    print("You didn't give me a name!")
                    print("****************************************")
                    self.droid.play_sound(7)
                    return False
                else:
                    self.name = self.extractName(command)
            else:
                print("****************************************")
                print(self.name)
                print("****************************************")
            return True
        elif re.search("(power|battery)", command):
            print("****************************************")
            self.droid.battery()
            print("****************************************")
            return True
        return False

    def extractName(self, command):
        words = re.split("[^a-zA-Z]", command)
        ret = ""
        activated = False
        for word in words:
            if activated:
                ret += word + " "
            if word == "you":
                activated = True
        if len(ret) > 0 and ret[-1] == " ":
            return ret[:-1]
        else:
            return ret

    # Parser for a grid command
    def gridParser(self, command):
        # Convert the words to lowercase
        # Convert numbers as words to ints
        # Remove duplicate spaces
        words = re.split("(x|[^a-zA-Z0-9])", command.lower())
        command = ""
        for word in words:
            try:
                command += str(w2n.word_to_num(word)) + " "
            except:
                command += word + " "
        command = re.sub(" +", " ", command)
        print(command)

        if re.search("(\d+|to).*(x|by).*(\d+|to) grid", command):
            words = re.split("(x|[^a-zA-Z0-9])", command)
            x, y = self.extractCoord(words)
            self.grid = [["" for col in range(y)] for row in range(x)]
            print("****************************************")
            for row in self.grid:
                print(row)
            print("****************************************")
            self.droid.animate(1)
            return True
        elif re.search("(s|re) .+ at.*(\d+|to).*(\d+|to)", command):
            words = re.split("[^a-zA-Z0-9]", command)
            x, y = self.extractCoord(words)
            obj = self.extractObj(words)
            if len(self.grid) == 0 or len(self.grid[0]) == 0:
                print("****************************************")
                print("Grid is not initialized yet!")
                print("****************************************")
                self.droid.play_sound(7)
                return False
            if x < 0 or x >= len(self.grid) or y < 0 or y >= len(self.grid[0]):
                print("****************************************")
                print("Coordinate is out of grid!")
                print("****************************************")
                self.droid.play_sound(7)
                return False
            if obj == "":
                print("****************************************")
                print("You didn't specify what the obstacle(s) is/are!")
                print("****************************************")
                self.droid.play_sound(7)
                return False
            self.grid[x][y] = obj
            self.objCoord[obj] = (x, y)
            print("****************************************")
            for row in self.grid:
                print(row)
            print("****************************************")
            self.droid.animate(1)
            return True
        elif re.search("you.*re.*at.*(\d+|to).*(\d+|to)", command):
            arr = re.split("[^a-zA-Z0-9]", command)
            x, y = self.extractCoord(arr)
            if len(self.grid) == 0 or len(self.grid[0]) == 0:
                print("****************************************")
                print("Grid is not initialized yet!")
                print("****************************************")
                self.droid.play_sound(7)
                return False
            if x < 0 or x >= len(self.grid) or y < 0 or y >= len(self.grid[0]):
                print("****************************************")
                print("Coordinate is out of grid!")
                print("****************************************")
                self.droid.play_sound(7)
                return False
            self.pos = (x, y)
            self.grid[x][y] = "you"
            self.objCoord["you"] = (x, y)
            print("****************************************")
            for row in self.grid:
                print(row)
            print("****************************************")
            self.droid.animate(1)
            return True
        elif re.search("go.*to", command):
            # Replace obj with its coordinates
            for obj in self.objCoord:
                if re.search(obj, command):
                    x, y = self.objCoord[obj]
                    print(x)
                    print(y)
                    if re.search("(left|west)", command):
                        y -= 1
                    elif re.search("(right|east)", command):
                        y += 1
                    elif re.search("(below|south|bottom)", command):
                        x -= 1
                    elif re.search("(above|north|top)", command):
                        x += 1
                    command = "go to " + str(x) + " , " + str(y)
                    print(command)
                    break
            if re.search("(\d+|to).*(\d+|to)", command):
                arr = re.split("[^a-zA-Z0-9]", command)
                x, y = self.extractCoord(arr)
                if len(self.grid) == 0 or len(self.grid[0]) == 0:
                    print("****************************************")
                    print("Grid is not initialized yet!")
                    print("****************************************")
                    self.droid.play_sound(7)
                    return False
                if x < 0 or x >= len(self.grid) or y < 0 or y >= len(self.grid[0]):
                    print("****************************************")
                    print("Coordinate is out of grid!")
                    print("****************************************")
                    self.droid.play_sound(7)
                    return False
                if self.pos == (-1, -1):
                    print("****************************************")
                    print("Current position hasn't been specified!")
                    print("****************************************")
                    self.droid.play_sound(7)
                    return False
                target = (x, y)
                if target == self.pos:
                    print("****************************************")
                    print("You are already there!")
                    print("****************************************")
                    self.droid.animate(1)
                    return True
                self.grid[x][y] = "target"
                print("****************************************")
                for row in self.grid:
                    print(row)
                print("****************************************")
                # Create a graph and calculate the moves
                G = Graph(self.grid)
                moves = A_star(G, self.pos, target, manhattan_distance_heuristic)
                if moves is None:
                    print("****************************************")
                    print("Impossible to get to the target!")
                    print("****************************************")
                    self.grid[x][y] = ""
                    self.droid.play_sound(7)
                    return False
                else:
                    print("****************************************")
                    print(moves)
                    print("****************************************")
                # Store the initial position
                init_x, init_y = self.pos
                for i in range(1, len(moves)):
                    if moves[i][1] > moves[i - 1][1]:
                        # Move right
                        self.droid.roll(0, 90, 0)
                        time.sleep(0.35)
                        self.droid.roll(self.speed, 90, 0.6)
                    elif moves[i][1] < moves[i - 1][1]:
                        # Move left
                        self.droid.roll(0, 270, 0)
                        time.sleep(0.35)
                        self.droid.roll(self.speed, 270, 0.6)
                    elif moves[i][0] > moves[i - 1][0]:
                        # Move down
                        self.droid.roll(0, 180, 0)
                        time.sleep(0.35)
                        self.droid.roll(self.speed, 180, 0.6)
                    elif moves[i][0] < moves[i - 1][0]:
                        # Move up
                        self.droid.roll(0, 0, 0)
                        time.sleep(0.35)
                        self.droid.roll(self.speed, 0, 0.6)
                    self.pos = moves[i]
                self.grid[x][y] = "you"
                self.objCoord["you"] = (x, y)
                self.grid[init_x][init_y] = ""
                print("****************************************")
                for row in self.grid:
                    print(row)
                print("****************************************")
                self.droid.animate(1)
                return True
        self.droid.play_sound(7)
        return False

    # Given a sentence as an array of words, extract the coordinate as an array of ints
    def extractCoord(self, words):
        ret = []
        for i in range(len(words) - 1, -1, -1):
            word = words[i]
            if word.isdigit():
                ret = [int(word)] + ret
            if word in {"to", "too"}:
                ret = [2] + ret
            if len(ret) == 2:
                break
        return ret

    # Given a sentence as an array of words, extract the object as a string
    def extractObj(self, words):
        ret = ""
        activated = False
        for word in words:
            if word == "at":
                activated = False
            if activated and word not in {"a", "an", "the"}:
                ret += word + " "
            if word in {"s", "re", "is", "are"}:
                activated = True
        if len(ret) > 0 and ret[-1] == " ":
            return ret[:-1]
        else:
            return ret