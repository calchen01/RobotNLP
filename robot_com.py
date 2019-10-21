"""
Allow user to control robot using natural English language via command line IO
"""

import random
import re
import time
from client import DroidClient
from matplotlib import colors
from r2d2_commands import *

__author__ = "Zhenghua (Calvin) Chen"

# Uncomment the code below for command line IO, remember to recomment if you are using voice IO

def main():
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

    # Replace this with your own robot serial ID
    robot = Robot("D2-F75E")
    while True:
        print("\nEnter your instruction: ")
        cmd = input().lower()
        if re.search(r"\b(exit|quit|bye|goodbye)\b", cmd, re.I):
            break

        tokens = re.split("[^a-zA-Z]", cmd)
        if re.search(r"\b(light|lights|led|leds|backlight|backlights)\b", cmd, re.I):
            if re.search(r"\b(front|forward)\b", cmd, re.I):
                robot.set_front_LED_color(tokens)
            elif re.search(r"\b(back|rear|backlight|backlights)\b", cmd, re.I):
                robot.set_back_LED_color(tokens)
            else:
                robot.set_front_LED_color(tokens)
                robot.set_back_LED_color(tokens)
        else:
            flag = False
            for token in tokens:
                if token in {"up", "forward", "ahead", "straight"}:
                    robot.roll("up")
                    flag = True
                elif token in {"down", "back"}:
                    robot.roll("down")
                    flag = True
                elif token in {"left", "right"}:
                    robot.roll(token)
                    flag = True
                elif token in {"dance", "move", "moves"}:
                    robot.animate()
                    flag = True
                elif token in {"sing", "sound", "sounds", "noise", "noises"}:
                    robot.play_sound()
                    flag = True
            if flag:
                commandEmbedding = calcPhraseEmbedding(cmd)

                closestSentences = rankSentences(commandEmbedding, sentenceEmbeddings)
                if cosineSim(commandEmbedding, sentenceEmbeddings[closestSentences[0], :]) < 0.85:
                    print(robot.name + ": I could not understand your command.")
                    continue

                commandType = getCommandType(categories, closestSentences, indexToTrainingSentence)
                result = getattr(robot, commandType + "Parser")(cmd)
                if result:
                    print(robot.name + ": Done executing "  + commandType + " command.")
                else:
                    print(robot.name + ": I could not understand your " + commandType + " command.")

    robot.disconnect()

if __name__ == "__main__":
    main()
