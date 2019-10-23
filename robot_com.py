"""
Allow user to control robot using natural English language via command line IO
"""

import random
import re
from client import DroidClient
from r2d2_commands import *

__author__ = "Zhenghua (Calvin) Chen"
__author__ = "John Zhang"

# Uncomment the code below for command line IO, remember to recomment if you are using voice IO

def main():
    print("Welcome to CommandDroid, where we will try our best to understand what you want our R2D2 to do.")
    print("In this environment, type 'exit', 'quit', 'bye', or 'goodbye' to quit.")
    print("***********************************************")

    # Replace this with your own robot serial ID
    robot = AnimateR2('D2-152E', 0.70, False)

    while(True):
        command = input("You: ")
        command = command.strip()
        if re.search(r"\b(exit|quit|bye|goodbye)\b", command, re.I):
            break

        flag = False
        tokens = re.split("[^a-zA-Z]", command.lower())
        if len(tokens) == 0:
            print("Please type something!")
            continue
        for token in tokens:
            if token in {"dance", "move", "moves"}:
                robot.animate()
                flag = True
            elif token in {"sing", "sound", "sounds", "noise", "noises"}:
                robot.play_sound()
                flag = True
        if not flag:
            robot.inputCommand(command)
    robot.disconnect()

if __name__ == "__main__":
    main()
