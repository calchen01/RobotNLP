"""
Allow user to control robot using natural English language via command line IO
"""

import random, re
from client import DroidClient
from r2d2_commands import *

def main():
    # 1st param: replace this with your own robot ID
    # 2nd param: wordSimilarityCutoff, range: 0.0 - 1.0. A higher value means we are more
    #  confident about the prediction but it also rejects sentences which we are less
    #  confident about
    # 3rd param: voiceIO?
    robot = Robot('D2-F75E', 0.70, False)

    while True:
        cmd = input("Please enter your instruction: ").lower()
        if re.search(".*(exit|quit|bye|goodbye).*", cmd):
            print('Exiting...')
            break
        if len(cmd) == 0:
            print("Please enter something")
        else:
            robot.inputCommand(cmd)
    # Disconnect the robot
    robot.disconnect()

if __name__ == "__main__":
    main()