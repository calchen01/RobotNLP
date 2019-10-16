"""
Allow user to control robot using natural English language via command line IO
"""

import random
import re
import time
from client import DroidClient
from matplotlib import colors

__author__ = "Zhenghua (Calvin) Chen"

class Robot(object):

    def __init__(self, robot_id):
        self.robot = DroidClient()
        connected = self.robot.connect_to_droid(robot_id)
        while not connected:
            connected = self.robot.connect_to_droid(robot_id)
        self.dirMap = {
            "up": 0,
            "right": 90,
            "down": 180,
            "left": 270
        }
        
    def roll(self, heading):
        self.robot.roll(0, self.dirMap.get(heading), 0)
        time.sleep(0.35)
        self.robot.roll(1, self.dirMap.get(heading), 0.62)

    def set_front_LED_color(self, tokens):
        for token in tokens:
            if colors.is_color_like(token):
                rgb = colors.to_rgba(token)
                self.robot.set_front_LED_color(rgb[0] * 255, rgb[1] * 255, rgb[2] * 255)
                break
            elif token in {"kill", "off"}:
                self.robot.set_front_LED_color(0, 0, 0)
                break

    def set_back_LED_color(self, tokens):
        for token in tokens:
            if colors.is_color_like(token):
                rgb = colors.to_rgba(token)
                self.robot.set_back_LED_color(rgb[0] * 255, rgb[1] * 255, rgb[2] * 255)
                break
            elif token in {"kill", "off"}:
                self.robot.set_back_LED_color(0, 0, 0)
                break

    def animate(self):
        # 55 animations available
        self.robot.animate(random.randint(1, 55))

    def play_sound(self):
        # 49 sounds available
        self.robot.play_sound(random.randint(1, 49))

    def reset(self):
        self.robot.roll(0, 0, 0)

    def disconnect(self):
        self.robot.disconnect()

# Uncomment the code below for command line IO, remember to recomment if you are using voice IO
'''
def main():
    # Replace this with your own robot serial ID
    robot = Robot("XX-XXXX")
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
            for token in tokens:
                if token in {"up", "forward", "ahead", "straight"}:
                    robot.roll("up")
                elif token in {"down", "back"}:
                    robot.roll("down")
                elif token in {"left", "right"}:
                    robot.roll(token)
                elif token in {"dance", "move", "moves"}:
                    robot.animate()
                elif token in {"sing", "sound", "sounds", "noise", "noises"}:
                    robot.play_sound()
    robot.reset()
    robot.disconnect()
            
if __name__ == '__main__':
    main()
'''