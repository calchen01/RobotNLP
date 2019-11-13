import os

'''
This program merges all training sentences into a single txt file
'''

# Change the directory below which contains all txt files to be merged
directory = os.fsencode("/Users/calchen/Desktop/RobotNLP/merge")
store = dict()
store["stateSentences"] = set()
store["directionSentences"] = set()
store["lightSentences"] = set()
store["animationSentences"] = set()
store["headSentences"] = set()
store["gridSentences"] = set()

# Read & process all txt files
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".txt"): 
        with open(filename) as fp:
            line = fp.readline()
            while line:
                arr = line.strip().split("::")
                if len(arr) > 1:
                    if arr[0].strip() in store:
                        print("\"" + arr[0].strip() + "\" \"" + arr[1].strip() + "\"")
                        store[arr[0].strip()].add(arr[1].strip())
                line = fp.readline()

# Write the merged results into a single txt file
fo = open("r2d2TrainingSentences.txt", "w")
for category in store:
    if len(store[category]) > 0:
        for sentence in store[category]:
            fo.write(category + " :: " + sentence + "\n")
        fo.write("\n")
fo.close()