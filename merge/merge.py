import os, re

'''
This program merges all training sentences into a single txt file
'''

# Change the directory below which contains all the files to be merged
directory = os.fsencode("/Users/John1999/temp_folder")

# Given a line, extract the category
def getCategory(line):
    i = -1
    while i < len(line):
        if line[i] == "=" or line[i] == " ":
            break
        i += 1
    return line[3:i-10] + "Sentences"

# Given a line, extract all the training sentences
def getSentences(line):
    indexes = []
    results = []
    for i in range(len(line)):
        if line[i] == "\"":
            indexes.append(i)
    for i in range(0, len(indexes)-1, 2):
        results.append(line[indexes[i] + 1 : indexes[i+1]])
    return results

# A map storing all parsed results {category: set(training sentences)}
store = dict() 

# Extract and parse all training sentences from every file in the given directory
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     # Modify according to inputs
     if filename.endswith(".py"):
        with open(filename) as fp:
            line = fp.readline()
            category = ""
            while line:
                # Modify according to inputs
                if re.search(".*sentences ?= ?\[", line):
                    category = getCategory(line)
                    if category != "" and category not in store:
                        store[category] = set()
                if category != "":
                    if len(getSentences(line)) > 0:
                        if "'" in line:
                            print(filename)
                            print(line)
                        for sentence in getSentences(line):
                            store[category].add(sentence)
                    if re.search(".*]", line):
                        category = ""
                line = fp.readline()

# Create the target file which contains processed results
fo = open("r2d2TrainingSentences.txt", "w")
for category in store:
    for sentence in store[category]:
        fo.write(category + " :: " + sentence + "\n")
    fo.write("\n")
# Close the target file
fo.close()
