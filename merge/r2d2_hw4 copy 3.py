############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Kuanhao Jiang"

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

	with open(file_path, 'r') as fin:
		for line in fin:
			line = line.rstrip('\n')
			if len(line.strip()) == 0 or "##" == line.strip()[0:2]:
				continue
			commandType, command = line.split(' :: ')
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

my_driving_sentences = ["Be ready for driving.",
"Stop moving.",
"Go towards South.",
"Turn around.",
"Double your speed.",
"Moving forward for 10 feet, then turn left.",
"Restart rolling.",
"Turn to your left.",
"Freeze.",
"Half your speed."]

my_light_sentences = ["Set the back color to be black.",
"Turn the front light yellow.",
"Turn off all lights.",
"Half the intensity on the holoemitter.",
"Set the RGB values on your lights to be 200, 100, 100.",
"Decrease the red intensity by 20%.",
"Turn off the front light.",
"Blink your logic display.",
"Lights out.",
"Change the back LED to yellow."]

my_head_sentences = ["Turn your head to face left.",
"Look back.",
"Face forward.",
"Rotate your head left for 30 degrees.",
"Turn your head to the opposite direction.",
"Face backward.",
"Look to the right.",
"Look behind.",
"Pay attention to your left.",
"Focus on your right."]

my_state_sentences =["How much power do you left?",
"Are you connected?",
"Are you ready to drive?",
"What is your speed right now?",
"Are you waddling?",
"What is the color of your front LED?",
"Are you awake?",
"What is the intensity of you holoemitter?",
"What is your logic display intensity?",
"What direction are you looking at?"]

my_connection_sentences =["Connect yourself to the server.",
"Disconnect yourself from the server.",
"Exit.",
"Scan for droids nearby.",
"Look for droids near you.",
"Detach yourself from the server.",
"Disengage yourself.",
"Link to the server.",
"Search for other droids.",
"Quit."]

my_stance_sentences=["Transition back to bipod.",
"Set stance to 2.",
"Change stance to tripod.",
"Put down the third wheel.",
"Stand on tiptoes.",
"Set to the normal stance.",
"Lift the third wheel.",
"Set your stance to tripod.",
"Begin waddling.",
"Stop waddling."]

my_animation_sentences =["Play a random sound.",
"Begin screaming.",
"Start laughing.",
"Show animation.",
"Collapse to the ground.",
"Fall down.",
"Make some noise.",
"Sing a song for me.",
"Perform an action.",
"Make a howling sound."]

my_grid_sentences = ["You are on a 10 by 10 square.",
"Find the shortest path from (0,0) to (3,5).",
"You are at position (3,7).",
"It's impossible to move up from (4,7).",
"There is an obstacle at (3,2).",
"Go to (6,2).",
"You are on a rectangle with height 5 and width 8.",
"There is a person at position 5, 7.",
"You cannot go through (3,6).",
"What is the shortest path to (4,6)?"]

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
	text = sentence.strip()
	text = text.lower()
	res = []
	item = ""
	for i in text:
		if i.isspace():
			if len(item) > 0:
				res.append(item)
			item = ""
		elif i in string.punctuation:
			if len(item) > 0:
				res.append(item)
			item = ""
		else:
			item += i
	if len(item) > 0:       
		res.append(item)
	return res

def cosineSimilarity(vector1, vector2):
	dot_prod = np.dot(vector1, vector2)
	two_norm1 = np.linalg.norm(vector1)
	two_norm2 = np.linalg.norm(vector2)
	return dot_prod / (two_norm1 * two_norm2)

categories = ["driving", "light", "head", "state", "connection", "stance", "animation", "grid"]

class WordEmbeddings:

	def __init__(self, file_path):
		self.vectors = Magnitude(file_path) 
		self.training_data = {}
		self.clf = None
		# for i in my_driving_sentences:



	def calcSentenceEmbeddingBaseline(self, sentence):
		res = tokenize(sentence)
		v = self.vectors.query("cat")
		result = np.zeros(v.shape[0])
		for i in res:
			result += self.vectors.query(i)
		return result


	def update_training_date(self, X, y):
		m,n = X.shape
		for i in range(m):
			if tuple(X[i]) not in self.training_data:
				self.training_data[tuple(X[i])] = y


	def update_buildin_date(self):
		for i in my_driving_sentences:
			if not tuple(self.calcSentenceEmbeddingBaseline(i)) in self.training_data:
				self.training_data[tuple(self.calcSentenceEmbeddingBaseline(i))] = "driving"

		for i in my_light_sentences:
			if not tuple(self.calcSentenceEmbeddingBaseline(i)) in self.training_data:
				self.training_data[tuple(self.calcSentenceEmbeddingBaseline(i))] = "light"


		for i in my_head_sentences:
			if not tuple(self.calcSentenceEmbeddingBaseline(i)) in self.training_data:
				self.training_data[tuple(self.calcSentenceEmbeddingBaseline(i))] = "head"


		for i in my_state_sentences:
			if not tuple(self.calcSentenceEmbeddingBaseline(i)) in self.training_data:
				self.training_data[tuple(self.calcSentenceEmbeddingBaseline(i))] = "state"		


		for i in my_connection_sentences:
			if not tuple(self.calcSentenceEmbeddingBaseline(i)) in self.training_data:
				self.training_data[tuple(self.calcSentenceEmbeddingBaseline(i))] = "connection"	


		for i in my_stance_sentences:
			if not tuple(self.calcSentenceEmbeddingBaseline(i)) in self.training_data:
				self.training_data[tuple(self.calcSentenceEmbeddingBaseline(i))] = "stance"	

		for i in my_animation_sentences:
			if not tuple(self.calcSentenceEmbeddingBaseline(i)) in self.training_data:
				self.training_data[tuple(self.calcSentenceEmbeddingBaseline(i))] = "animation"	


		for i in my_grid_sentences:
			if not tuple(self.calcSentenceEmbeddingBaseline(i)) in self.training_data:
				self.training_data[tuple(self.calcSentenceEmbeddingBaseline(i))] = "grid"	

	def sentenceToEmbeddings(self, commandTypeToSentences):
		'''Returns a tuple of sentence embeddings and an index-to-(category, sentence)
		dictionary.

		Inputs:
			commandTypeToSentences: A dictionary in the form returned by
			loadTrainingSentences. Each key is a string '[category]' which
			maps to a list of the sentences belonging to that category.

		Let m = number of sentences.
		Let n = dimension of vectors.

		Returns: a tuple (sentenceEmbeddings, indexToSentence)
			sentenceEmbeddings: A mxn numpy array where m[i:] containes the embedding
			for sentence i.

			indexToSentence: A dictionary with key: index i, value: (category, sentence).
		'''
		sentenceEmbeddings = []
		indexToSentence = {}

		index = 0
		for key in commandTypeToSentences:
			sentences = commandTypeToSentences[key]
			for sentence in sentences:
				sentenceEmbeddings.append(self.calcSentenceEmbeddingBaseline(sentence))
				indexToSentence[index] = (sentence, key)
				index+=1

		sentenceEmbeddings = np.array(sentenceEmbeddings)
		return sentenceEmbeddings, indexToSentence



	def closestSentence(self, sentence, sentenceEmbeddings):
		'''Returns the index of the closest sentence to the input, 'sentence'.

		Inputs:
			sentence: A sentence

			sentenceEmbeddings: An mxn numpy array, where m is the total number
			of sentences and n is the dimension of the vectors.

		Returns:
			an integer i, where i is the row index in sentenceEmbeddings 
			that contains the closest sentence to the input
		'''
		m,n = sentenceEmbeddings.shape
		opt_index = 0
		opt_val = 2

		vec1 = self.calcSentenceEmbeddingBaseline(sentence)

		for i in range(m):
			val = cosineSimilarity(sentenceEmbeddings[i,:], vec1)
			if val < opt_val:
				opt_val = val
				opt_index = i

		return opt_index



	def getCategory(self, sentence, file_path):
		'''Returns the supposed category of 'sentence'.

		Inputs:
			sentence: A sentence

			file_path: path to a file containing r2d2 commands

		Returns:
			a string 'command', where 'command' is the category that the sentence
			should belong to.
		'''
		self.update_buildin_date()
		# self.update_training_date()

		training_sentences_dict = loadTrainingSentences(file_path)
		sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(training_sentences_dict)


		this_sentence_embedding = self.calcSentenceEmbeddingBaseline(sentence)


		m,n = sentenceEmbeddings.shape

		for i in self.training_data:
			sentenceEmbeddings = np.concatenate((sentenceEmbeddings, np.array([list(i)])), axis  = 0)
			# sentenceEmbeddings.append(np.array(list(i)))
			indexToSentence[m] = ("", self.training_data[i])
			m+=1



		m,n = sentenceEmbeddings.shape

		res = []
		for i in range(m):
			res.append(cosineSimilarity(sentenceEmbeddings[i], this_sentence_embedding))


		k = 10
		top_indices = sorted(range(len(res)), key = lambda i : res[i], reverse = True)[:k]

		if res[top_indices[0]] < 0.78:
			return "no"

		votes = {"driving":0, "light":0, "head":0, "state":0, "connection":0, "stance":0, "animation":0, "grid":0}
		for i in top_indices:
			votes[indexToSentence[i][1]] += res[i]
			# votes[indexToSentence[i][1]] += 1


		max_key = None
		max_vote = -1

		for key in votes:
			if votes[key] > max_vote:
				max_vote = votes[key]
				max_key = key
		return max_key






	def accuracy(self, training_file_path, dev_file_path):
		'''Returns the accuracy of your implementation of getCategory

		Inputs:
			training_file_path: path to training set

			dev_file_path: path to development set

		Let c = number of correctly categorized sentences in the development set.
		Let s = total number of sentences in the development set.

		Returns:
			A float equal to c/s.
		'''
		# train_dict = loadTrainingSentences(training_file_path)
		dev_dict = loadTrainingSentences(dev_file_path)

		correct = 0
		count = 0

		for key in dev_dict:
			for sentence in dev_dict[key]:
				prediction = self.getCategory(sentence, training_file_path)
				if prediction == key:
					correct+=1
				count+=1
		return correct / count




	############################################################
	# Section 3: Slot filling
	############################################################

	def lightParser(self, command):
		'''Slots for light command
		The slot "lights" can have any combination of "front"/"back"
		'''
		slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

		### YOUR CODE HERE ###

		return slots

	def drivingParser(self, command):
		'''Slots for driving commands
		Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
		You may ignore special cases such as "make a left before you come back"
		'''
		slots = {"increase": False, "decrease": False, "directions": []}

		### YOUR CODE HERE ###

		return slots




# X = WordEmbeddings("GoogleNews-vectors-negative300.magnitude")
# print(X.accuracy("data/r2d2TrainingSentences.txt", "data/r2d2DevelopmentSentences.txt"))

# X = WordEmbeddings("GoogleNews-vectors-negative300.magnitude")
# sentenceEmbeddings, _ = X.sentenceToEmbeddings(loadTrainingSentences("data/r2d2TrainingSentences.txt"))
# print("???")
# print(X.closestSentence("Lights on.", sentenceEmbeddings))

############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 10

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
