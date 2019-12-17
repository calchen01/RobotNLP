############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Sae Yeon Chung"

############################################################
# Imports
############################################################

from pymagnitude import *
import numpy as np
import string
import re
from sklearn.linear_model import LogisticRegression
import heapq
import statistics

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
my_driving_sentences = ['Turn around 90 degrees.',
						'Do not stop!',
						'Change speed to 100%.',
						'Move west.',
						'Decrease speed by 25%.',
						'Start rolling heading 30 degrees.',
						'Go forward for 5 feet.',
						'South is at 90 degrees.',
						'Go forward for 3 seconds.',
						'Run away as fast as you can!',
						"Go forward for 3 feet, then turn left.",
						"North is at heading 60 degrees.",
						"Go North direction.",
						"Go East direction.",
						"Move South-by-southeast",
						"Run away droid!",
						"Turn to heading 45 degrees.",
						"Reset heading to zero degree",
						"Turn to face South.",
						"Start rolling backward.",
						"Increase your speed by 75%.",
						"Turn to your left.",
						"Stop moving",
						"Stop driving",
						"Moving stop",
						"Set your moving speed to be 0.",
						"Change the speed to 20%",
						"Move around"]
my_light_sentences = ["Change the intensity on the holoemitter to maximum.",
						"Turn off the holoemitter.",
						"Blink your logic display.",
						"Change the back LED to green.",
						"Turn your back light green.",
						"Dim your lights holoemitter.",
						"Turn off all your lights.",
						"Lights out.",
						"Set the RGB values on your lights to be 255,0,0.",
						"Add 100 to the red value of your front LED.",
						"Increase the blue value of your back LED by 50%.",
						"Display the following colors for 2 seconds each: red, orange, yellow, green, blue, purple.",
						"Change the color on both LEDs to be green."]
my_head_sentences = ["Turn your head to face forward.",
					 "Look behind you.",
					 'Rotate head 90 degrees',
					 'Rotate yourself towards west.',
					 'Turn towards west.',
					 'Turn towards south.',
					 'Turn around to face east.',
					 'Look to the left.',
					 'Look to the right.',
					 'Face your head to the left.'] 
my_state_sentences = ["What color is your front light?",
						"Tell me what color your front light is set to.",
						"Is your logic display on?",
						"What is your stance?",
						"What is your orientation?",
						"What direction are you facing?",
						"Are you standing on 2 feet or 3?",
						"What is your current heading?",
						"How much battery do you have left?",
						"What is your battery status?",
						"Are you driving right now?",
						"How fast are you going?",
						"What is your current speed?",
						"Is your back light red?",
						"Are you awake?"]
my_connection_sentences = ["Connect D2-55A2 to the server",
							"Are there any other droids nearby?",
							"Disconnect.",
							"Disconnect from the server.",
							'Are you connected to the server?',
							'Connect to the server',
							'Do you have network?',
							'Are you connected?',
							'Find droids',
							'Find droids close to you',
							'Check connection']
my_stance_sentences = ["Set your stance to be biped.",
						"Put down your third wheel.",
						"Put down your first wheel.",
						"Put down your second wheel.",
						"Stand on your tiptoes.",
						"Tilt your head",
						"Raise your hand",
						"Raise your leg",
						"Set waddle",
						"Set stance",
						"waddle",
						"totter",
						"todder",
						"teater",
						"wobble",
						"start to waddle"
						"start waddling",
						"begin waddling",
						"set your stance to waddle",
						"try to stand on your tiptoes",
						"move up and down on your toes",
						"rock from side to side on your toes",
						"imitate a duck's walk",
						"walk like a duck",
						"stop your waddle",
						"end your waddle",
						"don't waddle anymore",
						"stop waddling",
						"cease waddling",
						"stop standing on your toes",
						"stand still"
						"stop acting like a duck",
						"don't walk like a duck",
						"stop teetering like that"
						"put your feet flat on the ground"]
my_animation_sentences = ["Fall over",
							"Scream",
							'Sing',
							'Speak',
							"Make some noise",
							"Laugh",
							"Play an alarm",
							'Tell me a joke',
							'Roll over',
							'Make a circle',
							'Make some sound',
							'Play something']
my_grid_sentences = ["You are on a 4 by 5 grid.",
					"Each square is 1 foot large.",
					"You are at position (0,0).",
					"Go to position (3,3).",
					"There is an obstacle at position 2,1.",
					"There is a chair at position 3,3",
					"Go to the left of the chair.",
					"It's not possible to go from 2,2 to 2,3.",
					'Go to first row and third column.',
					'Move to the first cell'
					]
my_no_sentences = ['Execute.',
					'MacDonald\'s is good.',
					'Can I have a cheeseburger please.',
					'What is the square root of negative infinity?',
					'The car is in the parking lot.',
					'If you\'re happy and you know it, clap your hands.',
					'That\'s a lot of work on your plate.',
					'Humans are ok.',
					'That light over there is too bright.',
					'Show me your skills.',
					'Sally likes cake.',
					'I hate you.',
					'Trump can now do what he\'s wanted for two years.',
					'We have a lot of rain in June.',
					'Sometimes it is better to just walk away from things and go back to them later when you’re in a better frame of mind.',
					'I am never at home on Sundays.',
					'The attack on Pearl Harbor was a surprise preemptive military strike by the Imperial Japanese Navy Air Service upon the United States.',
					'Tom got a small piece of pie.',
					'I hear that Nancy is very pretty.']

############################################################
# Section 2: Intent Detection
############################################################

magnitudeFile = "google"

def tokenize(sentence):
	curr_word = ''
	l = []
	for c in sentence:
		if c in string.punctuation:
			sentence = sentence.replace(c, ' ')
	
	return sentence.lower().split()

def cosineSimilarity(vector1, vector2):
	return float(np.dot(vector1, vector2)) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

class WordEmbeddings:

	def __init__(self, file_path):
		self.vectors = Magnitude(file_path)
		self.vec_dim = self.vectors.query('dog').shape
		self.categories = ['driving', 'light', 'head', 'state', 'connection', 'stance', 'animation', 'grid', 'no']
		self.stop_words = ['the', 'a', 'to', 'not', 'do', 'don\'t'] 
		# self.stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

	def calcSentenceEmbeddingBaseline(self, sentence):
		tokens = tokenize(sentence)
		vec = np.zeros(self.vec_dim)
		if not tokens:
			return vec
		for t in tokens:
			cur_vec = self.vectors.query(t)
			vec += cur_vec
		return vec

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
		
		n = self.vec_dim
		m = 0
		sentences = []
		for k,v in commandTypeToSentences.items():
			m += len(v)
			for s in v:
				sentences.append((k, s))

		sentenceEmbeddings = np.zeros((m, n[0]))
		indexToSentence = {}
		for i, (category, s) in enumerate(sentences):
			sent_vec = self.calcSentenceEmbeddingBaseline(s)
			sentenceEmbeddings[i] = sent_vec
			indexToSentence[i] = (s, category)

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
		sent_vec = self.calcSentenceEmbeddingBaseline(sentence)

		min_dist, min_dist_vec_i = float('-inf'), -1
		for i in range(len(sentenceEmbeddings)):
			sim = cosineSimilarity(sent_vec, sentenceEmbeddings[i,:])
			if sim > min_dist:
				min_dist = sim
				min_dist_vec_i = i
		return min_dist_vec_i


	""" <START> Advanced helper methods """
	def calcSentenceEmbeddingStopwordRemoved(self, sentence):
		tokens = tokenize(sentence)
		filtered_sentence = [w for w in tokens if not w in self.stop_words] 

		vec = np.zeros(self.vec_dim)
		
		if not filtered_sentence:
			return vec
		for t in filtered_sentence:
			cur_vec = self.vectors.query(t)
			vec += cur_vec
		return vec

	def sentenceToEmbeddingsStopwordRemoved(self, commandTypeToSentences):
		n = self.vec_dim
		m = 0
		sentences = []
		for k,v in commandTypeToSentences.items():
			m += len(v)
			for s in v:
				sentences.append((k, s))

		sentenceEmbeddings = np.zeros((m, n[0]))
		indexToSentence = {}
		for i, (category, s) in enumerate(sentences):
			sent_vec = self.calcSentenceEmbeddingStopwordRemoved(s)
			sentenceEmbeddings[i] = sent_vec
			indexToSentence[i] = (s, category)

		return sentenceEmbeddings, indexToSentence

	# TODO: make this return k closest categories
	def closest_K_Sentences(self, sentence, sentenceEmbeddings, indexToSentence, k):

		sent_vec = self.calcSentenceEmbeddingStopwordRemoved(sentence)
		closest_categories = []

		for i in range(len(sentenceEmbeddings)):
			sim = cosineSimilarity(sent_vec, sentenceEmbeddings[i,:])
			s, category = indexToSentence[i]
			heapq.heappush(closest_categories, (-sim, (s, category)))

			if len(closest_categories) > k:
				closest_categories = closest_categories[:-1]

		return closest_categories

	""" <END> Advanced helper methods """

	def getCategory(self, sentence, file_path):
		'''Returns the supposed category of 'sentence'.

		Inputs:
			sentence: A sentence

			file_path: path to a file containing r2d2 commands

		Returns:
			a string 'command', where 'command' is the category that the sentence
			should belong to.
		'''
		commandTypeToSentences = loadTrainingSentences(file_path)
		commandTypeToSentences['driving'] += my_driving_sentences
		commandTypeToSentences['animation'] += my_animation_sentences
		commandTypeToSentences['light'] += my_light_sentences
		commandTypeToSentences['head'] += my_head_sentences
		commandTypeToSentences['state'] += my_state_sentences
		commandTypeToSentences['grid'] += my_grid_sentences
		commandTypeToSentences['no'] = my_no_sentences
		commandTypeToSentences['stance'] = my_stance_sentences
		commandTypeToSentences['connection'] = my_connection_sentences

		sentenceEmbeddings_train, indexToSentence_train = self.sentenceToEmbeddingsStopwordRemoved(commandTypeToSentences)
		closest_sentences_categories = self.closest_K_Sentences(sentence, sentenceEmbeddings_train, indexToSentence_train, 10)

		# get average sim for each category
		category_dict = {}
		over_threshold_found = False
		threshold = 0.69
		for (sim, (s, category)) in closest_sentences_categories:
			sim = -sim
			if sim >= threshold:
				over_threshold_found = True
			if category in category_dict:
				category_dict[category].append(sim)
			else:
				category_dict[category] = [sim]

		# print(category_dict)
		for (k, v) in category_dict.items():
			# max, avg, median, length
			category_dict[k] = [max(v), (sum(v) / len(v)), statistics.median(v), len(v)]

		# print()
		# print(sorted(category_dict.items(), key=lambda kv:kv[1], reverse=True))
		# print()
		
		sorted_category_dict = sorted(category_dict.items(), key=lambda kv:kv[1], reverse=True)
		max_category = sorted_category_dict[0][0]

		if len(sorted_category_dict) > 1:
			# diff between first & second is significant enough
			if sorted_category_dict[0][1][0] - sorted_category_dict[1][1][0] > 0.15:
				max_category = sorted_category_dict[0][0]
			elif sorted_category_dict[0][1][0] > 0.75:
				max_category = sorted_category_dict[0][0]
			else:
				max_avg, max_med, max_len = float('-inf'), float('-inf'), float('-inf')
				max_avg_cat, max_med_cat, max_len_cat = 'no', 'no', 'no'
				for (k, v) in sorted_category_dict:
					if v[1] > max_avg:
						max_avg = v[1]
						max_avg_cat = k
					if v[2] > max_med:
						max_med = v[2]
						max_med_cat = k
					if v[3] > max_len:
						max_len = v[3]
						max_len_cat = k

				max_category = max_med_cat

		# for (k, v) in sorted_category_dict:
		# 	print(k,v)
		# 	if v[0] > 0.77:
		# 		print('here!')
		# for (word, tag) in sent_tagged:


		if not over_threshold_found:
			max_category = 'no'

		# print('sentence: {}, max_category: {}'.format(sentence, max_category))
		return max_category

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
		sentenceEmbeddings_train, indexToSentence_train = self.sentenceToEmbeddingsStopwordRemoved(loadTrainingSentences(training_file_path))
		sentenceEmbeddings_dev, indexToSentence_dev = self.sentenceToEmbeddingsStopwordRemoved(loadTrainingSentences(dev_file_path))
		
		c = 0
		s = 0

		for i, (sentence, category) in indexToSentence_dev.items():
			estimate = self.getCategory(sentence, training_file_path)
			if estimate == category:
				c += 1
			s += 1
		return c/s


	############################################################
	# Section 3: Slot filling
	############################################################

	def lightParser(self, command):
		'''Slots for light command
		The slot "lights" can have any combination of "front"/"back"
		'''
		print('\'{}\''.format(command))
		slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

		### YOUR CODE HERE ###
		command_tokens = tokenize(command)
		command_sent = command.lower()

		for command in command_tokens:
			if 'holo' in command:
				print('holo in command')
				slots['holoEmit'] = True
			if 'logic' in command:
				print('logic in command')
				slots['logDisp'] = True
			if 'increase' in command or 'add' in command:
				print('add in command')
				slots['add'] = True
			if 'decrease' in command or 'sub' in command or 'reduce' in command:
				print('sub in command')
				slots['sub'] = True
			if 'off' == command or 'min' in command:
				print('off in command')
				if not slots['holoEmit'] and not slots['logDisp']:
					slots['off'] = True
			if 'on' == command or 'max' in command:
				print('on in command')
				if not slots['holoEmit'] and not slots['logDisp']:
					slots['on'] = True

		append_front, append_back = False, False
		if 'front' in command_sent or 'forward' in command_sent:
			append_front = True
		if 'back' in command_sent or 'rear' in command_sent:
			append_back = True
		if append_front:
			print('front in command')
			slots['lights'].append('front')
		if append_back:
			print('back in command')
			slots['lights'].append('back')
		if (not append_front) and (not append_back):
			slots['lights'].append('front')
			slots['lights'].append('back')
		if slots['holoEmit'] or slots['logDisp']:
			slots['lights'] = []

		print(slots)
		print()
		return slots

	def drivingParser(self, command):
		'''Slots for driving commands
		Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
		You may ignore special cases such as "make a left before you come back"
		'''
		print(command)
		slots = {"increase": False, "decrease": False, "directions": []}

		### YOUR CODE HERE ###
		command = command.lower()
		directions = ['forward', 'back', 'left', 'right']
		cardinal = {}
		cardinal['east'] = 'right'
		cardinal['west'] = 'left'
		cardinal['north'] = 'forward'
		cardinal['south'] = 'back'

		if 'increase' in command or 'add' in command:
			slots['increase'] = True
		if 'decrease' in command or 'sub' in command or 'reduce' in command:
			slots['decrease'] = True
		command_tokens = tokenize(command)
		for token in command_tokens:
			if token in directions:
				slots['directions'].append(token)
			if token in cardinal:
				slots['directions'].append(cardinal[token])

		return slots


if __name__ == '__main__':
	X = WordEmbeddings('/Users/Charlotte/Downloads/GoogleNews-vectors-negative300.magnitude')
	# print(X.accuracy("data/r2d2TrainingSentences.txt", "data/r2d2DevelopmentSentences.txt"))
	print(X.getCategory('move forward', 'data/r2d2TrainingSentences.txt'))
	# sentenceEmbeddings, _ = X.sentenceToEmbeddings(loadTrainingSentences("data/r2d2TrainingSentences.txt"))
	# print(X.closestSentence("Lights on.", sentenceEmbeddings))

############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 10

feedback_question_2 = """
H
"""

feedback_question_3 = """
T
"""
