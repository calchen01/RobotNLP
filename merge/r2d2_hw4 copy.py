############################################################
# CIS 521: R2D2-Homework 4
############################################################

student_name = "Philippe Sawaya"

############################################################
# Imports
############################################################

from pymagnitude import *
import numpy as np
import string
import re
import collections


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
my_driving_sentences = ['Stop right now.',
                        'Go North-west.',
                        'Run forward!',
                        'Reset your heading to 0 degrees.',
                        'Rest your heading to 180 degrees.',
                        'Turn to your left.',
                        'Increase your speed.',
                        'Decrease your speed.',
                        'Turn to your right.',
                        'Turn backwards.']
my_light_sentences = ['Turn on the holoemitter.',
                      'Change any LED to green.',
                      'Turn on everything.',
                      'Lights on!',
                      'Display red and blue for three seconds each.',
                      'Change all LEDs to yellow.',
                      'Blink your logic display twice.',
                      'Subtract 50 from the green value of your back LED',
                      'Decrease the red value of your back LED by 20%',
                      'Make a light party!']
my_head_sentences = ['Look ahead.',
                      'Look straight forawrd.',
                      'Look ahead of you.',
                      'Turn your head backwards',
                      'Turn your head backward',
                      'Turn your head to face backward',
                      'Turn your head 360 degrees',
                      'Turn you head northwest',
                      'Turn your head to the left',
                      'Turn your head to the right']
my_state_sentences = ['What is the state of your logic display?',
                      'Are you asleep?',
                      'Is your back light green?',
                      'Where are you facing?',
                      'How much battery do you have?',
                      'Is your logic display off?',
                      'Tell me what direction you are facing.',
                      'Tell me whether you are driving.',
                      'Tell me whether you are driving or not.',
                      'What color is your back light set to?']
my_connection_sentences = ['Tell me if any droids are nearby.',
                      'Please disconnect.',
                      'Please disconnect from the server.',
                      'Connect droid D2-55A2 to the server.',
                      'How many droids are connected?',
                      'Tell me if droid D2-55A2 is connected.',
                      'Tell me if droid D2-55A2 is connected to the server.',
                      'Tell me how many droids are connected.',
                      'Tell me how many droids are connected to the server.',
                      'What number of droids are connected?']
my_stance_sentences = ['Become biped.',
                      'Set stance to biped.',
                      'Raise your third wheel.',
                      'Lower your third wheel.',
                      'Get down from your tiptoes.',
                      'Let your third wheel fall.',
                      'Bring up your third wheel.',
                      'Let your stance be biped.',
                      'Get up on your tiptoes.',
                      'Stand up onto your tip toes.']
my_animation_sentences = ['Scream loudly!',
                      'Make a lot of noise!',
                      'Make some sound.',
                      'Play a noise.',
                      'Laugh at my joke.',
                      'Fall on the floor.',
                      'Fall on the ground.',
                      'Get on the ground.',
                      'Play a scary a noise.',
                      'Play an alarming noise.']
my_grid_sentences = ['The grid you are on is 4 by 5.',
                      'The grid you are on has 1 foot squares',
                      'You are at the root position.',
                      'It is impossible to move forward.',
                      'There is an obstacle ahead of you.',
                      'Go to the position two ahead of you.',
                      'Go back by two positions.',
                      'The grid ends to the right of position (3,3)',
                      'You cannot move forward any more.',
                      'There is a chair right behind you.']


############################################################
# Section 2: Intent Detection
############################################################
magnitudeFile = "google"


def clean_corpus(corpus):
    cleaned = collections.defaultdict(list)
    for category, sentences in corpus.items():
        for sentence in sentences:
            cleaned[category].append(clean_sentence(sentence))
    return cleaned


def clean_sentence(text):
    if '?' in text:
        text = text.replace('?', ' question ')

    if '%' in text:
        text = text.replace('%', ' percent ')

    stopwords = {'the', 'a', 'your', 'of', 'to', 'an'}
    tokens = [t for t in tokenize(text) if t not in stopwords]
    return ' '.join(tokens)


def tokenize(text):
    for c in string.punctuation:
        text = text.replace(c, ' ')
    return text.strip().lower().split()


def cosineSimilarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# Source: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(v):
    e_x = np.exp(v - np.max(v))
    return e_x / e_x.sum(axis=0)


class WordEmbeddings:
    def __init__(self, file_path):
        self.vectors = Magnitude(file_path)

    def calcSentenceEmbeddingBaseline(self, sentence):
        embedding = np.zeros(self.vectors.dim, dtype=float)
        for word in tokenize(sentence):
            embedding += self.vectors.query(word)
        return embedding

    def sentenceToEmbeddings(self, commandTypeToSentences):
        embeddings, indices, i = [], {}, 0
        for category, sentences in commandTypeToSentences.items():
            for sentence in sentences:
                embeddings.append(self.calcSentenceEmbeddingBaseline(sentence))
                indices[i] = (sentence, category)
                i += 1

        # embeddings = np.stack(embeddings, 0) if embeddings else np.zeros(self.vectors.dim, dtype=float)
        return np.stack(embeddings, 0).astype(float), indices

    def closestSentence(self, sentence, sentenceEmbeddings):
        v1 = self.calcSentenceEmbeddingBaseline(sentence)
        i_max, max_similarity = 0, float('-inf')

        for i, v2 in enumerate(sentenceEmbeddings):
            similarity = cosineSimilarity(v1, v2)
            if similarity > max_similarity:
                i_max, max_similarity = i, similarity
        return i_max

    def getCategory(self, sentence, file_path, k=4, threshold=0.35):
        # Load corpus embeddings
        corpus = clean_corpus(loadTrainingSentences(file_path))
        embeddings, indices = self.sentenceToEmbeddings(corpus)

        # Compute distances to all sentences in corpus
        v1 = self.calcSentenceEmbeddingBaseline(clean_sentence(sentence))
        distances = []
        for i, v2 in enumerate(embeddings):
            distances.append((i, cosineSimilarity(v1, v2)))

        # Find k nearest neighbors
        neighbors = sorted(distances, key=lambda t: t[1], reverse=True)[:k]

        # One hot encode categories
        category_to_index = {category: int(i) for i, category in enumerate(corpus.keys())}
        index_to_category = {i: category for category, i in category_to_index.items()}
        category_scores = np.zeros(len(category_to_index), dtype=float)

        # Find category likelihoods from k nearest neighbors
        for i, similarity in neighbors:
            category = category_to_index[indices[i][1]]
            category_scores[category] += similarity
        category_scores = softmax(category_scores)

        # Predict category
        if np.amax(category_scores) < threshold:
            return 'no'
        else:
            return index_to_category[np.argmax(category_scores)]
        # if np.amax(category_scores) < threshold:
        #     return 'no', np.amax(category_scores)
        # else:
        #     return index_to_category[np.argmax(category_scores)], np.amax(category_scores)

    def accuracy(self, training_file_path, dev_file_path, k=4, t=0.4):
        n_correct = n_total = 0
        for y, sentences in loadTrainingSentences(dev_file_path).items():
            for x in sentences:
                # y_hat, score = self.getCategory(x, training_file_path, k=k, threshold=t)
                # y_hat = self.getCategory(x, training_file_path, k=k, threshold=t)
                y_hat = self.getCategory(x, training_file_path)
                if y_hat == y:
                    n_correct += 1
                    # print('{}: {} ({})'.format(y, y_hat, score))
                else:
                    pass
                    # print('\t{}: {} ({})'.format(y, y_hat, score))
                n_total += 1
        return n_correct / n_total

    ############################################################
    # Section 3: Slot filling
    ############################################################
    def lightParser(self, command):
        slots = {"holoEmit": False, "logDisp": False, "lights": [], "add": False, "sub": False, "off": False, "on": False}

        words = tokenize(clean_sentence(command))
        word_embeddings = [self.calcSentenceEmbeddingBaseline(word) for word in words]

        slot_embeddings = {'holoEmit': ['holoemitter'],
                           'logDisp': ['logic', 'display'],
                           'lights': ['lights', 'back', 'rear', 'front'],
                           'add': ['add', 'increase', 'raise', 'strengthen'],
                           'sub': ['subtract', 'decrease', 'lower', 'weaken', 'reduce'],
                           'off': ['off', 'min', 'minimum'],
                           'on': ['on', 'max', 'maximum']}
        # , 'red', 'periwinkle', 'chocolate', 'yellow', 'magenta', 'blue', 'green'
        slot_embeddings = {k: [self.calcSentenceEmbeddingBaseline(w) for w in v] for k, v in slot_embeddings.items()}
        slot_scores = {k: 0. for k in slots.keys()}

        for v_w in word_embeddings:
            for slot, slot_word_embeddings in slot_embeddings.items():
                similarity = max(cosineSimilarity(v_w, v_slot) for v_slot in slot_word_embeddings)
                slot_scores[slot] = max(slot_scores[slot], similarity)

        slots['holoEmit'] = slot_scores['holoEmit'] > slot_scores['logDisp'] and slot_scores['holoEmit'] > slot_scores['lights']
        slots['logDisp'] = slot_scores['logDisp'] > slot_scores['holoEmit'] and slot_scores['logDisp'] > slot_scores['lights']
        slots['lights'] = slot_scores['lights'] > slot_scores['holoEmit'] and slot_scores['lights'] > slot_scores['logDisp']

        slots['add'] = slot_scores['add'] > 0.5 and slot_scores['add'] > slot_scores['sub']
        slots['sub'] = slot_scores['sub'] > 0.5 and slot_scores['sub'] > slot_scores['add']

        slots['off'] = slot_scores['off'] > 0.3 and slot_scores['off'] > slot_scores['on'] and not slots['add']
        slots['on'] = slot_scores['on'] > 0.3 and slot_scores['on'] > slot_scores['off'] and not slots['sub']
        return slots

    def drivingParser(self, command):
        '''Slots for driving commands
        Directions should support sequential directional commands in one sentence, such as "go straight and turn left". 
        You may ignore special cases such as "make a left before you come back"
        '''
        slots = {"increase": False, "decrease": False, "directions": []}

        words = tokenize(clean_sentence(command))
        word_embeddings = [self.calcSentenceEmbeddingBaseline(word) for word in words]

        slot_embeddings = {'increase': ['add', 'increase', 'raise', 'strengthen'],
                           'decrease': ['subtract', 'decrease', 'lower', 'weaken', 'reduce']}
        direction_embeddings = {'forward': ['forward', 'north'],
                                'backward': ['backward', 'south'],
                                'left': ['left', 'west'],
                                'right': ['right', 'east']}

        slot_embeddings = {k: [self.calcSentenceEmbeddingBaseline(w) for w in v] for k, v in slot_embeddings.items()}
        direction_embeddings = {k: [self.calcSentenceEmbeddingBaseline(w) for w in v] for k, v in direction_embeddings.items()}
        slot_scores = {k: 0. for k in slots.keys()}

        for v_w in word_embeddings:
            for slot, slot_word_embeddings in slot_embeddings.items():
                similarity = max(cosineSimilarity(v_w, v_slot) for v_slot in slot_word_embeddings)
                slot_scores[slot] = max(slot_scores[slot], similarity)

            max_dir, max_sim = '', float('-inf')
            for direction, dir_embeddings in direction_embeddings.items():
                similarity = max(cosineSimilarity(v_w, v_dir) for v_dir in dir_embeddings)
                if similarity > max_sim:
                    max_dir, max_sim = direction, similarity
            if max_sim > 0.5:
                slots['directions'].append(max_dir)
            
        slots['increase'] = slot_scores['increase'] > slot_scores['decrease']
        slots['decrease'] = not slot_scores['increase']
        return slots


if __name__ == '__main__':
    trainingSentences = loadTrainingSentences("data/r2d2TrainingSentences.txt")
    X = WordEmbeddings("GoogleNews-vectors-negative300.magnitude")  # Change this to where you downloaded the file.
    sentenceEmbeddings, _ = X.sentenceToEmbeddings(loadTrainingSentences("data/r2d2TrainingSentences.txt"))
    # print(X.getCategory("Turn your lights green.", "data/r2d2TrainingSentences.txt"))
    # print(X.getCategory("Drive forward for two feet.", "data/r2d2TrainingSentences.txt"))
    # print(X.getCategory("Do not laugh at me.", "data/r2d2TrainingSentences.txt"))
    # X.accuracy("data/r2d2TrainingSentences.txt", "data/r2d2DevelopmentSentences.txt", k=7)

    # Search for best k, fixed threshold
    # print('\n')
    # best_k, best_acc = 0, float('-inf')
    # for k in range(2, 14):
    #     accuracy = X.accuracy("data/r2d2TrainingSentences.txt", "data/r2d2DevelopmentSentences.txt", k=k, t=0.4)
    #     print('\t{}: {}'.format(k, accuracy))
    #     if accuracy > best_acc:
    #         best_k, best_acc = k, accuracy
    # print('Best: k = {} -> {}'.format(best_k, best_acc))
    #
    # # Search for best threshold, fixed k
    # print('\n')
    # best_t, best_acc = 0, float('-inf')
    # for t in range(0, 20):
    #     thresh = t / 20
    #     accuracy = X.accuracy("data/r2d2TrainingSentences.txt", "data/r2d2DevelopmentSentences.txt", k=4, t=thresh)
    #     print('\t{}: {}'.format(thresh, accuracy))
    #     if accuracy > best_acc:
    #         best_t, best_acc = thresh, accuracy
    # print('Best: t = {} -> {}'.format(best_t, best_acc))

    # Grid search on k and threshold
    # print('\n')
    # min_k, max_k = 2, 15
    # min_t, max_t, t_scale = 0, 12, 20
    # best_t, best_k, best_acc = 0, 0, float('-inf')
    # for k in range(min_k, max_k):
    #     print('k = {}'.format(k))
    #     for t in range(min_t, max_t):
    #         thresh = t / t_scale
    #         accuracy = X.accuracy("data/r2d2TrainingSentences.txt", "data/r2d2DevelopmentSentences.txt", k=k, t=thresh)
    #         print('\t{}: {}'.format(thresh, accuracy))
    #         if accuracy > best_acc:
    #             best_k, best_t, best_acc = k, thresh, accuracy
    # print('-------------------')
    # print('Best: k = {}, t = {} -> {}'.format(best_k, best_t, best_acc))
    # print(X.lightParser("Set your lights to maximum"))
    # print(X.lightParser("Increase the red RGB value of your front light by 50."))
    # print(X.lightParser('Reduce the green value on your lights by 50.'))


############################################################
# Section XXX: Feedback
############################################################

# Please let us know how many hours you spent on this assignment (approximate is fine).
feedback_question_1 = 0

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
