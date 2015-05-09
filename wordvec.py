import pickle
import neural_net as nn
import numpy as np
from numpy.linalg import norm
import math


#--------------COMPARISON FUNCTION---------#
def sim(v1, v2):
	return np.dot(v1/norm(v1), v2/norm(v2))

#--------------DISTRIBUTIONS---------------#
# uniform distribution for window radius w
def unif(w):
	return np.ones(2*w + 1)

#--------------NONLINEARITIES--------------#
def sigmoid(z):
	return 1./(1 + np.exp(-z))

def tanh(z):
	return math.tanh(z)



#-------------PYTHON FUNCTIONS-------------#

# w is the radius of the word window
# returns a dictionary from words to their vectors
# we set a hard hidden layer size of 100, learning rate of 0.1, and competitive k = 10. 
# we use sigmoid function for the nonlinearity
def word_vectors(wordset, text, dist_func, w):
	skew_distr = dist_func(w) #for example, unif(2)
	net = nn.hebb_net(vocab=wordset, H=100, nonlinearity=sigmoid, lrate=0.1, K=10, distr=skew_distr)
	print("Training...\n")
	net.train(text, w)
	print("Training complete.\n")
	weight_matrix = net.W.T
	vectors = dict()
	for w in wordset:
		if w not in vectors:
			vectors[w] = weight_matrix[net.input_map[w]]
	return vectors


#-----------TESTS--------------#
# of course, while english and french have different vocabulary sizes in the text, 
# we will restrict our analysis to the top most common words: maybe top 100 interesting
# vectors (that are translation pairs), including names of characters! 

# possibly, avoid experimenting with w size cause not enough time. make mention of possibility though. 

# uniform vectors for English (w = 2)
def test_en():
	hp1_wordset_en = pickle.load(open("hp1_wordset_en.p", "rb"))
	hp1_text_en = pickle.load(open("hp1_text_en.p", "rb"))	
	V_en = len(hp1_wordset_en) # vocab size
	print("Loaded the English version of Book 1. Vocabulary size: %d\n" % V_en)
	en_vectors = word_vectors(hp1_wordset_en, hp1_text_en, unif, 2)
	return en_vectors

# uniform vectors for French (w = 2)
def test_fr():
	hp1_wordset_fr = pickle.load(open("hp1_wordset_fr.p", "rb"))
	hp1_text_fr = pickle.load(open("hp1_text_fr.p", "rb"))
	V_fr = len(hp1_wordset_fr) # vocab size
	print("Loaded the French version of Book 1. Vocabulary size: %d\n" % V_fr)
	fr_vectors = word_vectors(hp1_wordset_fr, hp1_text_fr, unif, 2)
	return fr_vectors



