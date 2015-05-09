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
	v = np.ones(2*w + 1)
	return v/norm(v)
# skewed left
def skew_left(w):
	v = np.ones(2*w + 1)
	for i in range(0, len(v)):
		v[i] = (1./i)*v[i]
	return v/norm(v)
# skewed right
def skew_right(w):
	v = np.ones(2*w + 1)
	for i in range(0, len(v)):
		v[i] = i*v[i]
	return v/norm(v)
# bimodal at ends -> THIS IS DEPENDENT ON W (distance how far away from center is important?)
def bimodal(w):
	v = np.ones(2*w + 1)
	for i in range(0, len(v)/2 + 1):
		v[i] = (1./i)*v[i]
	for i in range(len(v)/2 + 1, len(v)):
		v[i] = (i - len(v)/2)*v[i]
	return v/norm(v)
# unimodal (basically hump in the middle)
def unimodal(w):
	v = np.ones(2*w + 1)
	for i in range(0, len(v)/2 + 1):
		v[i] = i*v[i]
	for i in range(len(v)/2 + 1, len(v)):
		v[i] = (1./(i - len(v)/2))*v[i]
	return v/norm(v)
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

# we probably should try for w = 2 -> w = 4....
# we have 5 different types of distributions
# most interesting are uniform, bimodal, unimodal: reduce to just these 3. 
# therefore, 3 distributions, 2 languages, w = 2, 3, 4: 18 total. 
# we can run maybe 4 at a time. 4 takes 3 hours at most. 
# this means a max total of 12 hours of training time. need to get another computer to help run this. 
# REQUIREMENTS: numpy up to date, scikit-learn up to date, pickle, my own code: progressbar
# automate all the tests/ plots and there will be enough time to write everything up. 


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

# english, uniform 3, 4, bimodal 2, 3, 4
# should take 12 hours
def vectors_comp1():
	hp1_wordset_en = pickle.load(open("hp1_wordset_en.p", "rb"))
	hp1_text_en = pickle.load(open("hp1_text_en.p", "rb"))	
	V_en = len(hp1_wordset_en) # vocab size
	print("Loaded the English version of Book 1. Vocabulary size: %d\n" % V_en)
	en_vectors3unif = word_vectors(hp1_wordset_en, hp1_text_en, unif, 3)
	pickle.dump(en_vectors3unif, open("hp1_vecs_en_unif3.p", "rb"))
	en_vectors4unif = word_vectors(hp1_wordset_en, hp1_text_en, unif, 4)
	pickle.dump(en_vectors4unif, open("hp1_vecs_en_unif4.p", "rb"))
	en_vectors3bi = word_vectors(hp1_wordset_en, hp1_text_en, bimodal, 3)
	pickle.dump(en_vectors3bi, open("hp1_vecs_en_bi3.p", "rb"))
	en_vectors4bi = word_vectors(hp1_wordset_en, hp1_text_en, bimodal, 4)
	pickle.dump(en_vectors4bi, open("hp1_vecs_en_bi4.p", "rb"))

# french, uniform 3, 4, bimodal 2, 3, 4
# should take 12 hours
def vectors_comp2():
	hp1_wordset_fr = pickle.load(open("hp1_wordset_fr.p", "rb"))
	hp1_text_fr = pickle.load(open("hp1_text_fr.p", "rb"))	
	V_fr = len(hp1_wordset_fr) # vocab size
	print("Loaded the French version of Book 1. Vocabulary size: %d\n" % V_fr)
	fr_vectors3unif = word_vectors(hp1_wordset_fr, hp1_text_fr, unif, 3)
	pickle.dump(fr_vectors3unif, open("hp1_vecs_fr_unif3.p", "rb"))
	fr_vectors4unif = word_vectors(hp1_wordset_fr, hp1_text_fr, unif, 4)
	pickle.dump(fr_vectors4unif, open("hp1_vecs_fr_unif4.p", "rb"))
	fr_vectors3bi = word_vectors(hp1_wordset_fr, hp1_text_fr, bimodal, 3)
	pickle.dump(fr_vectors3bi, open("hp1_vecs_fr_bi3.p", "rb"))
	fr_vectors4bi = word_vectors(hp1_wordset_fr, hp1_text_fr, bimodal, 4)
	pickle.dump(fr_vectors4bi, open("hp1_vecs_fr_bi4.p", "rb"))

# english, unimodal 2, 3, 4
# should take 12 hours
def vectors_comp3():
	hp1_wordset_en = pickle.load(open("hp1_wordset_en.p", "rb"))
	hp1_text_en = pickle.load(open("hp1_text_en.p", "rb"))	
	V_en = len(hp1_wordset_en) # vocab size
	print("Loaded the English version of Book 1. Vocabulary size: %d\n" % V_en)
	en_vectors2uni = word_vectors(hp1_wordset_en, hp1_text_en, unimodal, 2)
	pickle.dump(en_vectors2uni, open("hp1_vecs_en_uni2.p", "rb"))
	en_vectors3uni = word_vectors(hp1_wordset_en, hp1_text_en, unimodal, 3)
	pickle.dump(en_vectors3uni, open("hp1_vecs_en_uni3.p", "rb"))
	en_vectors4uni = word_vectors(hp1_wordset_en, hp1_text_en, unimodal, 4)
	pickle.dump(en_vectors4uni, open("hp1_vecs_en_uni4.p", "rb"))
	en_vectors2bi = word_vectors(hp1_wordset_en, hp1_text_en, bimodal, 2)
	pickle.dump(en_vectors2bi, open("hp1_vecs_en_bi2.p", "rb"))

# french, unimodal 2, 3, 4
# should take 12 hours
def vectors_comp4():
	hp1_wordset_fr = pickle.load(open("hp1_wordset_fr.p", "rb"))
	hp1_text_fr = pickle.load(open("hp1_text_fr.p", "rb"))	
	V_fr = len(hp1_wordset_fr) # vocab size
	print("Loaded the French version of Book 1. Vocabulary size: %d\n" % V_fr)
	fr_vectors2uni = word_vectors(hp1_wordset_fr, hp1_text_fr, unimodal, 2)
	pickle.dump(fr_vectors2uni, open("hp1_vecs_fr_uni2.p", "rb"))
	fr_vectors3uni = word_vectors(hp1_wordset_fr, hp1_text_fr, unimodal, 3)
	pickle.dump(fr_vectors3uni, open("hp1_vecs_fr_uni3.p", "rb"))
	fr_vectors4uni = word_vectors(hp1_wordset_fr, hp1_text_fr, unimodal, 4)
	pickle.dump(fr_vectors4uni, open("hp1_vecs_fr_uni4.p", "rb"))
	fr_vectors2bi = word_vectors(hp1_wordset_fr, hp1_text_fr, bimodal, 2)
	pickle.dump(fr_vectors2bi, open("hp1_vecs_fr_bi2.p", "rb"))
