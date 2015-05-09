import pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import procrustes
import wordvec as wv
from wordvec import sim
from math import sqrt

# first we get the set of word vectors to analyze.
# we choose some top #words from the english text, and choose the
# corresponding word vectors for french. 


def build_counts(text):
	counts = dict()
	for w in text:
		if w not in counts:
			counts[w] = 0
		counts[w] += 1
	return counts


# lang = en or fr
# k is the list of top words by frequency (we will choose only interesting words, not like 'the' and so on)
# (this will require some human pre-processing)
def top_k_words(lang, k):
	if lang == "en":
		hp1_text_en = pickle.load(open("hp1_text_en.p", "rb"))
		en_counts = build_counts(hp1_text_en)
		return sorted(en_counts, key=en_counts.get, reverse= True)[0:k]
	elif lang == "fr":
		hp1_text_fr = pickle.load(open("hp1_text_fr.p", "rb"))
		fr_counts = build_counts(hp1_text_fr)
		return sorted(fr_counts, key=fr_counts.get, reverse=True)[0:k]
	else: 
		print(lang + " is not supported.\n")
		return None


# DISTR = uniform, w = 2
en_vecs_unif = pickle.load(open("hp1_vecs_en_unif2.p", "rb"))
fr_vecs_unif = pickle.load(open("hp1_vecs_fr_unif2.p", "rb"))


# returns a vector dictionary for the top k words of language lang
def top_k_vec_dict(lang, k):

# returns dictionary from English->French for the top k English words.
# (this can include maps like harry -> harry, and other hogwarts specific things- google translate
#  won't work for these)
def top_k_translation(k):

# returns a dictionary from each word-concept in the dict to its "language similarity score": a square sum of the 
# differences between the dot product of the word to another word in english and the dot product in french, taken
# over all other words in the dictionary. translation maps english word to french word.
def lang_similarity_dict(vec_dict_en, vec_dict_fr, translation):
	lang_sim_dict = dict()
	for w in vec_dict.keys():
		if w not in lang_sim_dict:
			pos_dict[w] = sqrt(sum(pow((sim(vec_dict_en[w], vec_dict_en[w2]) - sim(vec_dict_fr[translation[w]], vec_dict_fr[translation[w2]])), 2) for w2 in vec_dict_en.keys() if w != w2))
	return lang_sim_dict

# takes in two vector dicts
# vector dict is mapping from common set of words to vectors
# in this case, "common set" is parametrized by an English->French translation function. 
def compare_vector_sets(vec_dict_en, vec_dict_fr):

