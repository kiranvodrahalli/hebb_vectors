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


# DISTR = uniform
en_vecs_unif2 = pickle.load(open("hp1_vecs_en_unif2.p", "rb"))
fr_vecs_unif2 = pickle.load(open("hp1_vecs_fr_unif2.p", "rb"))
en_vecs_unif3 = pickle.load(open("hp1_vecs_en_unif3.p", "rb"))
#fr_vecs_unif3 = pickle.load(open("hp1_vecs_fr_unif3.p", "rb"))
en_vecs_unif4 = pickle.load(open("hp1_vecs_en_unif4.p", "rb"))
#fr_vecs_unif4 = pickle.load(open("hp1_vecs_fr_unif4.p", "rb"))

# DISTR = unimodal
en_vecs_uni2 = pickle.load(open("hp1_vecs_en_uni2.p", "rb"))
#fr_vecs_uni2 = pickle.load(open("hp1_vecs_fr_uni2.p", "rb"))
en_vecs_uni3 = pickle.load(open("hp1_vecs_en_uni3.p", "rb"))
#fr_vecs_uni3 = pickle.load(open("hp1_vecs_fr_uni3.p", "rb"))
#en_vecs_uni4 = pickle.load(open("hp1_vecs_en_uni4.p", "rb"))
#fr_vecs_uni4 = pickle.load(open("hp1_vecs_fr_uni4.p", "rb"))

# DISTR =  bimodal
#en_vecs_bi2 = pickle.load(open("hp1_vecs_en_bi2.p", "rb"))
#fr_vecs_bi2 = pickle.load(open("hp1_vecs_fr_bi2.p", "rb"))
en_vecs_bi3 = pickle.load(open("hp1_vecs_en_bi3.p", "rb"))
#fr_vecs_bi3 = pickle.load(open("hp1_vecs_fr_bi3.p", "rb"))
en_vecs_bi4 = pickle.load(open("hp1_vecs_en_bi4.p", "rb"))
#fr_vecs_bi4 = pickle.load(open("hp1_vecs_fr_bi4.p", "rb"))


# word_subset will be the top 2000 or so words we choose to analyze, 
# disregarding things like 'the' and 'and' and so on. 
# save the english subset as a separate file
# then produce the translations
# then make an (en, fr) file and save that separately. 
# once we have this, we can build the other methods easily.

# returns a subset of the vector dict
# vecs is the vector dictionary to take the subset of
# word_subset is a set of the words we want to use
def vec_subdict(word_subset, vecs):
	new_dict = dict()
	for w in word_subset:
		if w not in new_dict:
			new_dict[w] = vecs[w]
	return new_dict

# returns dictionary from English->French for the subset of English words.
# (this can include maps like harry -> harry, and other hogwarts specific things- google translate
#  won't work for these)
def top_k_translation(word_subset):

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

