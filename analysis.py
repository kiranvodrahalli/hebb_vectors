import pickle
import numpy as np
from numpy.linalg import norm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import procrustes
import wordvec as wv
from wordvec import sim
from math import sqrt
import goog_translate as gt

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


# for a given wordset, get the english translation
# otherwise make hand translation list
def get_translations(eng_wordset):
	hp1_fr_words = pickle.load(open("hp1_wordset_fr.p", "rb"))
	fr_wordset = set()
	translation_pairs = []
	to_hand_translate = []
	for w in eng_wordset:
		t = gt.translate(w, "fr", "en").lower()
		if t in hp1_fr_words:
			print (w, t)
			translation_pairs.append((w, t))
			fr_wordset.add(t)
		else:
			to_hand_translate.append(w)
	return fr_wordset, translation_pairs, to_hand_translate


# DISTR = uniform
en_vecs_unif2 = pickle.load(open("hp1_vecs_en_unif2.p", "rb"))
fr_vecs_unif2 = pickle.load(open("hp1_vecs_fr_unif2.p", "rb"))
en_vecs_unif3 = pickle.load(open("hp1_vecs_en_unif3.p", "rb"))
fr_vecs_unif3 = pickle.load(open("hp1_vecs_fr_unif3.p", "rb"))
en_vecs_unif4 = pickle.load(open("hp1_vecs_en_unif4.p", "rb"))
fr_vecs_unif4 = pickle.load(open("hp1_vecs_fr_unif4.p", "rb"))

# DISTR = unimodal
en_vecs_uni2 = pickle.load(open("hp1_vecs_en_uni2.p", "rb"))
fr_vecs_uni2 = pickle.load(open("hp1_vecs_fr_uni2.p", "rb"))
en_vecs_uni3 = pickle.load(open("hp1_vecs_en_uni3.p", "rb"))
fr_vecs_uni3 = pickle.load(open("hp1_vecs_fr_uni3.p", "rb"))
en_vecs_uni4 = pickle.load(open("hp1_vecs_en_uni4.p", "rb"))
fr_vecs_uni4 = pickle.load(open("hp1_vecs_fr_uni4.p", "rb"))

# DISTR =  bimodal
en_vecs_bi2 = pickle.load(open("hp1_vecs_en_bi2.p", "rb"))
fr_vecs_bi2 = pickle.load(open("hp1_vecs_fr_bi2.p", "rb"))
en_vecs_bi3 = pickle.load(open("hp1_vecs_en_bi3.p", "rb"))
fr_vecs_bi3 = pickle.load(open("hp1_vecs_fr_bi3.p", "rb"))
en_vecs_bi4 = pickle.load(open("hp1_vecs_en_bi4.p", "rb"))
fr_vecs_bi4 = pickle.load(open("hp1_vecs_fr_bi4.p", "rb"))

en_vec_strs = ["en_unif2", "en_unif3", "en_unif4", "en_uni2", "en_uni3", "en_uni4", "en_bi2", "en_bi3", "en_bi4"]
fr_vec_strs = ["fr_unif2", "fr_unif3", "fr_unif4", "fr_uni2", "fr_uni3", "fr_uni4", "fr_bi2", "fr_bi3", "fr_bi4"]
all_vecstrs = en_vec_strs + fr_vec_strs

en_vecsets = [en_vecs_unif2, en_vecs_unif3, en_vecs_unif4, en_vecs_uni2, en_vecs_uni3, en_vecs_uni4, en_vecs_bi2, en_vecs_bi3, en_vecs_bi4]
fr_vecsets = [fr_vecs_unif2, fr_vecs_unif3, fr_vecs_unif4, fr_vecs_uni2, fr_vecs_uni3, fr_vecs_uni4, fr_vecs_bi2, fr_vecs_bi3, fr_vecs_bi4]
all_vecsets = en_vecsets + fr_vecsets

translation_dict = pickle.load(open("translation_dict.p", "rb"))

eng_subset = set(translation_dict.keys())
fr_subset = set(translation_dict.values())




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

# returns a dictionary from each word-concept in the dict to its "language similarity score": a square sum of the 
# differences between the dot product of the word to another word in english and the dot product in french, taken
# over all other words in the dictionary. translation maps english word to french word.
def lang_similarity_dict(vec_dict_en, vec_dict_fr, translation):
	lang_sim_dict = dict()
	for w in vec_dict_en.keys():
		if w not in lang_sim_dict:
			vw = vec_dict_en[w]
			vfw = vec_dict_fr[translation[w]]
			'''
			if norm(vw) == 0 or norm(vfw) == 0:
				print w
			'''
			lang_sim_dict[w] = sqrt(sum(pow((sim(vw, vec_dict_en[w2]) - sim(vfw, vec_dict_fr[translation[w2]])), 2) for w2 in vec_dict_en.keys() if w != w2))
	return lang_sim_dict

# takes in a lang_sim_dict and converts sqrt(sum) -> sqrt(avg sq distance btwn cosine distances for each language)
# this way you can tell what the average distance between the two cosines is! take arccos to find 
# difference in angle; recall that this is 100-dim space!! - this is more for explaining.
def lang_avg_sim_dict(lang_sim_dict):
	avg_sim_dict = dict()
	size = len(lang_sim_dict) - 1
	for w in lang_sim_dict.keys():
		if w not in avg_sim_dict:
			avg_sim_dict[w] = sqrt(pow(lang_sim_dict[w], 2)/(size + 0.))
	return avg_sim_dict

# similarity between an english vector set and a french vector set
# we want to find minimal lang similarity score for two vector sets -> these word vector encodings
# are closest across language. we see which ones these are, and infer what this tells us about the languages.
# we also argue that they better represent some structure in the language. 
def lang_similarity_score(vec_dict_en, vec_dict_fr, translation):
	lang_sim_dict = lang_similarity_dict(vec_dict_en, vec_dict_fr, translation)
	total_score = 0
	for w in vec_dict_en.keys():
		total_score += lang_sim_dict[w]
	return total_score/(len(vec_dict_en.keys()) + 0.)


# compare distances between all possible pairs of english-french wordvec sets. 
def compare_vector_sets():
	vec_pair_scores = dict()
	for i in range(0, len(en_vecsets)):
		en_vecs = vec_subdict(eng_subset, en_vecsets[i])
		for j in range(0, len(fr_vecsets)):
			fr_vecs = vec_subdict(fr_subset, fr_vecsets[j])
			score = lang_similarity_score(en_vecs, fr_vecs, translation_dict)
			vec_pair_scores[(en_vec_strs[i], fr_vec_strs[j])] = score
	# small scores are better
	return sorted(vec_pair_scores, key=vec_pair_scores.get, reverse=False)




#------------- PLOTS ------------ #

# TSNE plotting of each set of vectors

# Cluster plots (of TSNE reduced vectors)

