import pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import procrustes
import wordvec as wv

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