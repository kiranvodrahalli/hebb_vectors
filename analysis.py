import pickle
import numpy as np
from numpy.linalg import norm
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from procrustes import procrustes
import wordvec as wv
from wordvec import sim
from math import sqrt
import goog_translate as gt
import matplotlib.pyplot as plt
from saving import save





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
print "Translation dictionary length: " + str(len(translation_dict))

eng_subset = set(translation_dict.keys())
fr_subset = set(translation_dict.values())


# word_subset will be the top 2000 or so words we choose to analyze, (1368) -> reduced to 1187
# disregarding things like 'the' and 'and' and so on. 
# save the english subset as a separate file
# then produce the translations
# then make an (en, fr) file and save that separately. 
# once we have this, we can build the other methods easily.

# returns a subset of the vector dict
# vecs is the vector dictionary to take the subset of
# lang is the language we want to take subset over: en and fr only.
def vec_subdict(lang, vecs):
	word_subset = set()
	if lang == "en":
		word_subset = eng_subset
	elif lang == "fr":
		word_subset = fr_subset
	else:
		print lang + " is not supported.\n"
		return None
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
			lang_sim_dict[w] = sqrt(sum(pow((sim(vec_dict_en[w], vec_dict_en[w2]) - sim(vec_dict_fr[translation[w]], vec_dict_fr[translation[w2]])), 2) for w2 in vec_dict_en.keys() if w != w2))
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
		en_vecs = vec_subdict("en", en_vecsets[i])
		print "# of English vectors: " + str(len(en_vecs))
		for j in range(0, len(fr_vecsets)):
			fr_vecs = vec_subdict("fr", fr_vecsets[j])
			print "# of French vectors: " + str(len(fr_vecs))
			score = lang_similarity_score(en_vecs, fr_vecs, translation_dict)
			vec_pair_scores[(en_vec_strs[i], fr_vec_strs[j])] = score
	# small scores are better
	return sorted(vec_pair_scores, key=vec_pair_scores.get, reverse=False), vec_pair_scores


# each row is a word vector
def build_mat_from_dict(vec_dict):
	mat = None
	first = True
	for w in vec_dict:
		if first == True:
			mat = vec_dict[w]
			first = False
		else:
			mat = np.c_[mat, vec_dict[w]]
	return mat.T

def build_parallel_mats_from_dicts(en_vecs, fr_vecs, en_fr):
	mat_en = None
	mat_fr = None
	en_vec_dict = vec_subdict("en", en_vecs)
	fr_vec_dict = vec_subdict("fr", fr_vecs)
	first = True
	index_map = dict()
	index = 0
	for w in en_vec_dict:
		if first == True:
			mat_en = en_vec_dict[w]
			mat_fr = fr_vec_dict[en_fr[w]]
			first = False
		else:
			mat_en = np.c_[mat_en, en_vec_dict[w]]
			mat_fr = np.c_[mat_fr, fr_vec_dict[en_fr[w]]]
		index_map[index] = (w, en_fr[w])
		index += 1
	return mat_en.T, mat_fr.T, index_map


# calculate procrustes transform for two vector pairs
# not that useful. can calculate difference between vectors before and after, decreases by a little.
# not enough to say that there is a useful linear transformation between the two. may be better with more
# data
def closest_transform(vec_dict_en, vec_dict_fr, translation):
	en_mat, fr_mat, index_map = build_parallel_mats_from_dicts(vec_dict_en, vec_dict_fr, translation)
	# Z is transformed fr_mat
	# transform is a dict specifying the transformation
	d, Z, transform = procrustes(en_mat, fr_mat)
	print "Normalized SSE: " + str(d) + "\n"
	return transform


# CHECKING IF THERE IS A GOOD LINEAR TRANSFORMATION FOR ANY PAIR OF VECTOR SETS
# for every pair of word vectors, first calculate the Frobenius distance
# between the english vectors and the french vectors
# then get the procrustes transform from english to french, and
# calculate the Frobenius distance to the new matrix.  (Recall that Frobenius distance is basically Euclidean)
# output is the ordered semantic vector set pairs from best to worst, and a dictionary
# mapping each pair to (procrustes distance, ratio of old dist/procrustes distance)
# (ideally, first value is small, second value is high)
def procrustes_vs_regular_distances():
	vec_pair_procrustes_dist = dict()
	vec_pair_procrustes_ratio = dict()
	for i in range(0, len(en_vecsets)):
		#en_vecs = vec_subdict("en", en_vecsets[i])
		#en_mat = build_mat_from_dict(en_vecs)
		for j in range(0, len(fr_vecsets)):
			#fr_vecs = vec_subdict("fr", fr_vecsets[j])
			#fr_mat = build_mat_from_dict(fr_vecs)
			en_mat, fr_mat, index_map = build_parallel_mats_from_dicts(en_vecsets[i], fr_vecsets[j], translation_dict)
			# frobenius distance between matrices
			original_dist = sqrt(pow(en_mat - fr_mat, 2).sum())
			d, Z, t = procrustes(en_mat, fr_mat)
			t_mat = np.dot(en_mat, t['rotation'])*t['scale'] + t['translation']
			new_dist = sqrt(pow(t_mat - fr_mat, 2).sum())
			vec_pair_procrustes_dist[(en_vec_strs[i], fr_vec_strs[j])] = new_dist
			vec_pair_procrustes_ratio[(en_vec_strs[i], fr_vec_strs[j])] = original_dist/new_dist
	return sorted(vec_pair_procrustes_dist, key = vec_pair_procrustes_dist.get, reverse=False), vec_pair_procrustes_dist, sorted(vec_pair_procrustes_ratio, key = vec_pair_procrustes_ratio.get, reverse=True), vec_pair_procrustes_ratio


# Now, we have two metrics for ordering the pairs of vectors by: 
# 
# (1) Distance between cosines across all word pairs (our Language Similarity Score - the samller the better.)
# (2) The extent to which there is a linear mapping from the English vectors to the French vectors (Procrustes score)
#     Note 1: The Procrustes Score is basically how much smaller the Frobenius distance between the vectors
#           became after applying the optimal linear transformation. The smaller the better.
#     Note 2: We could also sort by dist(org_eng, fr);
#             also could sort by the difference (dist(org_eng, fr) - dist(procrustes_trans_eng, fr)).  
#             Currently we're sorting by dist(procrustes_trans_eng, fr).

# We can draw interesting conclusions about the language and the vectors based on which pairs were most
# similar, for each of the two metrics. 

# k-nearest-neighbors metric 
# we find the words associated with the k closest vectors to the semantic vector associated with each word
# for both languages. we treat English as ground truth, and French as to-check. 
# we then calculate precision and recall for each word. 
# we will really have two metrics then: k-precision and k-recall. 
def k_nearest_neighbors_scores(k, eng_vec_dict, fr_vec_dict):
	eng_mat, fr_mat, index_map = build_parallel_mats_from_dicts(eng_vec_dict, fr_vec_dict, translation_dict)
	# k + 1 since we discard the top neighbor, which is itself
	neighbors_en = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(eng_mat)
	dist_en, indices_en = neighbors_en.kneighbors(eng_mat)
	neighbors_fr = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(fr_mat)
	dist_fr, indices_fr = neighbors_fr.kneighbors(fr_mat)
	# since we built the matrices in parallel, we know now that indices map to each other,
	# so we simply check the overlap of those to calculate precision and recall. 
	# calculate avg recall for k-recall
	avg_recall = 0.
	num_points = len(indices_en) + 0.
	knearest_map_en = dict()
	knearest_map_fr = dict()
	for i in range(0, int(num_points)):
		w_en = index_map[i][0]
		w_fr = index_map[i][1]
		index_set_en = set(indices_en[i][1:]) # should be size k
		index_set_fr = set(indices_fr[i][1:]) # should be size k
		if w_en not in knearest_map_en:
			knearest_map_en[w_en] = map(lambda z: index_map[z], index_set_en)
		if w_fr not in knearest_map_fr:
			knearest_map_fr[w_fr] = map(lambda z: index_map[z], index_set_fr)
		recall_count = sum(1 for i in index_set_fr if i in index_set_en)
		# precision = recall for this task
		recall = (recall_count + 0.)/len(index_set_en)
		avg_recall += recall
	return (avg_recall/num_points), knearest_map_en, knearest_map_fr

# we will use k = 10
def all_knearest(k):
	pair_recall_knearest_dict = dict()
	for i in range(0, len(en_vecsets)):
		for j in range(0, len(fr_vecsets)):
			pair_name = (en_vec_strs[i], fr_vec_strs[j])
			if pair_name not in pair_recall_knearest_dict:
				pair_recall_knearest_dict[pair_name] = k_nearest_neighbors_scores(k, en_vecsets[i], fr_vecsets[j])
	return pair_recall_knearest_dict

#pair_recall_knearest_dict = all_knearest(10)

# VISUAL REPRESENTATIONS
# (1) we want to plot all the vector spaces first of all in 2D, using TSNE
# (2) then we have these 18 vectors spaces each of dimension 2. We want to cluster these, 
#     and display the clusters on top of the 2D TSNE plots. 
# 
# This will complete our visual representation of the vectors. 

#------------- PLOTS ------------ #

# TSNE plotting of each set of vectors
# lang = 'en' or 'fr'
def tsne(lang, vec_dict):
	vecs = vec_subdict(lang, vec_dict)
	mat = build_mat_from_dict(vecs)
	model = TSNE(n_components=2, random_state=0)
	t_mat = model.fit_transform(mat)
	t_vecs = dict()
	for i in range(len(vecs.keys())):
		w = vecs.keys()[i]
		if w not in t_vecs:
			t_vecs[w] = t_mat[i]
	return t_vecs, t_mat

# build TSNE mappings for all vector sets:
# (keep in mind these are the SUBSETS OF THE VECTORS THAT WE HAVE BEEN ANALYZING)
def tsne_transform():
	tsne_en_vecsets = []
	for i in range(0, len(en_vecsets)):
		en_vecs = vec_subdict("en", en_vecsets[i])
		t_vecs, t_mat = tsne('en', en_vecs)
		tsne_en_vecsets.append(t_vecs)
	tsne_fr_vecsets = []
	for j in range(0, len(fr_vecsets)):
		fr_vecs = vec_subdict("fr", fr_vecsets[j])
		t_vecs, t_mat = tsne('fr', fr_vecs)
		tsne_fr_vecsets.append(t_vecs)
	return tsne_en_vecsets, tsne_fr_vecsets


# NOTE THAT THESE ALSO FOLLOW THE SAME ORDER AS en_vec_strs AND fr_vec_strs
#tsne_en_vecsets, tsne_fr_vecsets = tsne_transform()
tsne_en_vecsets = pickle.load(open("tsne_en_vecsets.p", "rb"))
tsne_fr_vecsets = pickle.load(open("tsne_fr_vecsets.p", "rb"))

# first plot TSNE vectors
def plot_tsne(vec_dict, name):
	vecs = vec_dict.values()
	x = [v[0] for v in vecs]
	y = [v[1] for v in vecs]
	plt.scatter(x, y)
	save("/Users/kiranv/college/3junior-year/spring2015/neu330/final_project/paper/figures/" + name, ext='jpg')
	plt.show()


def make_tsne_plots():
	for i in range(len(tsne_en_vecsets)):
		vecset = tsne_en_vecsets[i]
		name = en_vec_strs[i]
		plot_tsne(vecset,  name)
	for i in range(len(tsne_fr_vecsets)):
		vecset = tsne_fr_vecsets[i]
		name = fr_vec_strs[i]
		plot_tsne(vecset,  name)




'''
# IGNORE
# invert dict
for w in td:
	if td[w] in fr_dup_set:
		if td[w] not in fr_eng_dict:
			fr_eng_dict[td[w]] = []
		fr_eng_dict[td[w]].append(w)
	else:
		fr_eng_dict[td[w]] = [w]
'''

'''
# bidirectional language dictionary"
eng_fr_dict = dict()
new_fr_eng_dict = dict()
for f in fr_eng_dict: 
	e = fr_eng_dict[f]
	new_fr_eng_dict[f] = e[0]
	eng_fr_dict[e[0]] = f
'''
'''
# IGNORE
plt.xlabel("Semantic Vector Set Pairs Sorted by Procrustes' Ratio")
plt.ylabel("Procrustes' Distance and Ratio")
ratio_plot, = plt.plot(sorted(procrust_ratio, reverse=True), label="Procrustes' Ratio")
dist_plot, = plt.plot(procrust_dist, label="Procrustes' Distance")
plt.legend(handles=[ratio_plot, dist_plot], loc=2)
'''

