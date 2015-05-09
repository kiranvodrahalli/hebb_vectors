import pickle
import numpy as np
import progressbar

class hebb_net: 
	def __init__(self, vocab, H, nonlinearity, lrate, K, distr):
		# the set of words recognized
		self.words = vocab
		# vocab size
		self.V = len(vocab)
		# map from word to input unit
		self.input_map = dict(zip(vocab, range(0, self.V)))
		# hidden layer size (desired size of vectors)
		self.H = H
		# which nonlinearity is applied
		self.func = nonlinearity
		# learning rate
		self.eta = lrate
		# k-winner-take-all value for hidden layer
		self.k = K
		# bias for which words are important
		self.skew = distr
		# W is the weight matrix
		self.W = np.random.rand(H, self.V)
	def __str__(self):
		return "CPCA Hebbian Learning at rate " + str(self.eta) + " with k = " + str(self.k) + "\n----------------------\n" + "(" + str(self.H) + " x " + str(self.V) + ")\n" + "Skew: " + str(self.skew) + "\n----------------------\n"
	# update the network weights based on CPCA learning rule
	def update_net(self, frame, text):
		text_len = len(text)
		#### INPUT ####
		# list of relevant indices for input
		x = []
		# loop complexity: <= 2w + 1
		for j in range(frame[0], frame[1]):
			if j >= 0 and j < text_len:
				# grab the index of the current unit
				curr_unit = self.input_map[text[j]]
				x.append(curr_unit)
		#### OUTPUT ####
		# output vector
		y = np.zeros(self.H)
		# loop complexity: |H|*|x| <= |H|*(2w + 1)
		for i in range(0, self.H):
			y[i] = self.func(sum(self.skew[n]*self.W[i][x[n]] for n in range(0, len(x))))
		# minimum activation to be non-zero
		min_act = sorted(y, reverse=True)[self.k - 1]
		y_indices = []
		# loop complexity: |H|
		for i in range(0, self.H):
			if y[i] >= min_act:
				y_indices.append(i)
			else: y[i] = 0.0
		#### WEIGHT UPDATE ####
		# loop complexity: k*|V|
		for m in y_indices:
			for n in range(0, len(x)): 
				j = x[n]
				delta_Wmj = self.eta*y[m]*(1*self.skew[n] - self.W[m][j])
				self.W[m][j] += delta_Wmj
			for j in range(0, self.V):
				if j not in x:
					delta_Wmj = -1*self.eta*y[m]*self.W[m][j]
					self.W[m][j] += delta_Wmj

	# train on an input text with window radius w
	def train(self, text, w):
		assert 2*w+1 == len(self.skew)
		# starting frame: center can be accessed via frame[0] + w
		frame = (-w, w)
		tlen = len(text)
		for i in range(0, tlen):
			self.update_net(frame, text)
			frame = (frame[0] + 1, frame[1] + 1)
			progress = (i + 0.)/tlen
			progressbar.update_progress(progress)
