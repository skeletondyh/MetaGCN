import numpy as np
import json
import sys 

'''This file is to construct adjacency info about A-P-A and A-P-V-P-A'''

NUM_AUTHORS = 246678
NUM_PAPERS = 332372
NUM_VENUES = 134
FEATURE_SIZE = 100


def construct_adj(prefix, max_degree):

	author_p_author = json.load(open(prefix + 'author_p_author.json'))
	author_p_author_deg = np.load(prefix + 'author_p_author_degrees.npy')
	author_p_author_deg_max = max(author_p_author_deg)
	print("max degree for author-paper-author: ", author_p_author_deg_max)

	author_p_author_adj = NUM_AUTHORS * np.ones((NUM_AUTHORS + 1, author_p_author_deg_max), dtype=np.int32)
	print("author-paper-author adjacency shape: ", author_p_author_adj.shape)

	for a in range(NUM_AUTHORS):
		neighbors = np.array(author_p_author[str(a)])
		if len(neighbors) == 0:
			continue
		if len(neighbors) < author_p_author_deg_max:
			neighbors = np.random.choice(neighbors, author_p_author_deg_max, replace=True)
		author_p_author_adj[a, :] = neighbors


	author_v_author = json.load(open(prefix + 'author_v_author.json'))
	author_v_author_deg = np.load(prefix + 'author_v_author_degrees.npy')

	author_v_author_adj = NUM_AUTHORS * np.ones((NUM_AUTHORS + 1, max_degree), dtype=np.int32)
	print("author-venue-author adjacency shape: ", author_v_author_adj.shape)

	for a in range(NUM_AUTHORS):
		neighbors = np.array(author_v_author[str(a)])
		if len(neighbors) == 0:
			continue
		if len(neighbors) < max_degree:
			neighbors = np.random.choice(neighbors, max_degree, replace=True)
		elif len(neighbors) > max_degree:
			neighbors = np.random.choice(neighbors, max_degree, replace=False)
		author_v_author_adj[a, :] = neighbors

	np.save(prefix + 'author_p_author_adj.npy', author_p_author_adj)
	np.save(prefix + 'author_v_author_adj.npy', author_v_author_adj)


dirpath = sys.argv[1]
max_degree = int(sys.argv[2])

if __name__ == '__main__':
	construct_adj(dirpath, max_degree)