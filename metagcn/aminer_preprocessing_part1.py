import numpy as np
import json
import sys
from sklearn.preprocessing import StandardScaler

NUM_AUTHORS = 246678
NUM_PAPERS = 332372
NUM_VENUES = 134
FEATURE_SIZE = 100

'''This file is to pre-process the data and convert them to json format'''
'''This file also initializes node features as degrees'''

def pre_processing(prefix):

    author_paper = dict()    #### Must not be None
    paper_author = dict()    #### Must not be None

    paper_venue = dict()
    venue_paper = dict()

    author_venue = dict()
    venue_author = dict()

    author_p_author = dict()    #### Maybe None
    author_v_author = dict()    #### Maybe None


    fp_palinks = open(prefix + 'paper_author.txt')
    for line in fp_palinks:
        pid = int(line.split()[0])
        aid = int(line.split()[1])

        if pid not in paper_author:
            paper_author[pid] = []

        if aid not in paper_author[pid]:
            paper_author[pid].append(aid)

        if aid not in author_paper:
            author_paper[aid] = []

        if pid not in author_paper[aid]:
            author_paper[aid].append(pid)

    assert len(paper_author) == NUM_PAPERS
    assert len(author_paper) == NUM_AUTHORS
    fp_palinks.close()

    author_paper_degrees = np.array([len(author_paper[a]) for a in range(NUM_AUTHORS)], dtype=np.int32)
    np.save(prefix + 'author_paper_degrees.npy', author_paper_degrees)
    print('author_paper_degrees min: ', min(author_paper_degrees), 'max: ', max(author_paper_degrees), 'mean: ',
          sum(author_paper_degrees) / NUM_AUTHORS)

    paper_author_degrees = np.array([len(paper_author[p]) for p in range(NUM_PAPERS)], dtype=np.int32)
    np.save(prefix + 'paper_author_degrees.npy', paper_author_degrees)
    print('paper_author_degrees min: ', min(paper_author_degrees), 'max: ', max(paper_author_degrees), 'mean: ',
          sum(paper_author_degrees) / NUM_PAPERS)


    #### Construct Co-author relation
    for p in paper_author:
        num_authors = len(paper_author[p])
        for i in range(num_authors):
            a_i = paper_author[p][i]
            if a_i not in author_p_author:
                author_p_author[a_i] = []
            for j in range(num_authors):
                if i != j:
                    if paper_author[p][j] not in author_p_author[a_i]:
                        author_p_author[a_i].append(paper_author[p][j])

    author_p_author_degrees = np.array([len(author_p_author[a]) for a in range(NUM_AUTHORS)], dtype=np.int32)
    np.save(prefix + 'author_p_author_degrees.npy', author_p_author_degrees)
    print('author_p_author degrees min: ', min(author_p_author_degrees), 'max: ', max(author_p_author_degrees), 'mean: ',
          sum(author_p_author_degrees) / NUM_AUTHORS)


    fp_pclinks = open(prefix + 'paper_conf.txt')
    for line in fp_pclinks:
        pid = int(line.split()[0])
        cid = int(line.split()[1])

        if cid not in venue_paper:
            venue_paper[cid] = [pid]
        else:
            venue_paper[cid].append(pid)

        paper_venue[pid] = cid

    assert len(venue_paper) == NUM_VENUES
    assert len(paper_venue) == NUM_PAPERS
    fp_pclinks.close()

    venue_paper_degrees = np.array([len(venue_paper[v]) for v in range(NUM_VENUES)], dtype=np.int32)
    np.save(prefix + 'venue_paper_degrees.npy', venue_paper_degrees)
    print('venue_paper_degrees min: ', min(venue_paper_degrees), 'max: ', max(venue_paper_degrees), 'mean: ',
                                           sum(venue_paper_degrees) / NUM_VENUES)


    for venue in venue_paper:
        venue_author[venue] = []
        for paper in venue_paper[venue]:
            for author in paper_author[paper]:

                ### multiple ??
                if author not in venue_author[venue]:
                    venue_author[venue].append(author)

                if author not in author_venue:
                    author_venue[author] = []

                if venue not in author_venue[author]:
                    author_venue[author].append(venue)


    assert len(author_venue) == NUM_AUTHORS


    #### Construct Co-venue relation

    for a in author_venue:
        temp = []
        for v in author_venue[a]:
            temp.extend(venue_author[v])
        author_v_author[a] = list(set(temp) - set(author_p_author[a]) - {a})

    assert len(author_v_author) == NUM_AUTHORS

    author_v_author_degrees = np.array([len(author_v_author[a]) for a in range(NUM_AUTHORS)], dtype=np.int32)
    np.save(prefix + 'author_v_author_degrees.npy', author_v_author_degrees)
    print('author_v_author degrees min: ', min(author_v_author_degrees), 'max: ', max(author_v_author_degrees),
          'mean: ', sum(author_v_author_degrees) / NUM_AUTHORS)



    fp_pa_output = open(prefix + 'paper_author.json', 'w')
    json.dump(paper_author, fp_pa_output)
    fp_pa_output.close()

    fp_ap_output = open(prefix + 'author_paper.json', 'w')
    json.dump(author_paper, fp_ap_output)
    fp_ap_output.close()

    fp_cp_output = open(prefix + 'venue_paper.json', 'w')
    json.dump(venue_paper, fp_cp_output)
    fp_cp_output.close()

    fp_pc_output = open(prefix + 'paper_venue.json', 'w')
    json.dump(paper_venue, fp_pc_output)
    fp_pc_output.close()

    fp_ca_output = open(prefix + 'venue_author.json', 'w')
    json.dump(venue_author, fp_ca_output)
    fp_ca_output.close()

    fp_ac_output = open(prefix + 'author_venue.json', 'w')
    json.dump(author_venue, fp_ac_output)
    fp_ac_output.close()

    apa_a_out = open(prefix + 'author_p_author.json', 'w')
    json.dump(author_p_author, apa_a_out)
    apa_a_out.close()

    ava_a_out = open(prefix + 'author_v_author.json', 'w')
    json.dump(author_v_author, ava_a_out)
    ava_a_out.close()

    apa_a_features = np.vstack([author_p_author_degrees[i] * np.ones((FEATURE_SIZE, ), dtype=np.float32) for i in range(NUM_AUTHORS)])
    none_zero_ids = np.array([i for i in range(NUM_AUTHORS) if author_p_author_degrees[i] > 0])
    none_zero_feats = apa_a_features[none_zero_ids]
    scaler_apa = StandardScaler()
    scaler_apa.fit(none_zero_feats)
    apa_a_features = scaler_apa.transform(apa_a_features)


    ava_a_features = np.vstack([author_v_author_degrees[i] * np.ones((FEATURE_SIZE, ), dtype=np.float32) for i in range(NUM_AUTHORS)])
    none_zero_ids = np.array([i for i in range(NUM_AUTHORS) if author_v_author_degrees[i] > 0])
    none_zero_feats = ava_a_features[none_zero_ids]
    scaler_ava = StandardScaler()
    scaler_ava.fit(none_zero_feats)
    ava_a_features = scaler_ava.transform(ava_a_features)

    np.save(prefix + 'author_p_author_features.npy', apa_a_features)
    np.save(prefix + 'author_v_author_features.npy', ava_a_features)


dirpath = sys.argv[1]

if __name__ == "__main__":
    pre_processing(dirpath)