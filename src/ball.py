""" Data ball. I think the basic idea behind this was to have on object that
held references to every bit of data needed for any experiment.

The ball is used by almost every part of the experiment and has:
    - Word embeddings
    - Entity embeddings
    - Character embeddings
    - Word indexers
    - Entity indexers
    - Character indexers
    - Feature indexer
    - Dataset statistics (used for candidate generation)
"""


import json
import logging
import numpy as np

from nltk.corpus import stopwords

from data_readers.vectors import read_embeddings, Embeddings
from data_readers.statistics_reader import StatisticsReader
from data_readers.utils import Indexer


# All lowercase because the default wikilinks download lowercases everything
CHAR_LIST = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '.', '?',
    '!', '(', ')', '[', ']', ':', ';', ' '
]

# Entity-name limitation
MAX_TITLE_CHAR_LNGTH = 264
# Per-word character limitation
MAX_CTX_CHAR_LENGTH = 30

class Ball(object):
    def __init__(self, config):

        self.config = config

        file_config = config['files']
        base_data_dir = file_config['base-data-dir']

        logging.info("Loading page-title db from %s",
                     base_data_dir + file_config['wiki-page-titles'])

        self.id_title_db = json.load(open(base_data_dir + file_config['wiki-page-titles']))
        logging.info("Done! Loaded %d id-title maps", len(self.id_title_db.keys()))

        logging.info("Loading word embeddings")
        self.word_vecs = read_embeddings(base_data_dir + file_config['word-embeds'])
        logging.info("Loading entity embeddings")
        self.cbow_vecs = read_embeddings(base_data_dir + file_config['ent-embeds'], key_is_int=True)
        logging.info("Loading Wikilinks statistics")
        self.wiki_stats = StatisticsReader(base_data_dir + file_config['wikilink-stats'])

        self.stopwords = stopwords.words('english')

        self.feature_indexer = self.build_feat_indexer()
        self.character_vectors = self.build_alphabet_vecs()

    ## WRAPPERS
    def get_word_vecs(self):
        return self.word_vecs.vectors

    def get_ent_cbow_vecs(self):
        return self.cbow_vecs.vectors

    def get_stats(self):
        return self.wiki_stats

    def get_stats_for_mention(self, mention):
        return self.wiki_stats.get_candidates_for_mention(mention)

    def get_rand_cand_not(self, wiki_id):
        return self.wiki_stats.get_rand_candidate_from_pool(wiki_id)

    def get_word_index(self, word):
        return self.word_vecs.get_index(word.lower())

    def get_ent_cbow_index(self, wiki_id):
        return self.cbow_vecs.get_index(wiki_id)

    def get_title_for(self, wiki_id):
        if str(wiki_id) in self.id_title_db:
            return self.id_title_db[str(wiki_id)]
        return None

    def get_stopwords(self):
        return self.stopwords

    def get_feature_indxr(self):
        return self.feature_indexer

    def get_character_idx(self, char):
        return self.character_vectors.get_index(char)

    def get_character_vecs(self):
        return self.character_vectors.vectors

    def build_feat_indexer(self):
        idxr = Indexer()
        for ctx in ['R', 'L']:
            for cap in ['Cap', 'NoCap']:
                for pos in ['1', '2', '3', '4', '5', '6-10', '11-20']:
                    for match in ['exact', 'substr']:
                        for title_or_cat in ['Title']:
                            idxr.get_index('%s:%s:%s:%s:%s' % (ctx, cap, pos,
                                                               match,
                                                               title_or_cat))
        logging.info('%d hand features added to indexer', len(idxr))
        return idxr

    def build_alphabet_vecs(self):
        """ Randomly initialized character vectors """
        idxr = Indexer()
        character_vec_size = 300
        for char in CHAR_LIST:
            idxr.get_index(char)
        vectors = np.random.rand(len(idxr), character_vec_size)
        idxr.get_index("UNK")
        vectors = np.append(vectors, np.zeros((1, character_vec_size)), axis=0)
        idxr.get_index("PAD")
        vectors = np.append(vectors, np.zeros((1, character_vec_size)), axis=0)
        return Embeddings(idxr, vectors)
