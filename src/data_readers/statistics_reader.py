""" Statistics Reader - used for candidate generation.

This class is basically just a dict of mention strings and their potential wiki 
candidates.

Adapted from:
https://github.com/yotam-happy/NEDforNoisyText/blob/master/src/WikilinksStatistics.py
"""

import json
import os
import numpy as np
from random import randint

base_dir = '/Users/davidmueller/data/wikilinks_NED/wikilinksNED_dataset'


class StatisticsReader(object):
    """ Statistics reader - reads the ''statistics'' output of the wikilinks
    creation process.

    The purpose of this class is candidate generation.
    """

    def __init__(self, path_to_stats):
        self._path_to_stats = path_to_stats
        self._load_statistics()

    def _load_statistics(self):
        """ Simple helper. """
        data = []
        if os.path.isfile(self._path_to_stats):
            f = open(self._path_to_stats, 'r')
            line = f.readline()
            line = f.readline()
            self.stats_json = json.loads(line)
            entity_count_line = f.readline()
            entities_count = json.loads(entity_count_line)
            self.entity_ids = entities_count.keys()
        else:
            raise Exception("Invalid stats file: %s" % self._path_to_stats)

    def get_candidates_for_mention(self, mention):
        """ Get candidate entities for a given mention string. """
        try:
            candidates = self.stats_json[mention.lower()]
            candidates = np.array(candidates.keys(), dtype=int)
            return candidates
        except KeyError:
            return None

    def get_rand_candidate_from_pool(self, gold_id):
        """ Get a random candidate which is NOT equal to 'gold_id'. """
        rand_id = self.entity_ids[np.random.randint(0, len(self.entity_ids))]
        while int(rand_id) == int(gold_id):
            rand_id = self.entity_ids[np.random.randint(0, len(self.entity_ids))]
        return rand_id
