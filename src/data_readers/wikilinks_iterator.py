""" Dataset iterator. 

Dataset is too big to be loaded into memory, so it is pre-shuffled, pre-sharded, 
and loaded in one example at a time.


Adapted from:
https://github.com/yotam-happy/NEDforNoisyText/blob/master/src/WikilinksIterator.py
"""

import nltk
import unicodedata
import urllib
import os
import time
import json
from nltk.corpus import stopwords


class WikilinksIterator:
    def __init__(self, path, limit=None):
        self._path = path
        self._limit = limit
        self._stopwords = stopwords.words('english')

    def _wikilink_files(self):
        for i, f in enumerate(os.listdir(self._path)):
            if os.path.isdir(os.path.join(self._path, f)):
                continue
            print time.strftime("%H:%M:%S"), "- opening", f, "(", i, "opened so far in this epoch)"
            yield open(os.path.join(self._path, f), 'r')

    def jsons(self):
        """ Iterable method - yields one wikilinks example at a time. """
        for c, f in enumerate(self._wikilink_files()):
            lines = f.readlines()
            jsonObj = []
            for line in lines:
                if len(line) > 0:
                    wlink = json.loads(line) 

                    # filter
                    if not 'word' in wlink:
                        continue
                    if 'right_context' not in wlink and 'left_context' not in wlink:
                        continue

                    wlink['wikiId'] = int(wlink['wikiId']) if 'wikiId' in wlink else None

                    if 'mention_as_list' not in wlink:
                        mention_as_list = unicodedata.normalize('NFKD', wlink['word']).encode('ascii','ignore').lower()
                        mention_as_list = nltk.word_tokenize(mention_as_list)
                        wlink['mention_as_list'] = mention_as_list

                    # preprocess context (if not already processed
                    if 'right_context' in wlink and not isinstance(wlink['right_context'], list):
                        wlink['right_context_text'] = wlink['right_context']
                        r_context = unicodedata.normalize('NFKD', wlink['right_context']).encode('ascii','ignore').lower()
                        wlink['right_context'] = nltk.word_tokenize(r_context)
                        wlink['right_context'] = [w for w in wlink['right_context']]
                    if 'left_context' in wlink and not isinstance(wlink['left_context'], list):
                        wlink['left_context_text'] = wlink['left_context']
                        l_context = unicodedata.normalize('NFKD', wlink['left_context']).encode('ascii','ignore').lower()
                        wlink['left_context'] = nltk.word_tokenize(l_context)
                        wlink['left_context'] = [w for w in wlink['left_context']]

                    # return
                    yield wlink

            f.close()

