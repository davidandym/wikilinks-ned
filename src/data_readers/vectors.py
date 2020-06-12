""" Util function for reading in a word-embedding file and a class that stores
an indexer and vectors for easy look-ups.
"""

import numpy as np
from utils import *

class Embeddings:
    """ Embeddings class for easy lookup of index and embeddings. """

    def __init__(self, indexer, vectors, key_is_int=False):
        self.vectors = vectors
        self._indexer = indexer 
        self._key_is_int = key_is_int

    def get_embedding(self, obj):
        if self._key_is_int:
            obj_idx = self._indexer.get_index(int(obj), add=False)
        else:
            obj_idx = self._indexer.get_index(obj, add=False)

        if obj_idx != -1:
            return self.vectors[obj_idx]
        else:
            return self.vectors[self._indexer.get_index("UNK")]

    def get_index(self, obj):
        if self._key_is_int:
            obj_idx = self._indexer.get_index(int(obj), add=False)
        else:
            obj_idx = self._indexer.get_index(obj, add=False)

        if obj_idx != -1:
            return obj_idx
        else:
            return self._indexer.get_index("UNK")

def read_embeddings(embeddings_file, key_is_int=False):
    """ Read in embeddings.

    key_is_int - if the key is supposed to be an integer rather than a string.
        I think this was giving me trouble at some point in the code, so I
        guess I moved the handling of it to this file.
    """
    f = open(embeddings_file)
    indexer = Indexer()
    vectors = []
    first = True
    for line in f:
        if first:
            first = False
            continue
        if line.strip() != "":
            space_idx = line.find(' ')
            key = line[:space_idx]
            numbers = line[space_idx+1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            if key_is_int:
                indexer.get_index(int(key))
            else:
                indexer.get_index(key)
            vectors.append(vector)
    f.close()
    print "Read in " + repr(len(indexer)) + " vectors of size " + repr(vectors[0].shape[0])
    indexer.get_index("UNK")
    vectors.append(np.zeros(vectors[0].shape[0]))
    indexer.get_index("PAD")
    vectors.append(np.zeros(vectors[0].shape[0]))
    return Embeddings(indexer, np.array(vectors), key_is_int=key_is_int)

