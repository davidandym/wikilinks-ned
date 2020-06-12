""" Utils used for indexing and statistics. """

# Bijection between objects and integers starting at 0. Useful for mapping
# labels, features, etc. into coordinates of a vector space.
class Indexer(object):
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in xrange(0, len(self))])

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        return self.index_of(object) != -1

    # Returns -1 if the object isn't present, index otherwise
    def index_of(self, object):
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    # Adds the object to the index if it isn't present, always returns a nonnegative index
    def get_index(self, object, add=True):
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]


# Map from objects to doubles that has a default value of 0 for all elements
# Relatively inefficient (dictionary-backed); shouldn't be used for anything very large-scale,
# instead use an Indexer over the objects and use a numpy array to store the values
class Counter(object):
    def __init__(self):
        self.counter = {}

    def __repr__(self):
        return str([str(key) + ": " + str(self.get_count(key)) for key in self.counter.keys()])

    def __len__(self):
        return len(self.counter)

    def keys(self):
        return self.counter.keys()

    def get_count(self, key):
        if self.counter.has_key(key):
            return self.counter[key]
        else:
            return 0

    def increment_count(self, obj, count):
        if self.counter.has_key(obj):
            self.counter[obj] = self.counter[obj] + count
        else:
            self.counter[obj] = count

    def increment_all(self, objs_list, count):
        for obj in objs_list:
            self.increment_count(obj, count)

    def set_count(self, obj, count):
        self.counter[obj] = count

    def add(self, otherCounter):
        for key in otherCounter.counter.keys():
            self.increment_count(key, otherCounter.counter[key])

    # Bad O(n) implementation right now
    def argmax(self):
        best_key = None
        for key in self.counter.keys():
            if best_key is None or self.get_count(key) > self.get_count(best_key):
                best_key = key
        return best_key
