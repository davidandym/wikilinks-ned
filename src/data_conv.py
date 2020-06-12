""" Responsible for doing the work of converting a json example with text into
indexes and feature templates.
"""


import numpy as np
from nltk import word_tokenize

from ball import MAX_CTX_CHAR_LENGTH, MAX_TITLE_CHAR_LNGTH

class DataConverter(object):
    """
        Main method for this class is convert_training_data. Other functions
        are simply helpers.

        Class is responsible for taking a Wikilnks text-example and doing all
        the pre-processing necessary to convert it into a form ready to be
        input to a model.
    """
    def __init__(self, ball):
        self.ball = ball
        self.feature_indexer = ball.get_feature_indxr()

    def convert_training_data(self, mention, candidates, gold_id):
        """ Converts example into model-input object.

            Handles:
                - Context cleaning, reversing (right), indexing, removes unks
                - Entity indexing
                - Character indexing for contexts and title
                - Feature extraction
                - Label indexing
        """
        model_config = self.ball.config['model']

        right_context = mention['right_context']
        right_context = self.clean_and_cut_word_list(right_context, False, 20)
        right_context_indexed = [self.ball.get_word_index(word) for word in right_context]
        right_context_indexed = self.pad_to_length(right_context_indexed, 20)

        left_context = mention['left_context']
        left_context = self.clean_and_cut_word_list(left_context, True, 20)
        left_context_indexed = [self.ball.get_word_index(word) for word in left_context]
        left_context_indexed = self.pad_to_length(left_context_indexed, 20)

        if model_config['character_cnn']:
            left_context_char_idxd = self.characterize_context(left_context)
            right_context_char_idxd = self.characterize_context(right_context)
        else:
            left_context_char_idxd = []
            right_context_char_idxd = []

        cbow_ent_idxd = []
        ent_title_idxd = []
        features = []
        labels = []

        for cand in candidates:
            candidate_idx = self.ball.get_ent_cbow_index(int(cand))
            cbow_ent_idxd.append(candidate_idx)

            left_attn_guide = None
            right_attn_guide = None
            title = self.ball.get_title_for(cand)
            if model_config['features']:
                features.append(self.build_features(right_context,
                                                    left_context, title))
            else:
                features.append([])
            if model_config['character_cnn']:
                ent_title_idxd.append(self.characterize_title(title))
            else:
                ent_title_idxd.append([])

            labels.append(1 if int(gold_id) == int(cand) else 0)

        inputs = dict()
        inputs['left_context'] = left_context_indexed
        inputs['right_context'] = right_context_indexed
        inputs['left_context_chars'] = left_context_char_idxd
        inputs['right_context_chars'] = right_context_char_idxd
        inputs['cbow_ent'] = cbow_ent_idxd
        inputs['titles_chars'] = ent_title_idxd
        inputs['features'] = features
        inputs['left_attention_guide'] = left_attn_guide if left_attn_guide is not None else []
        inputs['right_attention_guide'] = right_attn_guide if right_attn_guide is not None else []
        inputs['labels'] = labels
        return inputs

    def clean_and_cut_word_list(self, word_list, left, length):
        """ Remove stopwords and unknowns, reverse right context. """
        clean_words = []
        for word in word_list:
            if word in self.ball.get_stopwords():
                continue
            if self.ball.get_word_index(word) == -1:
                continue
            clean_words.append(word)

        if left:
            clean_words.reverse()

        if len(clean_words) > length:
            clean_words = clean_words[:length]

        clean_words.reverse()
        return clean_words

    def characterize_context(self, words):
        """ Pull and index characters for each word in the context. """
        words_char_idxd = []
        if len(words) < 20:
            words_to_20 = [words[i] if i < len(words) else '' for i in range(0, 20)]
        else:
            words_to_20 = words[:20]
        for word in words_to_20:
            chars_idxd = [self.ball.get_character_idx(char) for char in word]
            words_char_idxd.append(self.pad_char_to_length(chars_idxd, MAX_CTX_CHAR_LENGTH))
        return words_char_idxd

    def characterize_title(self, title):
        """ Index characters in title. """
        if title is None:
            title = ''
        title_char_idxd = [self.ball.get_character_idx(char) for char in title]
        return self.pad_char_to_length(title_char_idxd, MAX_TITLE_CHAR_LNGTH)

    def build_features(self, right_ctx, left_ctx, title):
        """ Build up the feature vector, across all words in the left and right
        context and the entity title. """
        features = np.zeros(len(self.feature_indexer))

        if title is None:
            return features

        chopped_title = self.chop_title(title)

        for i, word in enumerate(list(reversed(right_ctx))):
            idx = self.extract_feature(word, i, title, 'R', chopped_title)
            if idx is not None and idx >= 0:
                features[idx] = 1

        for i, word in enumerate(list(reversed(left_ctx))):
            idx = self.extract_feature(word, i, title, 'L', chopped_title)
            if idx is not None and idx >= 0:
                features[idx] = 1

        for i, word in enumerate(list(reversed(left_ctx))):
            idx = self.extract_feature(word, i, title, 'L', chopped_title)
            if idx is not None and idx >= 0:
                features[idx] = 1

        return features

    def chop_title(self, title):
        """ Split title up into words and remove stop words and non-alphanumerics. """
        title = title.lower()
        tokenized_title = word_tokenize(title)
        no_stops_title = []
        for word in tokenized_title:
            if word not in self.ball.get_stopwords() and word.isalnum():
                no_stops_title.append(word)
        return no_stops_title

    def extract_feature(self, word, pos, title, l_or_r, chopped_title):
        """ Extract features from hand-crafted feature template for a given
        word and title combination. """
        if not word.isalnum():
            return None

        if word in chopped_title:
            match = "exact"
        elif word in title:
            match = "substr"
        else:
            return None

        if pos == 0:
            i_pos = '1'
        elif pos == 1:
            i_pos = '2'
        elif pos == 2:
            i_pos = '3'
        elif pos == 3:
            i_pos = '4'
        elif pos == 4:
            i_pos = '5'
        elif pos >= 5 and pos < 10:
            i_pos = '6-10'
        elif pos >= 10 and pos < 20:
            i_pos = '11-20'

        # This is a useless feature because the original dataset is all
        # lowercased - I put it here in case I got around to creating a new
        # version of the dataset, but alas... it didn't happen.
        cap = "NoCap"

        return self.feature_indexer.index_of('%s:%s:%s:%s:%s' % (l_or_r, cap,
                                                                 i_pos, match,
                                                                 'Title'))

    def pad_to_length(self, word_list, length):
        """ Padding helper. """
        if len(word_list) >= length:
            return word_list[:length]
        pad_idx = self.ball.get_word_index("PAD")
        result = np.ones(length) * pad_idx
        result[0:len(word_list)] = word_list
        return result

    def pad_char_to_length(self, chars_idxd, length):
        """ Character padding helper. """
        if len(chars_idxd) >= length:
            return chars_idxd[:length]
        pad_idx = self.ball.get_character_idx("PAD")
        result = np.ones(length) * pad_idx
        result[0:len(chars_idxd)] = chars_idxd
        return result
