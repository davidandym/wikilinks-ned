"""
Evaluation class - takes in examples, one at a time, runs them through the
model and logs statistics. Once done on all examples, will return accuracy.

Adapted from:
https://github.com/yotam-happy/NEDforNoisyText/blob/master/src/Evaluation.py
"""


import logging

import numpy as np


class Evaluator(object):
    """
    Evaluate_on runs over a single data sample.

    Evaluate_model compiles results into total test-set statistics.
    """

    def __init__(self, model, ball, data_converter):
        self._model = model
        self._ball = ball
        self._data_converter = data_converter
        self._title_match_wdw = 3
        self._total_count = 0
        self._num_correct = 0
        self._bad_prediction = 0
        self._bad_candidates = 0
        self._matched_gold = 0
        self._no_gold_id = 0
        self._stopwords = ball.get_stopwords()

    def evaluate_on(self, data_sample):
        """ Takes a single example - gets all candidate id's and uses the model
        to run inference.

        Does this by running the model on each candidate id, and picking
        one with the highest probability.
        """

        # Get candidates
        mention = data_sample['word']
        gold_id = int(data_sample['wikiId'])
        candidates = self._ball.get_stats_for_mention(mention)

        # Record Stats
        self._total_count += 1

        if candidates is None:
            self._bad_candidates += 1
            return False
        if gold_id not in candidates:
            self._no_gold_id += 1
            return False

        if gold_id in candidates and len(candidates) == 1:
            self._matched_gold += 1

        # Setup example for model input and evaluation
        left_inputs       = []
        right_inputs      = []
        entity_inputs     = []
        features_inputs   = []
        left_char_inputs  = []
        right_char_inputs = []
        title_char_inputs = []

        inputs = self._data_converter.convert_training_data(data_sample, candidates, gold_id)

        # For each candidate create a new input set. I'm not sure why we don't
        # just run once and take the argmax from that, this is much more
        # similar to the original code.
        for i, candidate in enumerate(inputs['cbow_ent']):
            left_inputs.append(inputs['left_context'])
            right_inputs.append(inputs['right_context'])
            left_char_inputs.append(inputs['left_context_chars'])
            right_char_inputs.append(inputs['right_context_chars'])
            entity_inputs.append(candidate)
            features_inputs.append([inputs['features'][i]])
            title_char_inputs.append(inputs['titles_chars'][i])

        inputs = {}
        inputs['lwords_in'] = np.array(left_inputs)
        inputs['rwords_in'] = np.array(right_inputs)
        inputs['lchars_in'] = np.array(left_char_inputs)
        inputs['rchars_in'] = np.array(right_char_inputs)
        inputs['ent_in'] = np.array(entity_inputs)
        inputs['feats_in'] = np.array(features_inputs)
        inputs['titlechars_in'] = np.array(title_char_inputs)

        predictions = self._model.predict(inputs, batch_size=len(candidates))

        max_idx = np.argmax([prediction[0] for prediction in predictions])

        data_sample['predicted_id'] = candidates[max_idx]
        if gold_id != int(candidates[max_idx]):
            self._bad_prediction += 1
            return False 
        else:
            self._num_correct += 1
            return True

    def evaluate_model(self):
        """return models accuracy"""
        logging.info("Model Evaluation Statistics:")
        logging.info('total evaluated: %d', self._total_count)
        logging.info('total correct: %d', self._num_correct)
        logging.info('total correct from just matching: %d', self._matched_gold)
        logging.info('total bad prediction: %d', self._bad_prediction)
        logging.info('number of no statistics: %d', self._bad_candidates)
        logging.info('number of gold id not in candidates: %d', self._no_gold_id)

        return (1.0 * self._num_correct) / self._total_count
