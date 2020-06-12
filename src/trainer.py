"""
Model trainer - it's a mess, but essentially it
    - takes in individual examples, and compiles them into a batch
    - once batch size has been achieved, passes batch into the model to train
    - clears out batch buffers and starts again

Loosely based on the trainer at:
https://github.com/yotam-happy/NEDforNoisyText/blob/master/src/ModelTrainer.py
but examples are no longer binary but rather multi-class classification problems
and supports batching.
"""

import logging
import numpy as np


class Trainer(object):
    """ Trains a model on objects passed in. Handles negative sampling, and batching.

    Call 'train_on' to generate the negative samples and add them to the batch.

    When batch hits a certain limit, and it will run training, then reset the batch.

    Call 'epoch_done' to run training on the last batch of data, and get the
     accumulated loss for that epoch.
    """

    def __init__(self, model, ball, data_converter, neg_sample_k=4, batch_size=100, 
                 neg_sample_from_cands=False):
        self._model = model
        self._ball = ball
        self._data_converter = data_converter
        self.ngfc = neg_sample_from_cands
        self._k = neg_sample_k
        self._left_context_batch = []
        self._right_context_batch = []
        self._cbow_cand_batch = []
        self._features_batch = []
        self._left_context_char_batch = []
        self._right_context_char_batch = []
        self._title_char_batch = []
        self._label_batch = []
        self._batch_size = batch_size
        self._loss_from_score = 0
        self._no_stats = 0

    def train_on(self, mention):
        """ Handle candidate negative sampling and adding to batch. """

        gold_wiki_id = int(mention['wikiId'])
        mention_string = mention['word']

        # Add the gold to the list of potentials.
        cands = [gold_wiki_id]

        # This is a wierd bit - have the option to neg-sample from entities
        # which are either random, or from entities which would be candidates
        # for this example. However, there may not be > k candidates for an
        # example, in which case we need to resort to filling up the missing
        # slots with randoms.
        if self.ngfc:
            candidates = self._ball.get_stats_for_mention(mention_string)
            if candidates is None:
                self._no_stats += 1
            else:
                neg_cands = [candidate for candidate in candidates if candidate != gold_wiki_id]
                if len(neg_cands) <= self._k:
                    cands += neg_cands
                else:
                    for _ in xrange(0, self._k):
                        neg_cand = neg_cands[np.random.randint(len(neg_cands))]
                        cands.append(neg_cand)
        cands = self._pad_cands_with_rands(cands, gold_wiki_id, self._k + 1)

        # add the example to the batch
        self._add_to_batch(mention, cands, gold_wiki_id)

    def _add_to_batch(self, mention, candidates, gold_id):
        inputs = self._data_converter.convert_training_data(mention, candidates, gold_id)

        self._left_context_batch.append(inputs['left_context'])
        self._right_context_batch.append(inputs['right_context'])
        self._cbow_cand_batch.append(inputs['cbow_ent'])
        self._features_batch.append(inputs['features'])
        self._left_context_char_batch.append(inputs['left_context_chars'])
        self._right_context_char_batch.append(inputs['right_context_chars'])
        self._title_char_batch.append(inputs['titles_chars'])
        self._label_batch.append(inputs['labels'])

        if len(self._label_batch) >= self._batch_size:
            self._run_training()

    def _run_training(self):

        if len(self._label_batch) < 1:
            return

        inputs = {}
        inputs['ent_in'] = np.array(self._cbow_cand_batch, ndmin=2)
        inputs['lwords_in'] = np.array(self._left_context_batch)
        inputs['rwords_in'] = np.array(self._right_context_batch)
        inputs['lchars_in'] = np.array(self._left_context_char_batch, ndmin=3)
        inputs['rchars_in'] = np.array(self._right_context_char_batch, ndmin=3)
        inputs['titlechars_in'] = np.array(self._title_char_batch)
        inputs['feats_in'] = np.array(self._features_batch)
        correct_outputs = np.array(self._label_batch)

        self._cbow_cand_batch = []
        self._left_context_batch = []
        self._right_context_batch = []
        self._features_batch = []
        self._left_context_char_batch = []
        self._right_context_char_batch = []
        self._title_char_batch = []
        self._label_batch = []

        loss = self._model.train_on_batch(inputs, correct_outputs)

        self._loss_from_score += loss

    def _pad_cands_with_rands(self, cands, gold_id, length_to_pad):
        while len(cands) < length_to_pad:
            cands.append(self._ball.get_rand_cand_not(gold_id))
        return cands

    def epoch_done(self):
        """ runs training on remaining training data,
        clears out data and returns loss from previous epoch
        """
        self._run_training()
        logging.info("Total loss for epoch: %.5f", self._loss_from_score)
        loss = self._loss_from_score
        self._loss_from_score = 0
        logging.info("No stats for: %d", self._no_stats)
        return loss
