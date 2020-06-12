""" class which builds most of our basic models """

import keras.layers as layers
import keras.backend as K
from keras.models import Model
import tensorflow as tf

from ball import MAX_CTX_CHAR_LENGTH, MAX_TITLE_CHAR_LNGTH

# pylint: disable=too-many-locals,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-branches,too-many-statements
# pylint: disable=cell-var-from-loop


#############################
# Model Helper for Attention
#############################


def build_attn_with_layer(seq, controller, layer, cell_size=300):
    """ Build attention mechanism in computation graph. """
    controller_repeated = layers.RepeatVector(20)(controller)
    controller_repeated = layer(controller_repeated)

    attention = layers.Lambda(my_dot, output_shape=(20,))([controller_repeated, seq])

    attention_s = layers.Flatten()(attention)
    attention = layers.Lambda(to_prob, output_shape=(20,))(attention_s)

    attention_repeated = layers.RepeatVector(cell_size)(attention)
    attention_repeated = layers.Permute((2, 1))(attention_repeated)

    weighted = layers.merge([attention_repeated, seq], mode='mul')
    summed = layers.Lambda(sum_seq, output_shape=(cell_size,))(weighted)
    return summed, attention


###################
# Lambda Functions
###################


def sum_seq(seq):
    """ Lambda wrapper for sum. """
    return K.sum(seq, axis=1, keepdims=False)

def to_prob(vec):
    """ Lambda wrapper for softmax. """
    return K.softmax(vec)

def my_dot(inputs):
    """ Lambda wrapper for dot product. """
    ele_wise = inputs[0] * inputs[1]
    return K.sum(ele_wise, axis=-1, keepdims=True)


######################
# Model Builder Class
######################


class ModelBuilder(object):
    """ Class that builds models based off of a config """
    def __init__(self, config, word_vecs, ch_vecs, features, ent_vecs):
        """ Initialize all layers. """
        self._config = config
        self._feature_size = len(features)

        self.emb_word = layers.Embedding(len(word_vecs),
                                         300,
                                         input_length=20,
                                         weights=[word_vecs],
                                         trainable=True,
                                         name='word-emb')
        self.emb_ent = layers.Embedding(len(ent_vecs),
                                        300,
                                        weights=[ent_vecs],
                                        trainable=True,
                                        name='ent-emb')

        # Character CNN over candidate entity title and context
        if self._config['character_cnn']:

            self.window_size = config['ccnn_window_size']

            # universal character embeddings
            self.emb_ch = layers.Embedding(len(ch_vecs),
                                           300,
                                           weights=[ch_vecs],
                                           trainable=True,
                                           name='char-emb')

            self.ent_reduc = layers.Dense(300, name='ent-reduc-layer')

            self.ch_ctx_cnn = layers.Conv1D(100,
                                            self.window_size,
                                            activation='relu',
                                            name='ch-cnn')

            self.ch_title_cnn = layers.Conv1D(100,
                                              self.window_size,
                                              activation='relu',
                                              name='ch-title-cnn')

            self.ch_ctx_pool = layers.MaxPooling1D(pool_size=MAX_CTX_CHAR_LENGTH
                                                   - self.window_size + 1,
                                                   name='ch-pool')

            self.ch_title_pool = layers.MaxPooling1D(pool_size=MAX_TITLE_CHAR_LNGTH
                                                     - self.window_size + 1,
                                                     name='ch-titile-pool')
        else:
            self.ch_ctx_cnn = None
            self.ch_ctx_pool = None
            self.ch_title_cnn = None
            self.ch_title_pool = None


        # left and right context encoders w/ attention
        self.cell_size = 300

        self.left_rnn = layers.GRU(self.cell_size, return_sequences=True, name='left-rnn')
        self.right_rnn = layers.GRU(self.cell_size, return_sequences=True, name='right-rnn')

        self.left_attn = layers.Dense(self.cell_size, name='left-attn')
        self.right_attn = layers.Dense(self.cell_size, name='right-attn')

        self.lattn_dist = layers.TimeDistributed(self.left_attn, name='lattn-dist')
        self.rattn_dist = layers.TimeDistributed(self.right_attn, name='rattn-tdist')

        # binary classification layer
        self.reduce_layer = layers.Dense(1, activation='relu', name='final-reduce-layer')

    def build_trainable_model(self, neg_sample_size=4, weights=None):
        """Compiles the trainable model.

        Negative sample size = how many negative candidates this model will
            be trained on in addition to the gold candidate.
        """

        k = neg_sample_size + 1  # candidate count

        inputs = []
        ent_in = layers.Input(shape=(k,), dtype='int32', name='ent_in')
        inputs.append(ent_in)
        l_words_in = layers.Input(shape=(20,), dtype='int32',
                                  name='lwords_in')
        r_words_in = layers.Input(shape=(20,), dtype='int32',
                                  name='rwords_in')
        inputs += [l_words_in, r_words_in]

        with tf.device('/cpu:0'):
            lctx_emb, rctx_emb = self.get_word_embs(l_words_in, r_words_in)

        if self._config['features']:
            feats_in = layers.Input(shape=(k, self._feature_size,),
                                    dtype='float32',
                                    name='feats_in')
            inputs.append(feats_in)

        if self._config['character_cnn']:
            l_ch_in = layers.Input(shape=(20, MAX_CTX_CHAR_LENGTH,),
                                   dtype='int32', name='lchars_in')
            r_ch_in = layers.Input(shape=(20, MAX_CTX_CHAR_LENGTH,),
                                   dtype='int32', name='rchars_in')
            title_ch_in = layers.Input(shape=(k, MAX_TITLE_CHAR_LNGTH,),
                                       dtype='int32',
                                       name='titlechars_in')
            inputs += [l_ch_in, r_ch_in, title_ch_in]

            with tf.device('/cpu:0'):
                lch_emb, rch_emb = self.get_ch_embs(l_ch_in, r_ch_in)
            l_filters, r_filters = self.filter_chars(lch_emb, rch_emb)
        else:
            l_filters, r_filters = (None, None)

        l_rnn_out = self.build_seq_output(lctx_emb, 'left', l_filters)
        r_rnn_out = self.build_seq_output(rctx_emb, 'right', r_filters)

        l_out = l_rnn_out
        r_out = r_rnn_out

        outs = []
        for i in range(k):
            ent_sl = layers.Lambda(lambda x: x[:, i])(ent_in)
            ent_emb = self.get_ent_emb(ent_sl)

            if self._config['character_cnn']:
                title_ch_sl = layers.Lambda(lambda x: x[:, i, :])(title_ch_in)
                tch_emb = self.get_title_ch_emb(title_ch_sl)
                title_filters = self.filter_title(tch_emb)
                ent_emb = layers.Concatenate(axis=1)([ent_emb, title_filters])
                ent_emb = self.ent_reduc(ent_emb)
            else:
                title_filters = None

            if self._config['features']:
                feat_sl = layers.Lambda(lambda x: x[:, i, :])(feats_in)
                feats = layers.Reshape((self._feature_size,))(feat_sl)
            else:
                feats = None

            l_out, _ = build_attn_with_layer(l_rnn_out, ent_emb,
                                             self.lattn_dist,
                                             self.cell_size)
            r_out, _ = build_attn_with_layer(r_rnn_out, ent_emb,
                                             self.rattn_dist,
                                             self.cell_size)

            out = self.compare_and_score(l_out, r_out, ent_emb, feats)
            outs.append(out)

        f_out_layer = layers.Concatenate(name='concat_output')(outs)
        probs = layers.Activation(K.softmax)(f_out_layer)

        model = Model(inputs=inputs, outputs=probs)
        model.compile(optimizer=tf.train.AdagradOptimizer(self._config['lr']),
                      loss='categorical_crossentropy')

        if weights is not None:
            model.load_weights(weights, by_name=True)
        return model

    def build_f(self, weights=None):
        """ Builds f, the single-candidate scoring function.
        This function is used for inference.

        It's very similar to the previous method, in fact I'm not sure what
        the difference is.
        """

        # word and entity input
        inputs = []
        ent_in = layers.Input(shape=(1,), dtype='int32', name='ent_in')
        inputs.append(ent_in)
        l_words_in = layers.Input(shape=(20,), dtype='int32',
                                  name='lwords_in')
        r_words_in = layers.Input(shape=(20,), dtype='int32',
                                  name='rwords_in')
        inputs += [l_words_in, r_words_in]

        with tf.device('/cpu:0'):
            lctx_emb, rctx_emb = self.get_word_embs(l_words_in, r_words_in)
            ent_emb = self.get_ent_emb(ent_in)

        # feature input
        if self._config['features']:
            feats_in = layers.Input(shape=(1, self._feature_size,),
                                    dtype='float32', name='feats_in')
            inputs.append(feats_in)
            feats = layers.Reshape((self._feature_size,))(feats_in)
        else:
            feats = None

        # character level input
        if self._config['character_cnn']:
            l_ch_in = layers.Input(shape=(20, MAX_CTX_CHAR_LENGTH,),
                                   dtype='int32',
                                   name='lchars_in')
            r_ch_in = layers.Input(shape=(20, MAX_CTX_CHAR_LENGTH,),
                                   dtype='int32',
                                   name='rchars_in')
            title_ch_in = layers.Input(shape=(MAX_TITLE_CHAR_LNGTH,),
                                       dtype='int32',
                                       name='titlechars_in')
            inputs += [l_ch_in, r_ch_in, title_ch_in]

            with tf.device('/cpu:0'):
                lch_emb, rch_emb = self.get_ch_embs(l_ch_in, r_ch_in)
                tch_emb = self.get_title_ch_emb(title_ch_in)

            l_filters, r_filters = self.filter_chars(lch_emb, rch_emb)
            title_filters = self.filter_title(tch_emb)
            ent_emb = layers.Concatenate(axis=1)([ent_emb, title_filters])
            ent_emb = self.ent_reduc(ent_emb)
        else:
            l_filters, r_filters = (None, None)
            title_filters = None

        l_rnn_out = self.build_seq_output(lctx_emb, 'left', l_filters)
        r_rnn_out = self.build_seq_output(rctx_emb, 'right', r_filters)

        l_rnn_out, _ = build_attn_with_layer(l_rnn_out, ent_emb,
                                             self.lattn_dist,
                                             cell_size=self.cell_size)
        r_rnn_out, _ = build_attn_with_layer(r_rnn_out, ent_emb,
                                             self.rattn_dist,
                                             cell_size=self.cell_size)

        out = self.compare_and_score(l_rnn_out, r_rnn_out, ent_emb, feats)

        model = Model(inputs=inputs, outputs=out)
        if weights is not None:
            model.load_weights(weights, by_name=True)
        return model

    def get_word_embs(self, l_words, r_words):
        """ get context word embeddings """
        l_emb = self.emb_word(l_words)
        r_emb = self.emb_word(r_words)
        return l_emb, r_emb

    def get_ch_embs(self, l_chs, r_chs):
        """ get context character embeddings """
        lch_emb = self.emb_ch(l_chs)
        rch_emb = self.emb_ch(r_chs)
        return lch_emb, rch_emb

    def get_ent_emb(self, ent):
        """ get both title-character and entity embeddings """
        ent_emb = self.emb_ent(ent)
        ent_emb = layers.Reshape((300,))(ent_emb)
        return ent_emb

    def get_title_ch_emb(self, title_chs):
        """ gets the character embedding for a title """
        tch_emb = self.emb_ch(title_chs)
        return tch_emb

    def build_seq_output(self, words, side, ch_filters=None):
        """ builds a sequence output. concatenates CNN filters to GRU inputs
        and feeds that to a GRU. Will output a sequence if attention is enabled
        """

        if side == 'left':
            rnn = self.left_rnn
        elif side == 'right':
            rnn = self.right_rnn

        if ch_filters is not None:
            words = layers.Concatenate(axis=2)([words, ch_filters])
        rnn_out = rnn(words)
        return rnn_out

    def filter_chars(self, left_chs, right_chs):
        """ builds an array of character cnn filters (max-pooled). one set of
        filters for each word, and then returns it as an array """

        l_filter_list = []
        r_filter_list = []

        # pylint: disable=cell-var-from-loop
        for i in range(0, 20):
            left_slice = layers.Lambda(lambda x: x[:, i, :])(left_chs)
            right_slice = layers.Lambda(lambda x: x[:, i, :])(right_chs)

            filters = self.ch_ctx_cnn(left_slice)
            pooled_filters = self.ch_ctx_pool(filters)
            l_filter_list.append(pooled_filters)

            filters = self.ch_ctx_cnn(right_slice)
            pooled_filters = self.ch_ctx_pool(filters)
            r_filter_list.append(pooled_filters)

        l_filters = layers.Concatenate(axis=1)(l_filter_list)
        r_filters = layers.Concatenate(axis=1)(r_filter_list)
        return l_filters, r_filters

    def filter_title(self, title_chs):
        """ computes CNN max-pooled filters over title characters (title
        characters should be on long sequence """
        filters = self.ch_title_cnn(title_chs)
        pool = self.ch_title_pool(filters)
        pool = layers.Reshape((100,))(pool)

        return pool

    def compare_and_score(self, left, right, ent, feats):
        """ Final layer of the compiled model
        Concatenates several comparisons between the vectors of left and right
        contexts and the entity vector.

        Final dense layer takes all of these comparisons, and the final feature
        vector, and outputs a binary prediction.
        """
        comparisons = []

        left_dot = layers.Dot(axes=1, normalize=True)([left, ent])
        right_dot = layers.Dot(axes=1, normalize=True)([right, ent])
        comparisons += [left_dot, right_dot]

        left_diff = layers.Subtract()([left, ent])
        right_diff = layers.Subtract()([right, ent])
        comparisons += [left_diff, right_diff]

        left_diff_sq = layers.Multiply()([left_diff, left_diff])
        right_diff_sq = layers.Multiply()([right_diff, right_diff])
        comparisons += [left_diff_sq, right_diff_sq]

        left_mult = layers.Multiply()([left, ent])
        right_mult = layers.Multiply()([right, ent])
        comparisons += [left_mult, right_mult]

        if feats is not None:
            comparisons.append(feats)

        comparisons_concat = layers.Concatenate(axis=1)(comparisons)
        out = self.reduce_layer(comparisons_concat)
        return out
