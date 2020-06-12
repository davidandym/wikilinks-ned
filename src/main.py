""" Train and evaluate a Wikilinks-NED model.

Adapted from:
https://github.com/yotam-happy/NEDforNoisyText/blob/master/src/Experiment.py
"""

import seed

import os
import json
import logging
import argparse as ap

from trainer import Trainer
from data_readers.wikilinks_iterator import WikilinksIterator
from data_conv import DataConverter
from model_builder import ModelBuilder
from evaluator import Evaluator
from ball import Ball

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


#########################
# Experiment entry point
#########################


def main(args):

    config = json.load(open(args.config, 'r'))
    # save config into experiment directory
    json.dump(config, open(os.path.join(args.experiment_dir, 'config.json'), 'w'))
    logging.info("Config json: %s", config)

    ball = Ball(config)

    file_config = config['files']
    base_data_dir = file_config['base-data-dir']
    model_config = config['model']
    train_config = config['training']

    model_builder = ModelBuilder(model_config,
                                 ball.get_word_vecs(),
                                 ball.get_character_vecs(),
                                 ball.get_feature_indxr(),
                                 ball.get_ent_cbow_vecs())

    logging.info("Model Summary:")
    model_builder.build_f().summary()

    logging.info("Building model...")
    model = model_builder.build_trainable_model()
    logging.info("Model Built!")

    logging.info("Building data ball...")
    data_converter = DataConverter(ball)
    logging.info("Data ball built!")

    trainer = Trainer(model,
                      ball,
                      data_converter,
                      neg_sample_k=train_config['neg_samples'],
                      batch_size=train_config['batch_size'],
                      neg_sample_from_cands=train_config['neg_sample_from_cands'])

    # Optimization Loop
    logging.info("--- Starting optimization loop ---")
    for epoch in xrange(0, train_config['epochs']):
        logging.info("Starting epoch %d", epoch + 1)

        data_iterator = WikilinksIterator(base_data_dir + 'train')

        for item in data_iterator.jsons():
            trainer.train_on(item)
        trainer.epoch_done()

        logging.info("Finished training epoch %d", epoch + 1)
        # temp weight save for evaluation
        tmp_weights_path = os.path.join(args.experiment_dir, 'tmp-model.weights')
        model.save_weights(tmp_weights_path)

        logging.info("Evaluating epoch %d", epoch + 1)
        test_model = model_builder.build_f(weights=tmp_weights_path)
        evaluator = Evaluator(test_model, ball, data_converter)
        data_iterator = WikilinksIterator(base_data_dir + 'dev')
        for item in data_iterator.jsons():
            evaluator.evaluate_on(item)
        accuracy = evaluator.evaluate_model()
        logging.info("Model accuracy for epoch %d is %.2f", epoch + 1, accuracy)

    logging.info("Saving final model")
    final_weights_path = os.path.join(args.experiment_dir, 'final-model.weights')
    model.save_weights(final_weights_path)

    test_model = model_builder.build_f(weights=final_weights_path)
    data_iterator = WikilinksIterator(base_data_dir + 'dev')
    final_evaluator = Evaluator(test_model, ball, data_converter)
    logging.info("Starting final model dev evaluation")
    for item in data_iterator.jsons():
        final_evaluator.evaluate_on(item)
    accuracy = final_evaluator.evaluate_model()
    logging.info("Final model dev accuracy is %.2f", accuracy)

    data_iterator = WikilinksIterator(base_data_dir + 'test')
    final_evaluator = Evaluator(test_model, ball, data_converter)
    logging.info("Starting final model test evaluation")
    for item in data_iterator.jsons():
        final_evaluator.evaluate_on(item)
    accuracy = final_evaluator.evaluate_model()
    logging.info("Final model test accuracy is %.2f", accuracy)


if __name__ == "__main__":
    p = ap.ArgumentParser()
    p.add_argument("--config", required=True, type=str,
                   help='Experiment config file.')
    p.add_argument("--experiment_dir", required=True, type=str,
                   help='Experiment output directory.')
    ARGS = p.parse_args()
    main(ARGS)
