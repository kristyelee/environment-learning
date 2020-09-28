from datetime import datetime
import numpy as np
import sys

import pretrain
import model as our_model
import baseline_model
import dataset
import tqdm
import torch

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_bool('baseline', False, 'use baseline model')
flags.DEFINE_bool('batch', False, 'use batch evaluation (only supported with some datasets)')
flags.DEFINE_bool('batch_increasing', False, 'use batch evaluation with larger and larger data sizes')
flags.DEFINE_string('correctness_log', 'dataset_sessions.txt', 'file to write log indicating which predictions were correct')

def topHundredCandidates():
    session_id = list(dataset.get_session_ids()[0])
    model = Model()
    session_correct = 0
    # session_examples = 0
    candidates = []

    for state, language, target_output in tqdm.tqdm(dataset.get_session_data(session_id)):
        while (len(candidates) < 100):
            predicted, discreteRepresentation = model.predictedOutputAndDiscreteTransformation(state, language)
            if predicted == target_output:
                session_correct += 1
                print(session_correct)
                candidates.append(discreteRepresentation)

        break # Use 1 state, language, target_output

    return candidates

if __name__ == '__main__':
    print(sys.argv)
    FLAGS(sys.argv)
    dataset.load()
    if FLAGS.baseline:
        Model = baseline_model.Model
    else:
        if not pretrain.saved_model_exists():
            print('No pretrained model found with prefix "%s"; running pretraining' % FLAGS.pretrain_prefix)
            pretrain.train()
        Model = our_model.Model
    print("test")
    print(topHundredCandidates())