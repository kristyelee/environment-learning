from datetime import datetime
import numpy as np
import sys

import pretrain
import model as our_model
import baseline_model
import dataset
import tqdm
import torch
import time
import os

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_bool('baseline', False, 'use baseline model')
flags.DEFINE_bool('batch', False, 'use batch evaluation (only supported with some datasets)')
flags.DEFINE_bool('batch_increasing', False, 'use batch evaluation with larger and larger data sizes')
flags.DEFINE_string('correctness_log', 'dataset_sessions.txt', 'file to write log indicating which predictions were correct')

def topKCandidates(k, state, language, target_output, model):
    # Top K candidates for (state, language, target output)
    candidateTuples = []
    seenCandidates = set()

    while len(candidateTuples) < k+20:
        predicted, discreteRepresentation, likelihood = model.predictedOutputAndDiscreteTransformation(state, language)
        if discreteRepresentation not in seenCandidates:
            #print(likelihood)
            seenCandidates.add(discreteRepresentation)
            candidateTuples.append((discreteRepresentation, likelihood, predicted))

    candidateTuples.sort(reverse=True, key=lambda x: x[1])
    candidates = []
    #print()

    for i in range(k):
        #print(candidateTuples[i][1])
        candidates.append((candidateTuples[i][0], candidateTuples[i][2]))


    for i in range(k):
        if candidates[i][1] == target_output:
            return True

    return False


def allTopKCandidates(k, n):
    #sessions will have key-value pairs of session_id, session_data
    sessions = dict()

    for session_id in dataset.get_session_ids():
        count = 0
        number_accurate = 0
        model = Model()

        for state, language, target_output in tqdm.tqdm(dataset.get_session_data(session_id)): 
            if count == n:
                print("Top K accuracy: " + str(number_accurate / n))
                return

            tuple_state = tuple([tuple(state[i]) for i in range(len(state))]) 
            tuple_target_output = tuple([tuple(target_output[i]) for i in range(len(target_output))])          
            tup = (tuple_state, language, tuple_target_output)

            # Add top K candidates list for this (state, language, target output) to session_data
            k_candidate_success = topKCandidates(k, state, language, target_output, model)

            if k_candidate_success:
                number_accurate += 1

            # Update model, as is done in evaluate() in evaluate.py
            model.update(state, language, target_output)

            count += 1

    #     # Append all session_data to sessions
    #     sessions[session_id] = number_accurate / n

    # return sessions

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
    print(allTopKCandidates(20))