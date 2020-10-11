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

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_bool('baseline', False, 'use baseline model')
flags.DEFINE_bool('batch', False, 'use batch evaluation (only supported with some datasets)')
flags.DEFINE_bool('batch_increasing', False, 'use batch evaluation with larger and larger data sizes')
flags.DEFINE_string('correctness_log', 'dataset_sessions.txt', 'file to write log indicating which predictions were correct')

def topKCandidates(k, state, language, target_output, model):
    # Top K candidates for (state, language, target output)
    candidateTuples = []

    while len(candidateTuples) < k+20:
        predicted, discreteRepresentation, likelihood = model.predictedOutputAndDiscreteTransformation(state, language)
        if predicted == target_output:
            print(likelihood)
            candidateTuples.append((discreteRepresentation, likelihood))

    candidateTuples.sort(reverse=True, key=lambda x: x[1])
    candidates = []
    print()

    for i in range(k):
        print(candidateTuples[i][1])
        candidates.append(candidateTuples[i][0])

    time.sleep(10)
    print(candidates)
    return candidates


def allTopKCandidates(k):
    print(k)
    #sessions will have key-value pairs of session_id, session_data
    sessions = dict()
   
    # session_correct = 0
    # session_examples = 0
    # candidates = []

    for session_id in dataset.get_session_ids():
        
        # Each session has session_data. session_data has key-value pairs of (state, language, target_output) with the top k candidates for that (state, language, target_output)
        session_data = dict()
        model = Model()
        count = 1

        for state, language, target_output in tqdm.tqdm(dataset.get_session_data(session_id)): 
            new_state = tuple([tuple(state[i]) for i in range(len(state))]) 
            new_target_output = tuple([tuple(target_output[i]) for i in range(len(target_output))])          
            tup = (new_state, language, new_target_output)
            # print(tup)

            # Add top K candidates list for this (state, language, target output) to session_data
            session_data[tup] = topKCandidates(k, state, language, target_output, model)

            # Save file
            for i in range(session_data[tup]):
                candidate = session_data[tup][i].cpu().detach().numpy()
                np.savetxt("./top_candidates/session_id/" + str(count) + "/" + str(i) + ".txt", candidate)

            # Update model, as is done in evaluate() in evaluate.py
            model.update(state, language, target_output)

            count += 1

        # Append all session_data to sessions
        sessions[session_id] = session_data

    return sessions

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