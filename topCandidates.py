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
import matplotlib.pyplot as plt

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_bool('baseline', False, 'use baseline model')
flags.DEFINE_bool('batch', False, 'use batch evaluation (only supported with some datasets)')
flags.DEFINE_bool('batch_increasing', False, 'use batch evaluation with larger and larger data sizes')
flags.DEFINE_string('correctness_log', 'dataset_sessions.txt', 'file to write log indicating which predictions were correct')

def topKCandidatesHelper(k, state, language, target_output, model):
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

    for i in range(k):
        if candidateTuples[i][2] == target_output:
            return i

    return float('inf')



def topKCandidatesPlot(state, language, target_output, model):
    # Top K candidates for (state, language, target output)

    count = []
    
    for k in range(5,20):
        found = False
        candidateTuples = []
        seenCandidates = set()

        while len(candidateTuples) < k+20:
            predicted, discreteRepresentation, likelihood = model.predictedOutputAndDiscreteTransformation(state, language)
            if discreteRepresentation not in seenCandidates:
                #print(likelihood)
                seenCandidates.add(discreteRepresentation)
                candidateTuples.append((discreteRepresentation, likelihood, predicted))

        candidateTuples.sort(reverse=True, key=lambda x: x[1])

        for i in range(k):
            if candidateTuples[i][2] == target_output:
                count.append(1)
                found = True
                break

        if not found:
            count.append(0)

    return count


def topKCandidatesHelperBatched(k, state, language, target_output, model):
    # Top K candidates for (state, language, target output)
    candidateTuples = []
    seenCandidates = set()

    while len(candidateTuples) < k+20:
        predicted, discreteRepresentation, likelihood = model.predictedOutputAndDiscreteTransformationBatched(state, language)
        if discreteRepresentation not in seenCandidates:
            #print(likelihood)
            seenCandidates.add(discreteRepresentation)
            candidateTuples.append((discreteRepresentation, likelihood, predicted))

    candidateTuples.sort(reverse=True, key=lambda x: x[1])

    for i in range(k):
        if candidateTuples[i][2] == target_output:
            return i

    return float('inf')


def topKCandidatesAccuracyPlot(k, n):
    start_time = datetime.now()
    
    topKAccuracy = []
    x = list(range(5,201))

    for session_id in dataset.get_session_ids():
        count = 0
        number_accurate = 0
        model = Model()

        for state, language, target_output in tqdm.tqdm(dataset.get_session_data(session_id)): 
            if count == 2:
                break

            tuple_state = tuple([tuple(state[i]) for i in range(len(state))]) 
            tuple_target_output = tuple([tuple(target_output[i]) for i in range(len(target_output))])          
            tup = (tuple_state, language, tuple_target_output)

            # Add top K candidates list for this (state, language, target output) to session_data
            k_candidate_success = topKCandidatesPlot(state, language, target_output, model)

            topKAccuracy = [x + y for x, y in zip(topKAccuracy, k_candidate_success)]

            # Update model, as is done in evaluate() in evaluate.py
            model.update(state, language, target_output)

            count += 1
        

    topKAccuracy /= count


    fig, ax1 = plt.subplots()
    ax1.plot(x, topKAccuracy, 'b-')
    ax1.grid(b=True, which='both')
    plt.savefig('kCandidatesPlot.pdf')
    print('Saved figure; Task complete')


        # with open("dataset_sessions_top_k_accuracies.txt", 'a') as f:
        #     f.write(str(datetime.now()-start_time) + " " + str(session_id) + " " + str(number_accurate/count) + " \n")


def topKCandidatesAccuracy(k, n):
    # sessions will have key-value pairs of session_id, session_data
    # sessions = dict()
    start_time = datetime.now()
    for session_id in dataset.get_session_ids():
        count = 0
        number_accurate = 0
        model = Model()

        for state, language, target_output in tqdm.tqdm(dataset.get_session_data(session_id)): 
            if count == n:
                break

            tuple_state = tuple([tuple(state[i]) for i in range(len(state))]) 
            tuple_target_output = tuple([tuple(target_output[i]) for i in range(len(target_output))])          
            tup = (tuple_state, language, tuple_target_output)

            # Add top K candidates list for this (state, language, target output) to session_data
            k_candidate_success = topKCandidatesHelper(k, state, language, target_output, model)

            if k_candidate_success != float('inf'):
                number_accurate += 1

            # Update model, as is done in evaluate() in evaluate.py
            model.update(state, language, target_output)

            count += 1
        
        print("Top K accuracy: " + str(number_accurate / count))

        with open("dataset_sessions_top_k_accuracies.txt", 'a') as f:
            f.write(str(datetime.now()-start_time) + " " + str(session_id) + " " + str(number_accurate/count) + " \n")



def topKCandidatesAccuracyBatched(k, n):
    # sessions will have key-value pairs of session_id, session_data
    # sessions = dict()

    for session_id in dataset.get_session_ids():
        count = 0
        number_accurate = 0
        model = Model()

        for state, language, target_output in tqdm.tqdm(dataset.get_session_data(session_id)): 
            if count == n:
                break

            tuple_state = tuple([tuple(state[i]) for i in range(len(state))]) 
            tuple_target_output = tuple([tuple(target_output[i]) for i in range(len(target_output))])          
            tup = (tuple_state, language, tuple_target_output)

            # Add top K candidates list for this (state, language, target output) to session_data
            k_candidate_success = topKCandidatesHelper(k, state, language, target_output, model)

            if k_candidate_success != float('inf'):
                number_accurate += 1

            # Update model, as is done in evaluate() in evaluate.py
            model.update(state, language, target_output)

            count += 1
        
        print("Top K accuracy: " + str(number_accurate / count))

        with open("dataset_sessions_top_k_accuracies.txt", 'a') as f:
            f.write(str(datetime.now()-start_time) + " " + str(session_id) + " " + str(number_accurate/count) + " \n")

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
    topKCandidatesAccuracyPlot(100, 300)