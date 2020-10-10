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


def evaluate():
    total_correct = 0
    total_examples = 0
    training_accuracies = []
    start_time = datetime.now()
    count = 0

    for session_id in dataset.get_session_ids():
        model = Model()
        session_correct = 0
        session_examples = 0
        session_correct_list = []
        session_data_count = 0

        for state, language, target_output in tqdm.tqdm(dataset.get_session_data(session_id)):
            print(str(count) + " : " + str(session_id) + " : " + str(session_data_count))
            predicted = model.predict(state, language)
            #print(predicted)

            if predicted == target_output:
                session_correct += 1
                session_correct_list.append(1)
            else:
                session_correct_list.append(0)
            session_examples += 1
            
            model.update(state, language, target_output)
            training_accuracies.append(model.training_accuracy())
            session_data_count += 1

        if FLAGS.correctness_log is not None:
            with open(FLAGS.correctness_log, 'a') as f:
                f.write(' '.join(str(c) for c in session_correct_list) + '\n')

        count += 1
        
        with open("dataset_sessions_accuracties.txt", 'a') as f:
            f.write(str(datetime.now()-start_time) + " " + str(session_id) + " " + str(session_correct/session_examples))

        print(datetime.now()-start_time, session_id, session_correct/session_examples)
        total_correct += session_correct
        total_examples += session_examples

    print('overall accuracy: %s%%' % (100*total_correct/total_examples))
    print('average training accuracy: %s%%' % (100*np.mean(training_accuracies)))

def evaluate_batch(data_size, test_size=500):
    results = []
    for session_id in dataset.get_session_ids():
        model = Model()
        session_data = list(dataset.get_session_data(session_id))
        assert len(session_data) > data_size+test_size
        for state, language, target_output in session_data[:data_size]:
            model.update(state, language, target_output, 0)

        for i in range(50):
            model.optimizer_step()

        print(' training accuracy: %s%%' % (100*model.training_accuracy()))

        total_correct = 0
        total_examples = 0
        for state, language, target_output in session_data[-test_size:]:
            predicted = model.predict(state, language)
            if predicted == target_output:
                total_correct += 1
            total_examples += 1

        print(' test accuracy: %s%%' % (100*total_correct/total_examples))
        results.append(total_correct/total_examples)
    print('average test accuracy: %s%%' % (100*np.mean(results)))

def evaluate_batch_increasing():
    for data_size in [5,10,21,46,100,215,464,1000,2154,4641]:
        print('data size:', data_size)
        evaluate_batch(data_size)

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
    if FLAGS.batch_increasing:
        print("1")
        evaluate_batch_increasing()
    elif FLAGS.batch:
        print("2")
        evaluate_batch(100)
    else:
        print("3")
        evaluate()
