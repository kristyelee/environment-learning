import numpy as np
import math

import torch
import torch.nn.functional as F

import message_flags

from absl import flags
FLAGS = flags.FLAGS

def sample_gumbel(shape, device, eps=1e-20):
    # gumbel(0,1)
    u = torch.rand(*shape, device=device)
    return -torch.log(-torch.log(u+eps)+eps)

def discretize(logits, dim):
    # make one-hot for mode of a distribution
    _, am = torch.max(logits, dim, keepdim=True)
    result = torch.zeros_like(logits)
    result.scatter_(dim, am, 1)
    return result

def gumbel_softmax(logits, dim, temp=1, straight_through=True):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    continuous = F.softmax(y/temp, dim)
    #print(logits)
    #print(continuous)
    #np.save('gumbel_softmax', continuous.cpu().detach().numpy())
    if straight_through:
        discrete = discretize(continuous, dim)
        #print(discrete) 
        return continuous + (discrete - continuous).detach()
    else:
        return continuous

def gumbel_softmax_with_likelihood(logits, dim, temp=1, straight_through=True):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    continuous = F.softmax(y/temp, dim)
    if straight_through:
        discrete = discretize(continuous, dim)
        return continuous + (discrete - continuous).detach(), torch.sum(torch.log(torch.sum(discrete*continuous, axis = -1)), axis = -1)
    else:
        return continuous

def gumbel_softmax_with_likelihood_batched(logits, dim, num_batch, temp=1, straight_through=True):
    samples = []
    for i in range(num_batch):
        samples.append(logits + sample_gumbel(logits.size(), device=logits.device))
    continuous = F.softmax(samples/temp, dim)
    if straight_through:
        discrete = [discretize(continuous_vector, dim) for continuous_vector in continuous]
        return [continuous_vector + (discrete_vector - continuous_vector).detach() for discrete_vector, continuous_vector in zip(discrete, continuous)], [torch.sum(torch.log(torch.sum(discrete_vector*continuous_vector, axis = -1)), axis = -1) for discrete_vector, continuous_vector in zip(discrete, continuous)]
    else:
        return continuous

def sample_multinomial(logits, dim):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return discretize(y, dim)

def kl(message_logits):
    # kl from uniform
    # message_logits has dimensions (batch_size, message_size, message_symbols)
    message_logits = F.log_softmax(message_logits, 2)
    probs = torch.exp(message_logits)
    return (probs*(message_logits-math.log(1./FLAGS.discrete_message_symbols))).sum()

def discrete_transformation(variable):
    # takes a 2d (batch_size by vector_size) variable and returns another with the same dimensions that has been discretized (in a differentiable way)
    message_size = FLAGS.discrete_message_size
    message_symbols = FLAGS.discrete_message_symbols
    assert len(variable.size()) == 2 and variable.size()[1] == message_size*message_symbols
    x = variable.view(-1, message_size, message_symbols)
    x = gumbel_softmax(x,2)
    return x.view(-1, message_size*message_symbols)


def discrete_transformation_with_likelihood(variable):
    # takes a 2d (batch_size by vector_size) variable and returns another with the same dimensions that has been discretized (in a differentiable way), as well as the likelihood of the sample used
    message_size = FLAGS.discrete_message_size
    message_symbols = FLAGS.discrete_message_symbols
    assert len(variable.size()) == 2 and variable.size()[1] == message_size*message_symbols
    x = variable.view(-1, message_size, message_symbols)
    x, likelihood = gumbel_softmax_with_likelihood(x,2)
    return x.view(-1, message_size*message_symbols), likelihood

def discrete_transformation_with_likelihood_batched(variable, num_batch):
    # takes a 2d (batch_size by vector_size) variable and returns another with the same dimensions that has been discretized (in a differentiable way), as well as the likelihood of the sample used
    message_size = FLAGS.discrete_message_size
    message_symbols = FLAGS.discrete_message_symbols
    assert len(variable.size()) == 2 and variable.size()[1] == message_size*message_symbols
    x = variable.view(-1, message_size, message_symbols)
    x, likelihood = gumbel_softmax_with_likelihood(x,2, num_batch)
    return [v.view(-1, message_size*message_symbols) for v in x], likelihood

def kl_flattened(variable):
    message_size = FLAGS.discrete_message_size
    message_symbols = FLAGS.discrete_message_symbols
    assert len(variable.size()) == 2 and variable.size()[1] == message_size*message_symbols
    return kl(variable.view(-1, message_size, message_symbols))
