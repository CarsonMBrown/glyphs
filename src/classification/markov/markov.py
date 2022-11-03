import math
import os.path
import pickle

import numpy as np
from torch import softmax

from src.classification.markov.viterbi import viterbi


def init_markov_chain(lang_file, hidden_states, *, cache_path=None, log=False):
    """
    :param lang_file:
    :param hidden_states:
    :param cache_path:
    :param log: 
    :return: hidden_states, initial_probabilities, transmit_probabilities
    """

    if cache_path is not None and os.path.exists(cache_path):
        print("Reading Markov Chain From Disk...")
        with open(cache_path, mode="rb") as f:
            return pickle.load(f)
    print("Initializing Markov Chain...")

    initial_probabilities = {}
    transmit_probabilities = {}
    with open(lang_file, mode="r", encoding="UTF_8") as f:
        glyphs = f.read()
        for g1, g2 in zip(glyphs[:-1], glyphs[1:]):
            if g1 not in transmit_probabilities:
                transmit_probabilities[g1] = {g2: 1}
                initial_probabilities[g1] = 1
            elif g2 not in transmit_probabilities[g1]:
                transmit_probabilities[g1][g2] = 1
            else:
                transmit_probabilities[g1][g2] += 1
                initial_probabilities[g1] += 1
        initial_probabilities[glyphs[-1]] += 1

    for s1 in hidden_states:
        if s1 not in transmit_probabilities:
            transmit_probabilities[s1] = {}
        for s2 in hidden_states:
            if s2 not in transmit_probabilities[s1]:
                transmit_probabilities[s1][s2] = 0
            else:
                transmit_probabilities[s1][s2] = (transmit_probabilities[s1][s2]) / \
                                                 (initial_probabilities[s1])
                if log:
                    transmit_probabilities[s1][s2] = math.log(transmit_probabilities[s1][s2])
        initial_probabilities[s1] /= len(glyphs)
        if log:
            initial_probabilities[s1] = math.log(initial_probabilities[s1])

    if cache_path is not None:
        with open(cache_path, mode="wb") as f:
            pickle.dump((hidden_states, initial_probabilities, transmit_probabilities), f)

    return hidden_states, initial_probabilities, transmit_probabilities


def generate_pseudo_observations(hidden_states, observations, log=False):
    observations = [softmax(o, dim=0).numpy() for o in observations]
    pseudo_observations = []
    pseudo_emit_probs = {}
    for i, o in enumerate(observations):
        pseudo_observations.append(f"o{i}")
    for i, s in enumerate(hidden_states):
        pseudo_emit_probs[s] = {}
        normalizing_factor = 0.0
        for j, o in enumerate(pseudo_observations):
            pseudo_emit_probs[s][o] = observations[j][i]
            normalizing_factor += observations[j][i]
        for o in pseudo_observations:
            if log:
                try:
                    pseudo_emit_probs[s][o] = math.log(pseudo_emit_probs[s][o] / normalizing_factor)
                except ValueError:
                    pseudo_emit_probs[s][o] = 0
            else:
                pseudo_emit_probs[s][o] /= normalizing_factor
    return pseudo_observations, pseudo_emit_probs


def pseudo_viterbi(markov_chain, observations):
    hidden_states, initial_probabilities, transmit_probabilities = markov_chain
    pseudo_observations, pseudo_emit_probs = generate_pseudo_observations(hidden_states, observations)
    return viterbi(pseudo_observations, hidden_states, initial_probabilities, transmit_probabilities, pseudo_emit_probs)


def interpolated_markov(markov_chain, observations, *, init_prob_weight=0, n_gram_prob_weight=.5):
    hidden_states, initial_probabilities, transmit_probabilities = markov_chain
    for i in range(len(observations)):
        observations[i] = softmax(observations[i], dim=0)
    observations *= 1000
    for i, i_state in enumerate(hidden_states):
        observations[0, i] = observations[0, i] + (init_prob_weight * initial_probabilities[i_state])
    for o in range(1, len(observations)):
        for i, i_state in enumerate(hidden_states):
            for j, j_state in enumerate(hidden_states):
                observations[o, j] = observations[o, j] + \
                                     (n_gram_prob_weight * transmit_probabilities[i_state][j_state] *
                                      observations[o - 1, i])

    return np.argmax(observations, axis=1)
