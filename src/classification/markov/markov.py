import math
import os.path
import pickle

import numpy as np
import torch
from torch import softmax

from src.classification.markov.viterbi import viterbi
from src.util.glyph_util import index_to_glyph, glyph_to_index


def init_markov_chain(lang_file, hidden_states, *, cache_path=None, log=False, overwrite=False):
    """
    :param lang_file:
    :param hidden_states:
    :param cache_path:
    :param log: 
    :return: hidden_states, initial_probabilities, transmit_probabilities
    """

    if not overwrite and cache_path is not None and os.path.exists(cache_path):
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
                transmit_probabilities[s1][s2] = 1
                initial_probabilities[s1] += 1

    for s1 in hidden_states:
        for s2 in hidden_states:
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


def top_n_markov_optimization(markov_chain, observations, n=1, uncertainty_threshold=1):
    """

    :param markov_chain: in log
    :param observations:
    :param n:
    :param uncertainty_threshold:
    :return:
    """
    if n <= 1:
        return np.argmax(observations, axis=1)

    observations = torch.softmax(observations, dim=1)
    hidden_states, initial_probabilities, transmit_probabilities = markov_chain

    top_n_observations = []
    certain_glyph = []
    for o in observations:
        top_n_indexes = np.argpartition(np.array(o), -n)[-n:]

        # if uncertain
        if probabilities_within_distance(o, uncertainty_threshold, top_n_indexes):
            certain_glyph.append(None)
        else:
            certain_glyph.append(index_to_glyph(np.argmax(o)))

        top_n_glyphs = [index_to_glyph(g) for g in top_n_indexes]
        top_n_observations.append(top_n_glyphs)

    paths = {}
    if certain_glyph[0] is not None:
        paths[certain_glyph[0]] = 1
    else:
        for o in top_n_observations[0]:
            paths[o] = initial_probabilities[o]

    for i, top_n_glyphs in enumerate(top_n_observations[1:], 1):
        new_paths = {}
        for path, prob in paths.items():
            if certain_glyph[i] is not None:
                new_paths[path + certain_glyph[i]] = prob
            else:
                for g in top_n_glyphs:
                    new_paths[path + g] = prob + transmit_probabilities[path[-1]][g]

        paths = new_paths

    # get maximum likelyhood path
    max_path = ""
    max_prob = None
    for path, prob in paths.items():
        if max_prob is None or prob > max_prob:
            max_path = path
            max_prob = prob

    if len(np.array([glyph_to_index(g) for g in max_path])) != 8:
        print("SHIT")
    return np.array([glyph_to_index(g) for g in max_path])


def probabilities_within_distance(observations, distance, top_n_indexes):
    for i, g1 in enumerate(top_n_indexes):
        for g2 in top_n_indexes[i + 1:]:
            if abs(observations[g1] - observations[g2]) < distance:
                return True
    return False
