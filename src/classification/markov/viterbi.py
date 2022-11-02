from src.util import glyph_util


def viterbi(observations, states, start_p, trans_p, emit_p):
    """
    An altered version of the viterbi algorithm that takes a probability distribution for each step 
    instead of for each state. This allows for the viterbi algorithm to be performed with the prior probabilities
    generated by a classification algorithm that is run indepentantly for each step.  Takes probabilties as logs.
    
    Source: https://en.wikipedia.org/wiki/Viterbi_algorithm
    :param observations: list
    :param states: list
    :param start_p: dict with states as keys, log probabilities as values
    :param trans_p: dict with states as keys, (dicts with states as keys, log probabilities as values) as values
    :param emit_p: dict with states as keys, (dicts with observations as keys, log probabilities as values) as values
    :return:
    """
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][observations[0]], "prev": None}

    # Run Viterbi when t > 0
    for t in range(1, len(observations)):
        V.append({})
        for st in states:
            max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob * emit_p[st][observations[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

    opt = []
    max_prob = 0
    best_st = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    # print([glyph_util.glyph_to_index(o) - 1 for o in opt])
    # print("The steps of states are " + " ".join(opt) + " with highest probability of %s" % min_prob)
    return [glyph_util.glyph_to_index(o) - 1 for o in opt]
