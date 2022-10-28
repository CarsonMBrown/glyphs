import numpy as np


class MarkovModel(object):
    def __init__(self, actions, state_to_observation, initial_state):
        """
        Initializes a markov model with a list of actions, transitions probabilities, and the initial state.
        States and Observations are represented as natural numbers in [0, n-1] and [0, m-1] with n = |S|, m = |Ω|
        Inputs are also natural numbers [0, i-1] with i = |A|

        Wikipedia: https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process
        Author: https://solutionspace.blog/2011/12/05/training-a-pomdp-with-python/

        :param actions: A, the list of actions
        :param state_to_observation: T, the set of conditional transition probabilities between states (S)
        :param initial_state: s ∈ S
        """
        # A
        self.actions = actions
        self.i = len(actions)  # number of potential inputs

        # Matrix defining T and Ω
        self.state_to_observation = state_to_observation
        self.n = state_to_observation.shape[1]  # number of states
        self.m = state_to_observation.shape[0]  # number of potential observations

        # initial state
        self.initial_state = initial_state


def make_tableaus(inputs, observations, markov_model: MarkovModel):
    """
    Generate the tableau (transition table) for the given markov model using Devijver’s forward-backward algorithm.
    Given a sequence of inputs (xs) and a sequence of observations (ys) estimate probability of being in a given state.

    Author: https://solutionspace.blog/2011/12/05/training-a-pomdp-with-python/

    :param inputs: inputs
    :param observations: observations
    :param markov_model: the markov model to generate tableaus for
    :return:
    """
    # Initialize arrays for  α, β, γ, and N
    alpha = np.zeros((len(observations), markov_model.n))
    beta = np.zeros((len(observations), markov_model.n))
    gamma = np.zeros((len(observations), markov_model.n))
    N = np.zeros((len(observations), 1))

    # Initialize value for common factor gamma
    gamma[0:1, :] = markov_model.initial_state.T * markov_model.state_to_observation[
                                                   observations[0]:observations[0] + 1, :]
    # Initialize values for N, α and β
    N[0, 0] = 1 / np.sum(gamma[0:1, :])
    alpha[0:1, :] = N[0, 0] * gamma[0:1, :]
    beta[len(observations) - 1:len(observations), :] = np.ones((1, markov_model.n))

    for i in range(1, len(observations)):
        gamma[i:i + 1, :] = markov_model.state_to_observation[observations[i]:observations[i] + 1, :] * \
                            np.sum((markov_model.actions[inputs[i - 1]].T * alpha[i - 1:i, :].T), axis=0)
        N[i, 0] = 1 / np.sum(gamma[i:i + 1, :])
        alpha[i:i + 1, :] = N[i, 0] * gamma[i:i + 1, :]

    for i in range(len(observations) - 1, 0, -1):
        beta[i - 1:i] = N[i] * np.sum(markov_model.actions[inputs[i - 1]] *
                                      (markov_model.state_to_observation[observations[i]:observations[i] + 1, :] *
                                       beta[i:i + 1, :]).T, axis=0)

    return alpha, beta, N


def state_estimates(inputs, observations, m: MarkovModel, tableaus=None):
    """
    Calculates the posterior distribution over all latent variables.

    Author: https://solutionspace.blog/2011/12/05/training-a-pomdp-with-python/

    :param inputs: inputs
    :param observations: observations
    :param m: the markov model to generate tableaus for
    :param tableaus: tableaus to use to estimate state
    :return:
    """
    if tableaus is None:
        tableaus = make_tableaus(inputs, observations, m)
    alpha, beta, _ = tableaus
    return alpha * beta


def transition_estimates(inputs, observations, markov_model: MarkovModel, tableaus=None):
    """
    Estimate the probability of transferring between two states at each time step.

    Author: https://solutionspace.blog/2011/12/05/training-a-pomdp-with-python/

    :param inputs: inputs
    :param observations: observations
    :param markov_model: the markov model to generate tableaus for
    :param tableaus: tableaus to use to estimate
    :return:
    """
    if tableaus is None:
        tableaus = make_tableaus(inputs, observations, markov_model)
    alpha, beta, N = tableaus
    result = np.zeros((markov_model.n, markov_model.n, len(observations)))

    for t in range(len(observations) - 1):
        a = markov_model.actions[inputs[t]]
        result[:, :, t] = \
            a * alpha[t:t + 1, :] * \
            markov_model.state_to_observation[observations[t + 1]:observations[t + 1] + 1, :].T * \
            beta[t + 1:t + 2, :].T * N[t + 1, 0]

    a = markov_model.actions[inputs[len(observations) - 1]]
    result[:, :, len(observations) - 1] = a * alpha[-1:, :]

    return result


def state_output_estimates(inputs, observations, markov_model: MarkovModel, state_estimate=None):
    """
    For each time step computes the posterior probability of being in a state and observing a certain output.

    Author: https://solutionspace.blog/2011/12/05/training-a-pomdp-with-python/

    :param inputs: inputs
    :param observations: observations
    :param markov_model: the markov model to generate tableaus for
    :return:
    """
    if state_estimate is None:
        state_estimate = state_estimates(inputs, observations, markov_model)
    result = np.zeros((markov_model.m, markov_model.n, len(observations)))
    for t in range(len(observations)):
        result[observations[t]:observations[t] + 1, :, t] = state_estimate[t:t + 1, :]
    return result


def improve_params(inputs, observations, markov_model: MarkovModel, tableaus=None):
    """
    Using Baum-Welch style EM update procedure for POMDPs, calculate posterior state estimates,
    posterior transition estimates, and posterior joint state/output estimates for each time step.

    Can be repeated until the model converges (for some definition of convergence).
    The return value of this function is a new list of transition probabilities and a
    new matrix of output probabilities. These two define an updated POMDP model which should
    explain the data better than the old model.

    Author: https://solutionspace.blog/2011/12/05/training-a-pomdp-with-python/

    :param inputs:
    :param observations:
    :param markov_model:
    :param tableaus:
    :return:
    """
    if tableaus is None:
        tableaus = make_tableaus(inputs, observations, markov_model)
    estimates = state_estimates(inputs, observations, markov_model, tableaus=tableaus)
    trans_estimates = transition_estimates(inputs, observations, markov_model, tableaus=tableaus)
    sout_estimates = state_output_estimates(inputs, observations, markov_model, state_estimate=estimates)

    # Calculate the numbers of each input in the input sequence.
    nlist = [0] * markov_model.i
    for x in inputs:
        nlist[x] += 1

    sstates = [np.zeros((markov_model.n, 1)) for _ in range(markov_model.i)]
    for t in range(len(observations)):
        sstates[inputs[t]] += estimates[t:t + 1, :].T / nlist[inputs[t]]

    # Estimator for transition probabilities
    alist = [np.zeros_like(a) for a in markov_model.actions]
    for t in range(len(observations)):
        alist[inputs[t]] += trans_estimates[:, :, t] / nlist[inputs[t]]
    for i in range(markov_model.i):
        alist[i] = alist[i] / sstates[i].T
        np.putmask(alist[i], (np.tile(sstates[i].T == 0, (markov_model.n, 1))), markov_model.actions[i])

    c = np.zeros_like(markov_model.state_to_observation)
    for t in range(len(observations)):
        x = inputs[t]
        c += sout_estimates[:, :, t] / (nlist[x] * markov_model.i * sstates[x].T)
    # Set the output probabilities to the original model if we have no state observation at all.
    sstatem = np.hstack(sstates).T
    mask = np.any(sstatem == 0, axis=0)
    np.putmask(c, (np.tile(mask, (markov_model.m, 1))), markov_model.state_to_observation)

    return alist, c


def likelihood(tableaus):
    _, _, N = tableaus
    return np.product(1 / N)


def log_likelihood(tableaus):
    _, _, N = tableaus
    return -np.sum(np.log(N))
