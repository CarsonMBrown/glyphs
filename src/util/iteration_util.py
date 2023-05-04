def n_booleans(n):
    """
    Generate all the lists of n elements that can be made of booleans
    :param n: number of elements to have in the lists being generated
    :return a list of lists, where each list is made up of n booleans, s.t each ordered list of booleans is unique
    """
    return n_choices(n, [True, False])


def n_choices(n, choices):
    """
    From a given list of choices, generate all the lists of n elements that can be made by picking an element
    from the list of choices at each index in the new list
    :param n: number of elements to have in the lists being generated
    :param choices: list of choices for each element in the new lists
    :return a list of lists, where each list is made up of n elements from the choices list s.t. the ordered lists that
    are returned are unique
    """
    x = [[c] for c in choices]
    if n == 1:
        return x
    for _ in range(n - 1):
        x = [C + [c] for c in choices for C in x]
    return x