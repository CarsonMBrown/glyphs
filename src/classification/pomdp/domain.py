import pomdp_py

from src.util.glyph_util import get_classes_as_glyph_names

GLYPH_NAMES = get_classes_as_glyph_names()


class State(pomdp_py.State):
    def __init__(self, name):
        if name not in GLYPH_NAMES:
            raise ValueError("Invalid state: %s" % name)
        self.name = name

    def __eq__(self, other):
        return isinstance(other, State) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Action(pomdp_py.Action):
    def __init__(self, name):
        if name != "open-left" and name != "open-right" \
                and name != "listen":
            raise ValueError("Invalid action: %s" % name)
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Action) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Observation(pomdp_py.Observation):
    def __init__(self, name):
        if name not in GLYPH_NAMES:
            raise ValueError("Invalid observation: %s" % name)
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Observation) and self.name == other.name

    def __hash__(self):
        return hash(self.name)
