glyph_look_a_likes = [
    '.', 'Ⲁ', 'Β', 'Γ', 'Δ', 'Ⲉ', 'Ζ', 'Η', 'Θ',
    'Ι', 'Κ', 'Ⲗ', 'Ⲙ', 'Ν', 'Ⲝ', 'Ο', 'Π',
    'Ρ', 'Ϲ', 'Τ', 'Υ', 'Ⲫ', 'Χ', 'Ⲯ', 'ω']
glyphs = [
    '.', 'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ',
    'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π',
    'Ρ', 'Ϲ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω']

required_map = {
    "Σ": "Ϲ"
}
glyph_map = {
    "Α": "Ⲁ",
    "Ε": "Ⲉ",
    "Λ": "Ⲗ",
    "Μ": "Ⲙ",
    "Ξ": "Ⲝ",
    "Σ": "Ϲ",
    "Φ": "Ⲫ",
    "Ψ": "Ⲯ",
    "Ω": "ω",
    "Ϝ": "Ϝ",  # Digamma
    "Ϟ": "Ϟ",  # Koppa
    "Ϛ": "Ϛ",  # Stigma
    "Ϡ": "Ϡ",  # Sampi
}

extended_names = ['period', 'alpha', 'beta', 'gamma',
                  'delta', 'epsilon', 'zeta',
                  'eta', 'theta', 'iota',
                  'kappa', 'lambda', 'mu',
                  'nu', 'xi', 'omicron',
                  'pi', 'rho', 'sigma',
                  'tau', 'upsilon', 'phi',
                  'chi', 'psi', 'omega']

names = ['alpha', 'beta', 'gamma',
         'delta', 'epsilon', 'zeta',
         'eta', 'theta', 'iota',
         'kappa', 'lambda', 'mu',
         'nu', 'xi', 'omicron',
         'pi', 'rho', 'sigma',
         'tau', 'upsilon', 'phi',
         'chi', 'psi', 'omega']

glyph_classes = [
    'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ',
    'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π',
    'Ρ', 'Ϲ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω']


def name_to_glyph(name):
    for i, n in enumerate(names):
        if name == n:
            return glyphs[i]


def name_to_glyph_class(name):
    for i, n in enumerate(names):
        if name == n:
            return glyph_classes[i]


def glyph_to_name(glyph, *, look_a_like_glyphs=False):
    if glyph is None:
        return None
    if look_a_like_glyphs:
        if glyph in glyph_map:
            glyph = glyph_map[glyph]
    for i, g in enumerate(glyphs):
        if glyph == g:
            return extended_names[i]
    print(glyph)


def glyph_class_to_name(glyph):
    for i, g in enumerate(glyph_classes):
        if glyph == g:
            return names[i]
    print(glyph)


def glyph_to_glyph(glyph, *, alternative_glyphs=True):
    """
    Converts greek letters to the form found in the manuscripts
    :param glyph: glyph to convert
    :param alternative_glyphs: if true, allow alternative glyphs from glyph map
    :return:
    """
    if glyph in glyphs:
        return glyph
    if glyph in required_map:
        return required_map[glyph]
    if alternative_glyphs and glyph in glyph_map:
        return glyph_map[glyph]


def glyph_to_index(glyph, *, look_a_like_glyphs=False):
    if look_a_like_glyphs:
        if glyph in glyph_map:
            glyph = glyph_map[glyph]
    for i, g in enumerate(glyph_classes):
        if glyph == g:
            return i
    print(glyph)


def index_to_glyph(index, *, look_a_like_glyphs=False):
    glyph = glyph_classes[index]
    if look_a_like_glyphs:
        if glyph in glyph_map:
            return glyph_map[glyph]
    else:
        return glyph


def get_classes_as_glyphs():
    return [name_to_glyph_class(n) for n in sorted(names)]


def get_classes_as_glyph_names():
    return [n for n in sorted(names)]


def get_num_classes():
    return len(names)
