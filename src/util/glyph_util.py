glyphs = [
    '.', 'Ⲁ', 'Β', 'Γ', 'Δ', 'Ⲉ', 'Ζ', 'Η', 'Θ',
    'Ι', 'Κ', 'Ⲗ', 'Ⲙ', 'Ν', 'Ⲝ', 'Ο', 'Π',
    'Ρ', 'Ϲ', 'Τ', 'Υ', 'Ⲫ', 'Χ', 'Ⲯ', 'ω']
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

names = ['period', 'alpha', 'beta', 'gamma',
         'delta', 'epsilon', 'zeta',
         'eta', 'theta', 'iota',
         'kappa', 'lambda', 'mu',
         'nu', 'xi', 'omicron',
         'pi', 'rho', 'sigma',
         'tau', 'upsilon', 'phi',
         'chi', 'psi', 'omega', ]


def name_to_glyph(name):
    for i, n in enumerate(names):
        if name == n:
            return glyphs[i]


def glyph_to_name(glyph):
    if glyph in glyph_map:
        glyph = glyph_map[glyph]
    for i, g in enumerate(glyphs):
        if glyph == g:
            return names[i]


def glyph_to_glyph(glyph):
    """
    Converts greek letters to the form found in the manuscripts
    :param glyph: glyph to convert
    :return:
    """
    if glyph in glyphs:
        return glyph
    if glyph in glyph_map:
        return glyph_map[glyph]
