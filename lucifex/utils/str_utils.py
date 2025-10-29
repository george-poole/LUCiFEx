from typing import Iterable, Literal

import numpy as np


SUPERSCRIPTS = {
    '0': '⁰', 
    '1': '¹', 
    '2': '²', 
    '3': '³', 
    '4': '⁴', 
    '5': '⁵', 
    '6': '⁶',
    '7': '⁷', 
    '8': '⁸', 
    '9': '⁹',
}

SUBSCRIPTS = {
    '0': '₀',
    '1': '₁',
    '2': '₂',
    '3': '₃',
    '4': '₄',
    '5': '₅',
    '6': '₆',
    '7': '₇',
    '8': '₈',
    '9': '₉',
}

def str_indexed(
    name: str, 
    n: int,
    mode: Literal['superscript', 'subscript'],
    sgn: bool = False,
    parantheses: bool = False,    
) -> str:
    if mode == 'superscript':
        d = SUPERSCRIPTS
    elif mode == 'subscript':
        d = SUBSCRIPTS
    else:
        raise ValueError
    
    n_str = str(n)
    if '-' in n_str:
        n_str = n_str.replace('-', '')
        
    superscript = ''
    for _s in n_str:
        superscript = f'{superscript}{d[_s]}'

    if n < 0:
        superscript = f'⁻{superscript}'
    if n > 0 and sgn:
        superscript = f'⁺{superscript}'

    if parantheses:
        superscript = f'⁽{superscript}⁾'

    return f'{name}{superscript}'


def str_scientifc(
    number: float | int,
    n_digits: int = 3,
    ignore: Iterable[int] = (-1, 0, 1, 2),
    tex: bool = False,
    mode: Literal['tex', 'unicode'] = 'tex'
) -> str:
    """
    Returns the number's string representation in the scientific format.
    
    e.g. `123400` –> `1.234 × 10⁵`
    """

    if np.isclose(number, 0):
        exponent = 0
        coeff = 0
    else:
        exponent = int(np.floor(np.log10(abs(number))))
        coeff = round(number / float(10**exponent), n_digits)

    if mode == 'tex':
        times = '\\times'
        exponent = f"^{{{exponent:d}}}"
    elif mode == 'unicode':
        times = '×'
        exponent = str_indexed('', exponent, 'superscript')
    else:
        raise ValueError

    if exponent in ignore:
        s = f"{coeff * 10**exponent:.{n_digits}f}"
    else:
        s = f"{coeff:.{n_digits}f} {times} 10{exponent}"

    if tex:
        s = f"${s}$"

    return s


TEX_SYMBOLS = (
        # lower case Greek
        "alpha",
        "beta",
        "gamma",
        "delta",
        'epsilon',
        'zeta',
        'eta',
        'theta',
        'iota',
        'kappa',
        'lambda',
        'mu',
        'nu',
        'xi',
        'pi',
        'rho',
        'sigma',
        'tau',
        'upsilon',
        'phi',
        'chi',
        'psi',
        'omega',
        # upper case Greek
        'Gamma',
        'Delta',
        'Theta',
        'Lambda',
        'Xi',
        'Pi',
        'Sigma',
        'Upsilon',
        'Phi',
        'Psi',
        'Omega',
        # mathematical operations
        "times",
        "cdot",
    )


def str_tex(
    label: str | float | int | None,
    escape: str = "/",
) -> str:
    """
    Returns a TeX string
    """

    if isinstance(label, (float, int)):
        return f"${label}$"

    if label == "" or label is None:
        return ""

    if label[0] == "$" and label[-1] == "$":
        return label
    
    for char in (escape, '$'):
        if label[0] == char and label[-1] == char:
            return label[1:-1]

    for i in TEX_SYMBOLS:
        if (i in label) and (f"\\{i}" not in label):
            label = label.replace(i, f"\\{i}")

    label = f"${label}$"
    return label


def str_plain(
    label: str | None,
    strip_wspace: bool = True,
) -> str:
    """
    Strips any Tex formatting.
    """
    if label == "" or label is None:
        return ""
    if label[0] == '$' and label[-1] == '$':
        return str_plain(label[1:-1])
    
    for i in TEX_SYMBOLS:
        if f'\\{i}' in label:
            label = label.replace(f'\\{i}', i)

    if strip_wspace:
        label = label.replace(' ', '')

    return label