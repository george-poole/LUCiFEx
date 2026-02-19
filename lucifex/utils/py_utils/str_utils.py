from typing import Iterable, Literal, TypeAlias, Any
import numpy as np


StrSlice: TypeAlias = str | slice
"""
Type alias for strings representing slices 
e.g. `"start:stop"` or `"start:stop:step"`
"""

COLON = ':'
DOUBLE_COLON = f'{COLON}{COLON}'


def as_slice(s: str | slice| Iterable[int]) -> slice:
    if not is_slice(s):
        raise ValueError(f'Invalid string {s} representing a slice.')

    if isinstance(s, slice):
        return s
    if isinstance(s, str):
        s = s.replace(' ', '')
        n_colon = s.count(COLON)
        if n_colon == 1:
            start, stop = s.split(COLON)
            step = ''
        elif n_colon == 2:
            if DOUBLE_COLON in s:
                start, step = s.split(DOUBLE_COLON)
                stop = ''
            else:
                start, stop, step = s.split(COLON)
        else:
            raise ValueError(f'Expected 1 or 2 colons in the string representing a slice.')
        
        if start == '':
            start = 0
        else:
            start = int(start)

        if stop == '':
            stop = None
        else:
            stop = int(stop)

        if step == '':
            step = None
        else:
            step = int(step)

        return slice(start, stop, step)
    
    if isinstance(s, Iterable):
        return slice(*s)
    
    raise MultipleDispatchTypeError(s)


def is_slice(s: str | slice | Iterable[int] | Any) -> bool:
    if isinstance(s, slice):
        return True
    elif isinstance(s, str):
        return s.count(COLON) >= 1 and s.count(COLON) <= 2
    elif isinstance(s, tuple) and all(isinstance(i, int) for i in s):
        return len(s) > 0 and len(s) <= 3
    else:
        return False
    

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
    show_plus: bool = False,
    parentheses: bool = False,    
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
    if n > 0 and show_plus:
        superscript = f'⁺{superscript}'

    if parentheses:
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
    s: str | float | int | None,
    escape: str = "/",
    tex_symbols: Iterable[str] = TEX_SYMBOLS,
) -> str:
    """
    Returns a TeX-formatted string
    """

    if isinstance(s, (float, int)):
        return f"${s}$"

    if s == "" or s is None:
        return ""

    if s[0] == "$" and s[-1] == "$":
        return s
    
    for char in (escape, '$'):
        if s[0] == char and s[-1] == char:
            return s[1:-1]

    for i in tex_symbols:
        if (i in s) and (f"\\{i}" not in s):
            s = s.replace(i, f"\\{i}")

    s = f"${s}$"
    return s


UNICODE_MAP = {
    **{v: k for v, k in SUPERSCRIPTS.items()},
    **{v: k for v, k in SUBSCRIPTS.items()},
}


def str_plain(
    s: str | None,
    strip_wspace: bool = True,
    tex_symbols: Iterable[str] = TEX_SYMBOLS,
    unicode_map: dict[str, str] | None = None, 
) -> str:
    """
    Strips any TeX or Unicode formatting.
    """
    if s == "" or s is None:
        return ""
    
    if s[0] == '$' and s[-1] == '$':
        return str_plain(s[1:-1])

    if unicode_map is None:
        unicode_map = UNICODE_MAP
    
    for i in tex_symbols:
        if f'\\{i}' in s:
            s = s.replace(f'\\{i}', i)

    for uni, plain in unicode_map.items():
        if uni in s:
            s = s.replace(uni, plain)

    if strip_wspace:
        s = s.replace(' ', '')

    return s