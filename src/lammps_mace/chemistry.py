from __future__ import annotations

from collections import defaultdict
import re


DEFAULT_MASSES = {
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.0983,
    "Ca": 40.078,
    "Br": 79.904,
    "Rb": 85.4678,
    "Sr": 87.62,
    "I": 126.90447,
    "Cs": 132.90545196,
    "Ba": 137.327,
}

_TOKEN_RE = re.compile(r"([A-Z][a-z]?|\(|\)|\d+)")


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a chemical formula into element counts.

    Supports simple formulae and parenthesized groups such as `AlF3` or
    `Ca(NO3)2`. Charge annotations are intentionally not supported.
    """
    compact = formula.strip().replace(" ", "")
    if not compact:
        raise ValueError("Formula cannot be empty.")
    tokens = _TOKEN_RE.findall(compact)
    if "".join(tokens) != compact:
        raise ValueError(f"Unsupported formula syntax: {formula}")

    counts, position = _parse_tokens(tokens, 0)
    if position != len(tokens):
        raise ValueError(f"Unexpected formula tail in: {formula}")
    return dict(counts)


def molar_mass(formula_counts: dict[str, int], masses: dict[str, float] | None = None) -> float:
    mass_table = DEFAULT_MASSES if masses is None else DEFAULT_MASSES | masses
    total = 0.0
    for element, count in formula_counts.items():
        if element not in mass_table:
            raise KeyError(f"Missing atomic mass for element '{element}'.")
        total += float(mass_table[element]) * count
    return total


def _parse_tokens(tokens: list[str], start: int) -> tuple[defaultdict[str, int], int]:
    counts: defaultdict[str, int] = defaultdict(int)
    position = start
    while position < len(tokens):
        token = tokens[position]
        if token == ")":
            return counts, position + 1
        if token == "(":
            nested, position = _parse_tokens(tokens, position + 1)
            multiplier, position = _read_multiplier(tokens, position)
            for element, count in nested.items():
                counts[element] += count * multiplier
            continue
        if token.isdigit():
            raise ValueError("Formula multiplier must follow an element or a group.")
        element = token
        position += 1
        multiplier, position = _read_multiplier(tokens, position)
        counts[element] += multiplier
    return counts, position


def _read_multiplier(tokens: list[str], position: int) -> tuple[int, int]:
    if position < len(tokens) and tokens[position].isdigit():
        return int(tokens[position]), position + 1
    return 1, position
