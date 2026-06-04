from __future__ import annotations

import unittest

from lammps_mace.chemistry import molar_mass, parse_formula


class ChemistryTests(unittest.TestCase):
    def test_parse_simple_formula(self) -> None:
        self.assertEqual(parse_formula("AlF3"), {"Al": 1, "F": 3})

    def test_parse_parenthesized_formula(self) -> None:
        self.assertEqual(parse_formula("Ca(NO3)2"), {"Ca": 1, "N": 2, "O": 6})

    def test_molar_mass(self) -> None:
        self.assertAlmostEqual(molar_mass(parse_formula("LiCl")), 42.39, places=2)


if __name__ == "__main__":
    unittest.main()
