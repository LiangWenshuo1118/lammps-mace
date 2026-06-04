from __future__ import annotations

from pathlib import Path


DEFAULT_SYSTEMS_CSV = (
    "system_index,salts,concentrations,concentration_type,total_atoms\n"
    "001,LiCl,100,mol%,128\n"
    "002,LiCl-KCl,60-40,mol%,128\n"
    "003,LiCl-KCl-AlF3,50-45-5,mol%,128\n"
)


def write_example_systems_csv(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(DEFAULT_SYSTEMS_CSV, encoding="utf-8")
    return output
