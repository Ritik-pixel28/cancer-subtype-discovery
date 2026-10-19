"""Rebuild the reference DNA geometry from the bundled experimental PDB.

Atom positions retain the experimental geometry after rigid orientation; bonds
are inferred by covalent radii plus 0.45 angstrom tolerance, not experimental
bond-order annotations. Water atoms are excluded.
"""

from pathlib import Path
import numpy as np, json

p = Path(__file__).resolve().parents[1] / "app/assets"
atoms = []
for line in (p / "1BNA.pdb").read_text().splitlines():
    if line.startswith("ATOM  "):
        atoms.append(
            {
                "p": [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                "e": line[76:78].strip(),
                "chain": line[21],
                "residue": int(line[22:26]),
                "name": line[12:16].strip(),
            }
        )
x = np.array([a["p"] for a in atoms])
x -= x.mean(0)
_, _, vt = np.linalg.svd(x, full_matrices=False)
x = x @ vt.T
x = x[:, [1, 0, 2]]
for atom, xyz in zip(atoms, x):
    atom["p"] = [round(float(v), 3) for v in xyz]
radii = {"C": 0.76, "N": 0.71, "O": 0.66, "P": 1.07}
bonds = []
for i, a in enumerate(atoms):
    for j in range(i):
        b = atoms[j]
        d = np.linalg.norm(x[i] - x[j])
        limit = radii.get(a["e"], 0.75) + radii.get(b["e"], 0.75) + 0.45
        if 0.5 < d < limit:
            bonds.append([i, j])
(p / "dna.json").write_text(
    json.dumps(
        {"source": "RCSB PDB 1BNA", "atoms": atoms, "bonds": bonds},
        separators=(",", ":"),
    )
)
print(len(atoms), len(bonds), x.max(0) - x.min(0))
