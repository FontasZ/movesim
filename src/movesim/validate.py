"""Check the body against published measurements of living people.

    uv run movesim validate        # prints the table and writes docs/validation.md

Each benchmark names its source and the range a good-enough body should fall in. The sources
come from different people, positions and methods, so they check parts of the body, not the
whole of it; the knee-tilt test (movesim.knee_tilt) is the lesson-level check.
"""

from dataclasses import dataclass

import numpy as np

from movesim.ligaments import Ligaments
from movesim.sim import ROOT, load_settled

LUMBAR = ["L5_S1", "L4_L5", "L3_L4", "L2_L3", "L1_L2"]
CERVICAL = ["C0_C1", "C1_C2", "C2_C3", "C3_C4", "C4_C5", "C5_C6", "C6_C7", "C7_T1"]

WATT_2022 = "Watt, Callaway & Williams 2022, in-vivo lumbar passive stiffness meta-analysis (Tables 5-7)"
FUJII_2007 = "Fujii, Sakaura, Mukai et al. 2007, Eur Spine J 16:1867; MRI of 10 volunteers, supine, pelvis fixed"
DE_BOE = "De Boe et al. 2016, Mains Libres; passive cervical axial rotation stiffness in 74 healthy adults (torque meter)"
HIP_STUDIES = ("Leite et al. 2012 (10.1 N·m/rad) and the Manual Therapy hip passive stiffness validity study "
               "(5.6 N·m/rad); isokinetic dynamometer, first 20° of rotation")


@dataclass
class Result:
    name: str
    measured: str
    model: str
    ok: bool
    source: str


# ------------------------------------------------------------------ static moment-angle curves

class Chain:
    """Joints in series, all carrying the same moment, as a region of the spine does."""

    def __init__(self, model, joints, side):
        self.lig = Ligaments(model)
        self.idx = [self.lig.names.index(j) for j in joints]
        self.k = np.array([model.jnt_stiffness[model.joint(j).id] for j in joints])
        self.side = side
        r = self.lig.r_hi if side > 0 else self.lig.r_lo
        self.ranges = r[self.idx]

    def moment(self, i, d):
        q = self.lig.rest.copy()
        q[self.idx[i]] += self.side * d
        return self.k[i] * d - self.side * self.lig.torque(q)[self.idx[i]]

    def angle(self, m):
        """Total bend (rad) of the chain under moment m."""
        total = 0.0
        for i, r in enumerate(self.ranges):
            lo, hi = 0.0, r * 1.3
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if self.moment(i, mid) < m else (lo, mid)
            total += lo
        return total

    def quartile_stiffness(self):
        """Secant stiffness (N·m/°) over each quarter of the chain's range."""
        full = self.ranges.sum()
        ms = np.linspace(0, 200, 4001)
        angles = np.array([self.angle(m) for m in ms])
        marks = [np.interp(f * full, angles, ms) for f in (0, 0.25, 0.5, 0.75, 1.0)]
        return [(b - a) / np.degrees(full / 4) for a, b in zip(marks, marks[1:])]


def lumbar_curves(model):
    data = {"flexion": ("bend", +1, [(0.34, 0.00, 0.68), (0.65, 0.29, 1.02), (1.28, 0.69, 1.87), (2.36, 1.88, 2.84)]),
            "side-bending": ("side", +1, [(0.32, 0.17, 0.47), (0.83, 0.46, 1.20), (2.03, 1.48, 2.58), (5.03, 4.62, 5.43)]),
            "axial rotation": ("twist", +1, [(0.21, 0.10, 0.32), (0.34, 0.32, 0.36), (0.71, 0.36, 1.07), (1.33, 0.67, 2.00)])}
    out = []
    for label, (plane, side, quartiles) in data.items():
        chain = Chain(model, [f"{j}_{plane}" for j in LUMBAR], side)
        ours = chain.quartile_stiffness()
        for q, ((mean, lo, hi), m) in enumerate(zip(quartiles, ours), 1):
            band = "95% CI" if not (label == "axial rotation" and q > 2) else "±50%, one study"
            out.append(Result(f"lumbar {label}, quartile {q} of range", f"{mean:.2f} N·m/° ({band} {lo:.2f}-{hi:.2f})",
                              f"{m:.2f} N·m/°", bool(lo <= m <= hi), WATT_2022))
    return out


def cervical_rotation(model):
    chain = Chain(model, [f"{j}_twist" for j in CERVICAL], +1)
    full = chain.ranges.sum()
    ms = np.linspace(0, 30, 3001)
    angles = np.array([chain.angle(m) for m in ms])
    average = np.interp(full, angles, ms) / np.degrees(full)
    return [Result("neck axial rotation, average over the range", "0.07 N·m/° (SD 0.02)", f"{average:.3f} N·m/°",
                   bool(0.05 <= average <= 0.09), DE_BOE)]


def hip_rotation(model):
    from movesim.build_model import HIP_TURN_TISSUE, HIP_TURN_TONE

    chain = Chain(model, ["hip_inrot_L"], +1)
    k_total = chain.moment(0, np.radians(20)) / np.radians(20)
    tissue = k_total - HIP_TURN_TONE
    return [Result("hip rotation stiffness over the first 20° (tissue only)", "5.6-10.1 N·m/rad",
                   f"{tissue:.1f} N·m/rad (+{HIP_TURN_TONE:g} tone in hook lying)", bool(5.6 <= tissue <= 10.1), HIP_STUDIES)]


# ------------------------------------------------------------------ simulated tests

def trunk_rotation_45(model):
    """The trunk turned 45° on a held pelvis: how much each lumbar level twists.

    Static: every level from the pelvis to T1 carries the same twisting moment, so each gives way
    according to its own range and passive tissue, as in the MRI study's slow, held positions.
    """
    joints = [f"{j}_twist" for j in LUMBAR] + [f"T{i}_{b}_twist" for i, b in
                                               [(12, "L1")] + [(n, f"T{n + 1}") for n in range(11, 0, -1)]]
    chain = Chain(model, joints, +1)
    lo, hi = 0.0, 200.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if np.degrees(chain.angle(m)) < 45 else (lo, m)
    per_level = []
    for i in range(len(LUMBAR)):
        a, b = 0.0, chain.ranges[i] * 1.3
        for _ in range(60):
            mid = 0.5 * (a + b)
            a, b = (mid, b) if chain.moment(i, mid) < lo else (a, mid)
        per_level.append(np.degrees(a))
    return [Result("lumbar twist per level, trunk turned 45° on a held pelvis",
                   "1.2-1.7° at each level", f"{min(per_level):.1f}-{max(per_level):.1f}° "
                   f"({', '.join(f'{t:.1f}' for t in per_level)} from L5-S1 up; lumbar total {sum(per_level):.1f}°)",
                   all(1.2 <= t <= 1.7 for t in per_level), FUJII_2007)]


# ------------------------------------------------------------------ report

def run():
    model, _ = load_settled()
    results = lumbar_curves(model) + cervical_rotation(model) + hip_rotation(model)
    results += trunk_rotation_45(model)
    return results


def report(results):
    passed = sum(r.ok is True for r in results)
    checked = sum(r.ok is not None for r in results)
    lines = ["# Validation against published measurements", "",
             "Generated by `uv run movesim validate`. Each line compares the body with a measurement "
             "of living people. The sources differ in subjects, positions and methods, so a pass means "
             "the part behaves within the published range, not that the whole body is proven.", "",
             f"**{passed} of {checked} checks within range.**", "",
             "| check | published | this body | |", "|---|---|---|---|"]
    for r in results:
        mark = {True: "ok", False: "**off**", None: "info"}[r.ok]
        lines.append(f"| {r.name} | {r.measured} | {r.model} | {mark} |")
    lines += ["", "## Sources", ""] + [f"- {s}" for s in dict.fromkeys(r.source for r in results)]
    lines += ["", "Not yet checked: the thoracic spine (no in-vivo through-range data found; assumed to "
              "resist like the lumbar levels), hip side-to-side movement (taken equal to hip rotation), "
              "the shoulders and the knee and ankle springs.", ""]
    return "\n".join(lines)


def main():
    import json
    from dataclasses import asdict

    results = run()
    text = report(results)
    (ROOT / "docs" / "validation.md").write_text(text)
    # The Model page of the website shows these.
    (ROOT / "web" / "validation.json").write_text(json.dumps([asdict(r) for r in results], indent=1))
    print(text)


if __name__ == "__main__":
    main()
