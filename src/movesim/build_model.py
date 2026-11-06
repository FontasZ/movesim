"""Build model/skeleton.xml from the anatomy tables below, then settle it in hook lying.

    uv run python -m movesim.build_model

The tables are the source of truth: edit them, not the generated XML.

Body frame (as if standing): x = forward (anterior), y = the person's left, z = up.
Right-side joint axes are the left ones mirrored: (x, y, z) -> (-x, y, -z), so a positive
angle means the same movement on both sides.

Sources (rounded, adult):
  - Per-level spinal ranges of motion: White & Panjabi, Clinical Biomechanics of the Spine
    (1990), representative values. "side" and "twist" are to each side.
  - Segment masses: de Leva (1996), for a 70 kg, 1.76 m adult.
  - Ankle and subtalar axes: Inman, The Joints of the Ankle (1976).
  - Spinal curves: relaxed values (sacral slope 35°, lumbar lordosis 40°, thoracic
    kyphosis 32°, cervical lordosis 30°). Lying down, gravity and the floor reshape them.
"""

import re
from dataclasses import dataclass

import mujoco
import numpy as np

from movesim.sim import KEY_PATH, MODEL_PATH

# ------------------------------------------------------------------ the spine
# Thoracic twist ranges (T1-T9 ±5°, T10 ±3°) follow the in-vitro segmental data of Wilke et al.
# 2017 (axial range 10-12° per segment over T1-T10) more than White & Panjabi's 2-5°, so that
# turning the chest on a held pelvis leaves the lumbar levels the 1.2-1.7° measured in vivo.

SACRAL_SLOPE = 35  # degrees the top of the sacrum faces forward-up


@dataclass
class Level:
    name: str      # the vertebra; its joint is with the one below
    below: str     # the vertebra (or sacrum) below
    region: str
    height: float  # vertebral body plus the disc below it (m), from BodyParts3D scaled to 1.76 m
    curve: float   # resting angle to the one below (deg, + = bent forward)
    flex: float    # range of motion (deg)
    ext: float
    side: float
    twist: float
    mass: float    # with the trunk slice or neck tissue around it (kg)


LEVELS = [
    Level("L5", "S1", "lumbar", 0.037, -14, 10, 7, 3, 2, 0.65),  # twist: 1° in W&P; in vivo >=1.2° (Fujii 2007)
    Level("L4", "L5", "lumbar", 0.033, -10, 10, 6, 6, 2, 0.65),
    Level("L3", "L4", "lumbar", 0.036, -7, 9, 6, 8, 2, 1.4),
    Level("L2", "L3", "lumbar", 0.033, -5, 9, 5, 6, 2, 1.4),
    Level("L1", "L2", "lumbar", 0.031, -4, 8, 4, 6, 2, 1.4),
    Level("T12", "L1", "thoracolumbar", 0.033, 0, 8, 4, 8, 2, 1.4),
    Level("T11", "T12", "thoracolumbar", 0.030, 1, 8, 4, 9, 2, 1.4),
    Level("T10", "T11", "thoracic", 0.030, 2, 5, 3, 7, 3, 1.4),
    Level("T9", "T10", "thoracic", 0.027, 3, 4, 2, 6, 4, 1.4),
    Level("T8", "T9", "thoracic", 0.027, 4, 4, 2, 6, 5, 1.4),
    Level("T7", "T8", "thoracic", 0.029, 4, 4, 2, 6, 5, 1.7),
    Level("T6", "T7", "thoracic", 0.026, 4, 3, 2, 6, 5, 1.6),
    Level("T5", "T6", "thoracic", 0.026, 4, 3, 2, 6, 5, 1.5),
    Level("T4", "T5", "thoracic", 0.024, 4, 3, 2, 6, 5, 1.35),
    Level("T3", "T4", "thoracic", 0.024, 3, 3, 1, 5, 5, 1.2),
    Level("T2", "T3", "thoracic", 0.022, 3, 3, 1, 5, 5, 1.0),
    Level("T1", "T2", "thoracic", 0.019, 3, 3, 1, 5, 5, 0.85),
    Level("C7", "T1", "cervical", 0.018, -2, 4, 3, 4, 3, 0.13),
    Level("C6", "C7", "cervical", 0.016, -4, 4, 3, 7, 6, 0.13),
    Level("C5", "C6", "cervical", 0.017, -5, 6, 5, 8, 7, 0.13),
    Level("C4", "C5", "cervical", 0.015, -5, 6, 5, 11, 7, 0.13),
    Level("C3", "C4", "cervical", 0.018, -5, 5, 4, 11, 7, 0.13),
    Level("C2", "C3", "cervical", 0.021, -4, 4, 3, 10, 3, 0.13),
    Level("C1", "C2", "cervical", 0.014, -3, 5, 5, 3, 40, 0.12),  # atlas on axis: most neck turning
]
SKULL = Level("skull", "C1", "cervical", 0.0, -2, 8, 15, 5, 3, 4.0)

# Passive resistance of the spine (all soft tissue, as nothing is active). At each level the
# moment resisting a bend of d from rest is c·(exp(a·d/r) − 1), with r that level's range on
# that side. Levels of one region share c, so in series they give way in proportion to their
# range. Per region and plane: (stiffness of the whole region at rest, N·m/°; sharpness a).
#   lumbar: fitted (weighted by the confidence intervals) to the in-vivo meta-analysis of Watt, Callaway & Williams 2022
#     (Med Biol Eng Comput 60:2133, Tables 5-7; stiffness per quartile of range).
#   thoracic: no in-vivo data found; assumed to resist like the lumbar levels (same c and a).
#   cervical: axial rotation averages 0.07 N·m/° over the range in healthy adults
#     (De Boe et al.); the other neck planes are assumed alike.
SPINE_PASSIVE = {
    "lumbar": {"bend": (0.243, 2.60), "side": (0.198, 3.67), "twist": (0.140, 2.33)},
}
CERVICAL_AVERAGE = (0.07, 2.5)  # N·m/° averaged over the range, sharpness
SPINE_DAMPING = {"lumbar": 2, "thoracolumbar": 2, "thoracic": 2, "cervical": 0.4}

PLANES = {"bend": ("ext", "flex"), "side": ("side", "side"), "twist": ("twist", "twist")}


def region_ranges(regions, plane):
    """Total range (deg) of the levels in these regions, for each side of a plane."""
    levels = [lv for lv in LEVELS + [SKULL] if lv.region in regions]
    lo, hi = PLANES[plane]
    return sum(getattr(lv, lo) for lv in levels), sum(getattr(lv, hi) for lv in levels)


def spine_ligaments(region, plane):
    """(c_lo, c_hi, a, start) for one level's joint in this region and plane."""
    if region == "cervical":
        avg, a = CERVICAL_AVERAGE
        r_lo, r_hi = region_ranges(["cervical"], plane)
        return (avg * r_lo / np.expm1(a), avg * r_hi / np.expm1(a), a, 0)
    stiffness, a = SPINE_PASSIVE["lumbar"][plane]  # thoracic levels: assumed like lumbar
    r_lo, r_hi = region_ranges(["lumbar"], plane)
    return (stiffness * r_lo / a, stiffness * r_hi / a, a, 0)


def limb_ligaments(end_moment, sharpness=3.0, start=0.5):
    """Limb joints: loose to halfway, then tightening to end_moment (N·m) at the end of range."""
    c = end_moment / np.expm1(sharpness)
    return (c, c, sharpness, start)


# Ribs, one pair per thoracic vertebra: half-width, depth to the front, how far the front end
# drops below the back end (m), and kind. True ribs reach the sternum, false ribs the costal arch.
RIBS = {
    "T1": (0.060, 0.050, 0.020, "true"), "T2": (0.080, 0.065, 0.030, "true"),
    "T3": (0.095, 0.080, 0.040, "true"), "T4": (0.110, 0.095, 0.050, "true"),
    "T5": (0.120, 0.105, 0.060, "true"), "T6": (0.130, 0.115, 0.070, "true"),
    "T7": (0.135, 0.120, 0.080, "true"), "T8": (0.140, 0.120, 0.080, "false"),
    "T9": (0.140, 0.115, 0.080, "false"), "T10": (0.135, 0.105, 0.070, "false"),
    "T11": (0.120, 0.000, 0.040, "floating"), "T12": (0.090, 0.000, 0.030, "floating"),
}

# ------------------------------------------------------------------ hips

# Passive stiffness of hip rotation near rest, N·m/rad: 5.6 and 10.1 in two in-vivo studies
# (isokinetic dynamometer, first 20°); taken for side-to-side movement too. Hook lying also
# needs a little muscle tone to keep the knees from falling open; it is kept separate.
HIP_TURN_TISSUE = 8.0
HIP_TURN_TONE = 8.0  # the smallest that keeps the knees standing is about 7

# ------------------------------------------------------------------ resting tone

# Gentle springs hold the legs in hook lying and let the arms lie by the sides.
# The leg values are tuned so the settled pose is hip 45°, knee 100° and the feet carry no
# sideways shear (see docs/knee-tilt-tuning.md).
TONE_DEG = {
    "hip_flex": 49.4, "knee": 112.5, "ankle": 40,
    "shoulder_abd": 15, "elbow": 10,
}

# ------------------------------------------------------------------ handles

# Points a movement can take hold of: an invisible hand on a spring that moves the point on
# purpose, holds it still, or just follows it (no force). Each is stiff enough to carry out
# the intention against the load it moves (the pelvis is heavy, a hand is light), so the
# passive response of the rest of the body is what the simulation shows.
# Site, stiffness (N/m), damping (N·s/m).
HANDLES = [
    ("knee_L", 2500, 120), ("knee_R", 2500, 120),
    ("forehead", 3000, 100),
    ("asis_L", 15000, 500), ("asis_R", 15000, 500),  # the front hip bones
    ("sternum", 5000, 200),
    ("palm_L", 800, 40), ("palm_R", 800, 40),
    ("shoulder_L", 2500, 100), ("shoulder_R", 2500, 100),
    ("ankle_L", 3000, 120), ("ankle_R", 3000, 120),
    ("elbow_L", 800, 40), ("elbow_R", 800, 40),
    ("chin", 3000, 100), ("temple", 3000, 100),
    ("pubis", 15000, 500),
    ("toes_L", 1500, 60), ("toes_R", 1500, 60),
]


def handle_bodies():
    return "\n".join(
        f'    <body name="guide_{site}" mocap="true">\n'
        f'      <site name="guide_{site}" type="sphere" size="0.02" material="hand"/>\n'
        f'      <geom type="sphere" size="0.025" material="hand" group="2"/>\n'
        f'    </body>' for site, _, _ in HANDLES)


def handle_tendons():
    return "\n".join(
        f'    <spatial name="guide_{site}" stiffness="{k}" damping="{b}" springlength="0" '
        f'rgba="0.2 0.7 0.3 1" width="0.003">\n'
        f'      <site site="guide_{site}"/>\n      <site site="{site}"/>\n    </spatial>'
        for site, k, b in HANDLES)


# ------------------------------------------------------------------ body parts touching each other

LEG_PARTS = ("thigh", "calf")
ARM_PARTS = ("arm", "elbow", "forearm", "hand")


def contact_pairs():
    """The legs can rest on each other, and an arm can meet either leg (elbow to knee)."""
    pairs = [("thigh_L", "thigh_R"), ("calf_L", "calf_R"), ("malleolus_L", "malleolus_R"),
             ("thigh_L", "calf_R"), ("calf_L", "thigh_R")]
    pairs += [(f"{a}_{s}", f"{l}_{t}") for a in ARM_PARTS for s in "LR" for l in LEG_PARTS for t in "LR"]
    return "\n".join(f'    <pair geom1="{g1}" geom2="{g2}"/>' for g1, g2 in pairs)


# ------------------------------------------------------------------ helpers


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, float).flatten())
    return q


def v(*xs):
    return " ".join(f"{x:.5g}" for x in np.concatenate([np.ravel(a) for a in xs]))


def mirror(axis, s):
    """Axis for side s (+1 left, -1 right), given the left one."""
    x, y, z = axis
    return (x, y, z) if s > 0 else (-x, y, -z)


def side_name(s):
    return "L" if s > 0 else "R"


class Xml:
    def __init__(self):
        self.lines, self.depth = [], 0

    def add(self, text):
        for line in text.strip("\n").splitlines():
            self.lines.append("  " * self.depth + line)

    def open(self, text):
        self.add(text)
        self.depth += 1

    def close(self, tag="body"):
        self.depth -= 1
        self.add(f"</{tag}>")

    def text(self):
        return "\n".join(self.lines)


def hinge(x, name, axis, lo, hi, cls=None, stiffness=None, damping=None, springref=None, pos=None,
          lig=None):
    attrs = [f'name="{name}"']
    if cls:
        attrs.append(f'class="{cls}"')
    attrs += [f'axis="{v(axis)}"', f'range="{lo:.4g} {hi:.4g}"']
    if pos is not None:
        attrs.append(f'pos="{v(pos)}"')
    if stiffness is not None:
        attrs.append(f'stiffness="{stiffness}"')
    if damping is not None:
        attrs.append(f'damping="{damping}"')
    if springref is not None:
        attrs.append(f'springref="{springref:.4g}"')
    if lig is not None:
        attrs.append(f'user="{v(lig)}"')
    x.add(f"<joint {' '.join(attrs)}/>")


# ------------------------------------------------------------------ body parts


def spine_poses():
    """Standing pose (rotation, origin) of every spinal body, in the pelvis frame."""
    poses, r, p = {}, np.eye(3), np.zeros(3)
    prev_h = 0.0
    for i, lv in enumerate(LEVELS + [SKULL]):
        tilt = lv.curve + (SACRAL_SLOPE if i == 0 else 0)
        p = p + r @ np.array([0, 0, prev_h])
        r = r @ rot_y(tilt)
        poses[lv.name] = (r.copy(), p.copy())
        prev_h = lv.height
    return poses


def vertebra(x, lv, i):
    damp = SPINE_DAMPING[lv.region]
    tilt = lv.curve + (SACRAL_SLOPE if i == 0 else 0)
    prev_h = LEVELS[i - 1].height if i > 0 else 0.0
    rg = lv.region
    x.open(f'<body name="{lv.name}" pos="0 0 {prev_h:.4g}" quat="{v(quat(rot_y(tilt)))}">')
    j = f"{lv.name}_{lv.below}"
    for plane, axis, lo, hi in (("bend", (0, 1, 0), -lv.ext, lv.flex), ("side", (1, 0, 0), -lv.side, lv.side),
                                ("twist", (0, 0, 1), -lv.twist, lv.twist)):
        hinge(x, f"{j}_{plane}", axis, lo, hi, stiffness=0, damping=damp, lig=spine_ligaments(lv.region, plane))

    h = lv.height
    zc = 0.45 * h
    if rg in ("lumbar", "thoracolumbar"):
        r = 0.024 if rg == "lumbar" else 0.021
        sp = (-0.075, -0.004)
    elif rg == "thoracic":
        t = int(lv.name[1:])
        r = 0.013 + 0.0006 * t
        sp = (-0.055, -0.03)
    else:
        r = 0.010
        sp = (-0.045 if lv.name == "C7" else -0.03, -0.01)
    # Bone: body, spinous process, transverse processes.
    x.add(f'<geom class="bone" type="cylinder" pos="0 0 {zc:.4g}" size="{r:.4g} {0.38 * h:.4g}"/>')
    x.add(f'<geom class="bone" type="capsule" size="0.005" fromto="{v(-r, 0, zc, sp[0], 0, zc + sp[1])}"/>')
    tp = 0.04 if rg != "cervical" else 0.025
    x.add(f'<geom class="bone" type="capsule" size="0.004" fromto="{v(-r, -tp, zc, -r, tp, zc)}"/>')

    # Mass of the trunk slice (or neck tissue) carried by this level.
    if rg == "cervical":
        x.add(f'<geom class="mass" type="cylinder" pos="0 0 {h / 2:.4g}" size="0.045 {h / 2:.4g}" mass="{lv.mass}"/>')
    elif lv.name in RIBS:
        w, d, _, _ = RIBS[lv.name]
        dd = max(d, 0.10)
        x.add(f'<geom class="mass" type="ellipsoid" pos="{v(0.4 * dd, 0, h / 2)}" '
              f'size="{v(0.55 * dd + 0.02, 0.9 * w, h / 2)}" mass="{lv.mass}"/>')
    else:
        x.add(f'<geom class="mass" type="ellipsoid" pos="{v(0.05, 0, h / 2)}" size="{v(0.09, 0.13, h / 2)}" mass="{lv.mass}"/>')

    # Flesh that meets the floor: muscle over the back of the ribs or beside the spine.
    if rg == "cervical":
        x.add(f'<geom name="{lv.name}_back" class="flesh" type="ellipsoid" pos="{v(-0.045, 0, h / 2)}" '
              f'size="{v(0.025, 0.04, h / 2 + 0.002)}"/>')
    else:
        cx, cy, sx, sy = (-0.07, 0.035, 0.032, 0.035) if lv.name[0] == "L" else (-0.06, 0.05, 0.042, 0.05)
        for s in (1, -1):
            x.add(f'<geom name="{lv.name}_back_{side_name(s)}" class="flesh" material="flesh_{side_name(s).lower()}" '
                  f'type="ellipsoid" pos="{v(cx, s * cy, h / 2)}" size="{v(sx, sy, h / 2 + 0.002)}"/>')

    if lv.name in RIBS and RIBS[lv.name][0] >= 0.1:
        # The sides of the chest, for lying on the side.
        w, d, _, _ = RIBS[lv.name]
        dd = max(d, 0.10)
        for s in (1, -1):
            x.add(f'<geom name="{lv.name}_side_{side_name(s)}" class="flesh" material="flesh_{side_name(s).lower()}" '
                  f'type="ellipsoid" pos="{v(0.3 * dd - 0.02, s * (w - 0.02), h / 2)}" '
                  f'size="{v(0.45 * dd, 0.04, h / 2 + 0.002)}"/>')

    if lv.name in RIBS:
        ribs(x, lv)


def ribs(x, lv):
    w, d, drop, kind = RIBS[lv.name]
    zc = 0.6 * lv.height
    for s in (1, -1):
        pts = [(-0.005, s * 0.02, zc), (-0.045, s * (0.45 * w + 0.02), zc - 0.005)]
        if kind == "floating":
            pts.append((-0.01, s * w, zc - drop))
        else:
            pts += [(0.3 * d - 0.02, s * w, zc - 0.25 * drop), (0.8 * d, s * 0.6 * w, zc - 0.7 * drop)]
            end = (d, s * 0.02, zc - drop) if kind == "true" else (0.95 * d, s * 0.35 * w, zc - drop)
            pts.append(end)
        for a, b in zip(pts, pts[1:]):
            x.add(f'<geom class="rib" fromto="{v(a, b)}"/>')
    if lv.name in ("T2", "T4", "T6"):  # the sternum, in three pieces riding on the ribs
        x.add(f'<geom class="bone" type="box" pos="{v(d + 0.005, 0, zc - drop - 0.02)}" size="0.006 0.015 0.025"/>')
        if lv.name == "T4":
            x.add(f'<site name="sternum" pos="{v(d + 0.015, 0, zc - drop - 0.02)}" size="0.012"/>')


def skull(x):
    lv = SKULL
    damp = SPINE_DAMPING["cervical"]
    x.open(f'<body name="skull" pos="0 0 {LEVELS[-1].height:.4g}" quat="{v(quat(rot_y(lv.curve)))}">')
    for plane, axis, lo, hi in (("bend", (0, 1, 0), -lv.ext, lv.flex), ("side", (1, 0, 0), -lv.side, lv.side),
                                ("twist", (0, 0, 1), -lv.twist, lv.twist)):
        hinge(x, f"C0_C1_{plane}", axis, lo, hi, stiffness=0, damping=damp, lig=spine_ligaments("cervical", plane))
    x.add(f"""
<geom class="mass" type="ellipsoid" pos="0.02 0 0.04" size="0.095 0.075 0.10" mass="{lv.mass}"/>
<geom class="bone" type="ellipsoid" pos="0.01 0 0.05" size="0.095 0.075 0.09"/>
<geom class="bone" type="ellipsoid" pos="0.065 0 -0.01" size="0.045 0.05 0.055"/>
<geom class="bone" type="sphere" pos="0.085 0.03 0.04" size="0.012" rgba="0.2 0.2 0.2 1"/>
<geom class="bone" type="sphere" pos="0.085 -0.03 0.04" size="0.012" rgba="0.2 0.2 0.2 1"/>
<!-- The back of the head as a sphere centred on the head's mass, so it rolls like a real head
     on the floor. It is too wide at the sides (skull 7 cm, sphere 11 cm), so on a pillow the
     head rests on its true outline instead: 11 cm to the back and top, 8 cm to the sides. A
     narrower outline on the floor would let a face-up head roll to one side and stay there.
     Contact bits: 4 = floor and chair only, 2 = pillow only (other flesh, 1, touches all). -->
<geom name="occiput" class="flesh" type="sphere" pos="0.02 0 0.04" size="0.11" contype="4"/>
<geom name="head_side" class="flesh" type="ellipsoid" pos="0.02 0 0.04" size="0.11 0.08 0.11" contype="2"/>
<site name="forehead" pos="0.1 0 0.08" size="0.012"/>
<site name="chin" pos="0.085 0 -0.05" size="0.012"/>
<site name="temple" pos="0.06 0.07 0.07" size="0.012"/>""")
    x.close()


def shoulder_and_arm(x, s, t2_pose):
    """Collarbone, shoulder blade and arm on side s, hung from the sternum at T2.

    These bodies use standing axes, whatever the tilt of T2.
    """
    n = side_name(s)
    r_t2, p_t2 = t2_pose
    t2_center = p_t2 + r_t2 @ np.array([0, 0, 0.5 * LEVELS[[l.name for l in LEVELS].index("T2")].height])
    sc = t2_center + np.array([0.075, s * 0.025, -0.02])
    x.open(f'<body name="clavicle_{n}" pos="{v(r_t2.T @ (sc - p_t2))}" quat="{v(quat(r_t2.T))}">')
    hinge(x, f"sc_elev_{n}", mirror((1, 0, 0), s), -10, 40, cls="girdle")
    hinge(x, f"sc_protract_{n}", mirror((0, 0, -1), s), -25, 25, cls="girdle")
    hinge(x, f"sc_rot_{n}", mirror((0, 1, 0), s), -5, 40, cls="girdle")
    ac = np.array([-0.025, s * 0.14, 0.015])
    x.add(f'<geom class="bone" type="capsule" fromto="{v(0, 0, 0, ac)}" size="0.008"/>')
    x.add(f'<geom class="mass" type="capsule" fromto="{v(0, 0, 0, ac)}" size="0.02" mass="0.3"/>')

    x.open(f'<body name="scapula_{n}" pos="{v(ac)}">')
    for ax, nm in (((1, 0, 0), "tilt"), ((0, 1, 0), "tip"), ((0, 0, 1), "turn")):
        hinge(x, f"ac_{nm}_{n}", mirror(ax, s), -15, 15, cls="girdle_ac")
    x.add(f"""
<geom class="bone" type="ellipsoid" pos="{v(-0.11, -s * 0.06, -0.07)}" size="0.006 0.05 0.08"/>
<geom class="bone" type="capsule" fromto="{v(0, 0, 0, -0.10, -s * 0.09, -0.03)}" size="0.006"/>
<geom class="mass" type="ellipsoid" pos="{v(-0.11, -s * 0.06, -0.07)}" size="0.03 0.06 0.08" mass="0.8"/>
<geom name="scapula_{n}" class="flesh" material="flesh_{n.lower()}" type="ellipsoid" pos="{v(-0.14, -s * 0.065, -0.07)}" size="0.03 0.06 0.08"/>
<site name="shoulder_{n}" size="0.012"/>""")

    gh = (0.0, s * 0.005, -0.045)
    x.open(f'<body name="humerus_{n}" pos="{v(gh)}">')
    hinge(x, f"shoulder_flex_{n}", mirror((0, -1, 0), s), -50, 170, cls="shoulder")
    hinge(x, f"shoulder_abd_{n}", mirror((1, 0, 0), s), -20, 170, cls="shoulder", springref=TONE_DEG["shoulder_abd"])
    hinge(x, f"shoulder_inrot_{n}", mirror((0, 0, -1), s), -60, 70, cls="shoulder")
    x.add(f"""
<geom class="bone" type="sphere" size="0.022"/>
<geom class="bone" type="capsule" fromto="0 0 0  0 0 -0.29" size="0.011"/>
<geom class="mass" type="capsule" fromto="0 0 -0.02  0 0 -0.27" size="0.045" mass="1.9"/>
<geom name="arm_{n}" class="flesh" material="flesh_{n.lower()}" type="ellipsoid" pos="-0.01 0 -0.15" size="0.045 0.045 0.13"/>
<geom name="elbow_{n}" class="flesh" type="sphere" pos="-0.02 0 -0.29" size="0.025"/>
<geom name="deltoid_{n}" class="flesh" material="flesh_{n.lower()}" type="sphere" pos="{v(0, s * 0.02, -0.03)}" size="0.045"/>""")

    x.open('<body name="forearm_{0}" pos="0 0 -0.29">'.format(n))
    hinge(x, f"elbow_{n}", mirror((0, -1, 0), s), 0, 145, cls="elbow", springref=TONE_DEG["elbow"])
    hinge(x, f"pronation_{n}", mirror((0, 0, -1), s), -10, 150, cls="forearm")
    x.add(f'<site name="elbow_{n}" pos="-0.02 0 0" size="0.012"/>')
    x.add(f"""
<geom class="bone" type="capsule" fromto="0 {s * 0.012:.3g} 0  0 {s * 0.02:.3g} -0.245" size="0.008"/>
<geom class="bone" type="capsule" fromto="0 {-s * 0.012:.3g} 0  0 {-s * 0.005:.3g} -0.245" size="0.008"/>
<geom class="mass" type="capsule" fromto="0 0 -0.02  0 0 -0.23" size="0.035" mass="1.15"/>
<geom name="forearm_{n}" class="flesh" material="flesh_{n.lower()}" type="ellipsoid" pos="-0.005 0 -0.12" size="0.03 0.035 0.11"/>""")

    x.open(f'<body name="hand_{n}" pos="0 0 -0.245">')
    hinge(x, f"wrist_flex_{n}", mirror((0, -1, 0), s), -70, 80, cls="wrist")
    hinge(x, f"wrist_dev_{n}", mirror((1, 0, 0), s), -20, 30, cls="wrist")
    x.add(f"""
<geom class="bone" type="box" pos="0 0 -0.08" size="0.012 0.042 0.085"/>
<geom class="mass" type="box" pos="0 0 -0.08" size="0.015 0.045 0.09" mass="0.45"/>
<geom name="hand_{n}" class="flesh" material="flesh_{n.lower()}" type="ellipsoid" pos="-0.005 0 -0.08" size="0.018 0.045 0.08"/>
<site name="palm_{n}" pos="0.01 0 -0.08" size="0.012"/>""")
    for _ in range(5):
        x.close()


def leg(x, s):
    n = side_name(s)
    hip = (0.01, s * 0.08, -0.10)
    x.open(f'<body name="thigh_{n}" pos="{v(hip)}">')
    hinge(x, f"hip_flex_{n}", mirror((0, -1, 0), s), -20, 125, cls="hip", springref=TONE_DEG["hip_flex"])
    hinge(x, f"hip_abd_{n}", mirror((1, 0, 0), s), -30, 45, cls="hip_turn")
    hinge(x, f"hip_inrot_{n}", mirror((0, 0, -1), s), -45, 40, cls="hip_turn")
    x.add(f"""
<geom class="bone" type="sphere" size="0.024"/>
<geom class="bone" type="capsule" fromto="0 0 0  0 0 -0.46" size="0.015"/>
<geom class="bone" type="capsule" fromto="0 -0.025 -0.45  0 0.025 -0.45" size="0.02"/>
<geom class="mass" type="capsule" fromto="0 0 -0.03  0 0 -0.43" size="0.07" mass="10"/>
<site name="knee_{n}" pos="0.03 0 -0.46" size="0.015"/>
<geom name="thigh_{n}" class="flesh" material="flesh_{n.lower()}" type="ellipsoid" pos="0 0 -0.23" size="0.075 0.075 0.21"/>""")

    x.open(f'<body name="shin_{n}" pos="0 0 -0.46">')
    hinge(x, f"knee_{n}", (0, 1, 0), 0, 150, cls="knee", springref=TONE_DEG["knee"])
    # With the knee bent, the shin can turn about its own length.
    hinge(x, f"tibia_rot_{n}", mirror((0, 0, 1), s), -25, 25, cls="tibia_rot")
    x.add(f"""
<geom class="bone" type="capsule" fromto="0 0 0  0 0 -0.40" size="0.014"/>
<geom class="bone" type="capsule" fromto="0 {s * 0.03:.3g} -0.02  0 {s * 0.03:.3g} -0.39" size="0.006"/>
<geom class="bone" type="sphere" pos="0.035 0 0.01" size="0.02"/>
<geom class="mass" type="capsule" fromto="0 0 -0.03  0 0 -0.36" size="0.05" mass="3.0"/>
<geom name="calf_{n}" class="flesh" material="flesh_{n.lower()}" type="ellipsoid" pos="-0.015 0 -0.15" size="0.05 0.05 0.15"/>
<geom name="malleolus_{n}" class="flesh" type="sphere" pos="{v(0, s * 0.03, -0.40)}" size="0.025"/>""")

    x.open(f'<body name="foot_{n}" pos="0 0 -0.40">')
    # Ankle (talocrural) axis runs slightly down and back toward the outer ankle bone;
    # the subtalar axis rises 42° forward and points 23° toward the inside of the foot.
    hinge(x, f"ankle_{n}", mirror((-0.10, 0.985, -0.14), s), -25, 50, cls="ankle", springref=TONE_DEG["ankle"])
    hinge(x, f"subtalar_{n}", mirror((0.684, -0.290, 0.669), s), -25, 15, cls="ankle", pos=(-0.01, 0, -0.03))
    m = -s  # toward the inside of the foot
    x.add(f"""
<geom class="bone" type="box" pos="0.035 0 -0.045" size="0.085 0.035 0.02"/>
<geom class="mass" type="box" pos="0.035 0 -0.04" size="0.085 0.04 0.03" mass="0.85"/>
<geom name="heel_{n}" class="sole" type="sphere" pos="-0.045 0 -0.045" size="0.03"/>
<geom name="met1_{n}" class="sole" type="sphere" pos="{v(0.12, m * 0.03, -0.06)}" size="0.015"/>
<geom name="met5_{n}" class="sole" type="sphere" pos="{v(0.10, s * 0.035, -0.06)}" size="0.015"/>
<geom name="midfoot_{n}" class="sole" type="sphere" pos="{v(0.03, s * 0.035, -0.06)}" size="0.015"/>
<site name="ankle_{n}" size="0.012"/>""")
    x.open(f'<body name="toes_{n}" pos="0.12 0 -0.05">')
    hinge(x, f"toes_{n}", (0, 1, 0), -60, 30, cls="toes")
    x.add(f"""
<geom class="bone" type="box" pos="0.025 0 -0.01" size="0.03 0.035 0.008"/>
<geom class="mass" type="box" pos="0.025 0 -0.01" size="0.03 0.04 0.012" mass="0.1"/>
<geom name="bigtoe_{n}" class="sole" type="sphere" pos="{v(0.05, m * 0.02, -0.015)}" size="0.01"/>
<site name="toes_{n}" pos="0.04 0 0" size="0.012"/>
<geom name="smalltoes_{n}" class="sole" type="sphere" pos="{v(0.035, s * 0.02, -0.015)}" size="0.01"/>""")
    for _ in range(4):
        x.close()


# ------------------------------------------------------------------ the whole model

HEADER = """<!--
  movesim skeleton, v0.2. GENERATED by `uv run python -m movesim.build_model`; edit the tables
  in src/movesim/build_model.py, not this file.

  Rigid segments only, no muscles. Joint limits, mild passive stiffness and damping stand in
  for soft tissue. Adult of about 1.76 m and 70 kg.

  Body frame (as if standing): x = forward (anterior), y = the person's left, z = up.
  The pelvis is placed lying on its back with the head toward world +x, face up (+z).
  The person's right side is world +y.

  Geom groups:
    0  bones (visible)
    1  "flesh" contact surfaces: the only geoms that touch the floor
    2  the guide's hands
    3  mass/inertia carriers (hidden, no contact)
    4  simple bone shapes, hidden when the real bone meshes are present

  Bone meshes: BodyParts3D, (c) The Database Center for Life Science, CC BY 4.0; see
  model/meshes/ATTRIBUTION.md.
-->"""

PREAMBLE = """
<mujoco model="movesim-skeleton">
  <compiler angle="degree" autolimits="true"/>
  <!-- Each joint's user values describe its passive tissue (see movesim/ligaments.py):
       c below rest, c above rest (N·m), sharpness, and where in the range tightening starts. -->
  <size nuser_jnt="4"/>
  <!-- noslip stops the slow creep that soft contacts otherwise allow under constant load. -->
  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic" impratio="10"
          noslip_iterations="5"/>

  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.80 0.76" rgb2="0.76 0.74 0.70"
             width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0"/>
    <material name="bone" rgba="0.93 0.90 0.82 1"/>
    <material name="flesh" rgba="0.80 0.55 0.45 0.25"/>
    <material name="flesh_l" rgba="0.35 0.55 0.85 0.30"/>
    <material name="flesh_r" rgba="0.85 0.45 0.35 0.30"/>
    <material name="hand" rgba="0.2 0.7 0.3 0.6"/>
  </asset>

  <default>
    <!-- Geoms carry no mass unless they are mass carriers. -->
    <geom density="0" contype="0" conaffinity="0"/>
    <joint armature="0.01" limited="true"/>
    <default class="bone">
      <geom material="bone" group="0"/>
      <default class="rib">
        <geom type="capsule" size="0.004"/>
      </default>
    </default>
    <default class="flesh">
      <!-- Soft contact stands in for skin, fat and muscle over the bones: it gives way over
           about 2 cm, so the load spreads over several levels as on a real back. -->
      <geom material="flesh" group="1" contype="1" conaffinity="0" condim="3"
            friction="0.9 0.01 0.001" solref="0.02 1" solimp="0.6 0.95 0.02"/>
      <default class="sole">
        <!-- Bare feet on a mat grip well, also against pivoting. -->
        <geom condim="4" friction="1.2 0.02 0.001"/>
      </default>
    </default>
    <default class="mass">
      <geom group="3" rgba="0.5 0.5 0.5 0.1"/>
    </default>
    <default class="hip"><joint stiffness="60" damping="5" user="LIMB:30"/></default>
    <!-- Hip side-to-side and rotation: tissue stiffness plus the small tone that keeps the knees up. -->
    <default class="hip_turn"><joint stiffness="HIP_TURN" damping="5" user="LIMB:30"/></default>
    <default class="knee"><joint stiffness="30" damping="3" user="LIMB:20"/></default>
    <default class="tibia_rot"><joint stiffness="10" damping="1" user="LIMB:8"/></default>
    <default class="ankle"><joint stiffness="10" damping="1" user="LIMB:15"/></default>
    <default class="toes"><joint stiffness="2" damping="0.2" user="LIMB:3"/></default>
    <default class="girdle"><joint stiffness="20" damping="1" user="LIMB:8"/></default>
    <default class="girdle_ac"><joint stiffness="30" damping="1" user="LIMB:8"/></default>
    <default class="shoulder"><joint stiffness="5" damping="1" user="LIMB:10"/></default>
    <default class="elbow"><joint stiffness="2" damping="0.5" user="LIMB:8"/></default>
    <default class="forearm"><joint stiffness="1" damping="0.2" user="LIMB:3"/></default>
    <default class="wrist"><joint stiffness="1" damping="0.2" user="LIMB:4"/></default>
  </default>

  <worldbody>
    <light name="key" pos="0.6 -1.2 2.5" dir="-0.2 0.5 -1" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"
           castshadow="true"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor" contype="0" conaffinity="5"
          condim="3" friction="0.9 0.01 0.001"/>

    <!-- Props, parked under the floor until a starting position places them. -->
    <body name="pillow" mocap="true" pos="0 0 -2">
      <geom type="box" size="0.2 0.13 0.05" contype="0" conaffinity="3" condim="3" friction="0.9 0.01 0.001"
            rgba="0.78 0.74 0.66 1"/>
    </body>
    <body name="chair" mocap="true" pos="0 0 -2">
      <!-- body origin at the middle of the seat's top surface -->
      <geom type="box" pos="0 0 -0.02" size="0.22 0.22 0.02" contype="0" conaffinity="5" condim="3"
            friction="0.9 0.01 0.001" rgba="0.55 0.42 0.3 1"/>
      <geom type="box" pos="0.19 0.19 -0.24" size="0.02 0.02 0.22" rgba="0.55 0.42 0.3 1"/>
      <geom type="box" pos="0.19 -0.19 -0.24" size="0.02 0.02 0.22" rgba="0.55 0.42 0.3 1"/>
      <geom type="box" pos="-0.19 0.19 -0.24" size="0.02 0.02 0.22" rgba="0.55 0.42 0.3 1"/>
      <geom type="box" pos="-0.19 -0.19 -0.24" size="0.02 0.02 0.22" rgba="0.55 0.42 0.3 1"/>
    </body>

    <!-- The guide's hands: mocap targets, each tied to one point of the body by a soft spring. -->
HANDLE_BODIES
"""

PELVIS = """
<body name="pelvis" pos="0 0 0.12" quat="0 0.7071068 0 0.7071068">
  <freejoint name="pelvis"/>
  <!-- Origin at the centre of the top of the sacrum (S1), axes as if standing. -->
  <geom class="mass" type="ellipsoid" pos="0.0 0 -0.06" size="0.10 0.16 0.10" mass="6.5"/>
  <geom class="bone" type="ellipsoid" pos="-0.045 0 -0.05" size="0.022 0.045 0.065" euler="0 -25 0"/>
  <geom class="bone" type="ellipsoid" pos="-0.01  0.075 -0.02" size="0.02 0.065 0.075" euler="0 0 -30"/>
  <geom class="bone" type="ellipsoid" pos="-0.01 -0.075 -0.02" size="0.02 0.065 0.075" euler="0 0 30"/>
  <geom class="bone" type="capsule" fromto="0.04 0.085 -0.10  0.075 0.02 -0.14" size="0.01"/>
  <geom class="bone" type="capsule" fromto="0.04 -0.085 -0.10  0.075 -0.02 -0.14" size="0.01"/>
  <geom class="bone" type="capsule" fromto="0.075 0.02 -0.14  0.075 -0.02 -0.14" size="0.012"/>
  <geom class="bone" type="sphere" pos="0.027 0.085 -0.101" size="0.028" rgba="0.93 0.90 0.82 0.4"/>
  <geom class="bone" type="sphere" pos="0.027 -0.085 -0.101" size="0.028" rgba="0.93 0.90 0.82 0.4"/>
  <geom name="sacrum" class="flesh" type="ellipsoid" pos="-0.06 0 -0.05" size="0.04 0.04 0.06"/>
  <geom name="glute_L" class="flesh" material="flesh_l" type="ellipsoid" pos="-0.04 0.07 -0.10" size="0.065 0.07 0.085"/>
  <geom name="glute_R" class="flesh" material="flesh_r" type="ellipsoid" pos="-0.04 -0.07 -0.10" size="0.065 0.07 0.085"/>
  <site name="asis_L" pos="0.07 0.11 -0.03" size="0.012"/>
  <site name="asis_R" pos="0.07 -0.11 -0.03" size="0.012"/>
  <site name="pubis" pos="0.075 0 -0.14" size="0.012"/>
"""

TENDONS = """
  <!-- Body parts that can touch each other (body parts otherwise only touch the floor and props). -->
  <contact>
CONTACT_PAIRS
  </contact>

  <!-- The guide's hands: a soft spring from each hand to its point on the body. -->
  <tendon>
HANDLE_TENDONS
  </tendon>

  <include file="keyframes.xml"/>
</mujoco>
"""


MESH_DIR = MODEL_PATH.parent / "meshes"


def add_meshes(xml):
    """Show real bone shapes (model/meshes, made by movesim.bones) instead of the simple ones."""
    names = sorted(p.stem for p in MESH_DIR.glob("*.stl"))
    if not names:
        return xml
    assets = "".join(f'\n    <mesh name="{n}" file="meshes/{n}.stl" inertia="shell"/>' for n in names)
    xml = xml.replace("\n  </asset>", assets + "\n  </asset>", 1)
    xml = xml.replace('<geom material="bone" group="0"/>', '<geom material="bone" group="4"/>', 1)
    xml = xml.replace('    <default class="mass">',
                      '    <default class="bonemesh">\n      <geom type="mesh" material="bone" group="0"/>\n'
                      '    </default>\n    <default class="mass">', 1)
    for n in names:
        xml = re.sub(rf'(\n(\s*)<body name="{n}"[^>]*>)', rf'\1\n\2  <geom class="bonemesh" mesh="{n}"/>', xml, count=1)
    return xml


def build(meshes=True):
    poses = spine_poses()
    x = Xml()
    x.depth = 2
    x.open(PELVIS.strip("\n").replace("\n  ", "\n").rstrip())
    for s in (1, -1):
        leg(x, s)
    for i, lv in enumerate(LEVELS):
        vertebra(x, lv, i)
        if lv.name == "T2":
            for s in (1, -1):
                shoulder_and_arm(x, s, poses["T2"])
    skull(x)
    for _ in LEVELS:
        x.close()
    x.close()  # pelvis
    preamble = PREAMBLE.replace('"HIP_TURN"', f'"{HIP_TURN_TISSUE + HIP_TURN_TONE:g}"')
    preamble = re.sub(r"LIMB:([\d.]+)", lambda m: v(limb_ligaments(float(m.group(1)))), preamble)
    preamble = preamble.replace("HANDLE_BODIES", handle_bodies())
    tendons = TENDONS.replace("HANDLE_TENDONS", handle_tendons()).replace("CONTACT_PAIRS", contact_pairs())
    xml = HEADER + preamble + x.text() + "\n  </worldbody>\n" + tendons
    return add_meshes(xml) if meshes else xml


def main():
    MODEL_PATH.write_text(build())
    KEY_PATH.write_text("<mujoco>\n  <keyframe/>\n</mujoco>\n")  # replaced by the settles below
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    segments = sum(1 for b in range(1, model.nbody) if model.body_mocapid[b] < 0)
    print(f"Wrote {MODEL_PATH.name}: {segments} body segments, {model.njnt} joints, "
          f"{model.body_subtreemass[model.body('pelvis').id]:.1f} kg.")
    from movesim import sim
    sim.main()


if __name__ == "__main__":
    main()
