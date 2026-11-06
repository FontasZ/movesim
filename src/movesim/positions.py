"""Starting positions: a pose, the muscle tone that holds it, and the props it needs.

Tissue alone does not hold a lying or sitting person in place: the knees of someone lying with
the feet standing would fall open, a sitting trunk would fold. The small tone a resting person
uses is kept explicit here, per position, as extra joint stiffness toward the pose. Each value
is about the least that holds the position (see `uv run movesim build` output).
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np


def quat_from_axes(x, y, z):
    """World orientation whose body x, y, z axes point along the given world vectors."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).astype(float).flatten())
    return q


X, Y, Z = np.eye(3)


@dataclass
class Position:
    help: str
    pelvis: np.ndarray                          # pelvis orientation (body x forward, y left, z up)
    joints: dict = field(default_factory=dict)  # deg; also where the tone pulls toward
    tone: dict = field(default_factory=dict)    # joint-name prefix -> extra stiffness, N·m/rad
    on: str = "floor"                           # "floor" or "chair"
    pillow: bool = False


ARMS_BY_SIDES = {"shoulder_abd_L": 15, "shoulder_abd_R": 15, "elbow_L": 10, "elbow_R": 10}

POSITIONS = {
    "hook_lying": Position(
        "Lying on the back, knees bent, feet standing on the floor, arms long by the sides.",
        quat_from_axes(Z, -Y, X)),  # face up, head toward +x
    "lying_long": Position(
        "Lying on the back with the legs long, arms long by the sides.",
        quat_from_axes(Z, -Y, X),
        {"hip_flex_L": 0, "hip_flex_R": 0, "knee_L": 3, "knee_R": 3, "ankle_L": 25, "ankle_R": 25,
         "hip_inrot_L": -10, "hip_inrot_R": -10, **ARMS_BY_SIDES}),
    "side_lying_left": Position(
        "Lying on the left side, knees bent and resting one on the other, head on a pillow, the "
        "left arm forward on the floor, the right hand on the floor in front of the chest.",
        quat_from_axes(-Y, -Z, X),
        {"hip_flex_L": 60, "hip_flex_R": 60, "knee_L": 85, "knee_R": 85, "ankle_L": 20, "ankle_R": 20,
         "shoulder_flex_L": 80, "elbow_L": 20, "shoulder_flex_R": 40, "elbow_R": 30,
         "shoulder_abd_L": 0, "shoulder_abd_R": 0,
         "pronation_R": 120,       # top palm on the floor
         "shoulder_inrot_L": -45},  # bottom palm on the floor: the arm turns out
        {"hip_flex": 40, "T": 300, "L": 300},  # keeps the chest from rolling over onto the floor
        pillow=True),
    "side_lying_right": Position(
        "Lying on the right side, knees bent and resting one on the other, head on a pillow, the "
        "right arm forward on the floor, the left hand on the floor in front of the chest.",
        quat_from_axes(Y, Z, X),
        {"hip_flex_L": 60, "hip_flex_R": 60, "knee_L": 85, "knee_R": 85, "ankle_L": 20, "ankle_R": 20,
         "shoulder_flex_R": 80, "elbow_R": 20, "shoulder_flex_L": 40, "elbow_L": 30,
         "shoulder_abd_L": 0, "shoulder_abd_R": 0,
         "pronation_L": 120,       # top palm on the floor
         "shoulder_inrot_R": -45},  # bottom palm on the floor: the arm turns out
        {"hip_flex": 40, "T": 300, "L": 300},  # keeps the chest from rolling over onto the floor
        pillow=True),
    "sitting_chair": Position(
        "Sitting on the front half of a chair, feet standing on the floor, arms hanging, upright.",
        quat_from_axes(X, Y, Z),
        {"hip_flex_L": 85, "hip_flex_R": 85, "knee_L": 90, "knee_R": 90, "ankle_L": 0, "ankle_R": 0,
         "shoulder_abd_L": 8, "shoulder_abd_R": 8, "elbow_L": 15, "elbow_R": 15},
        # Sitting upright is held: 3000 per spinal level is about the least that keeps the trunk
        # from folding forward (17 levels in series share the work of balancing the upper body).
        {"L": 3000, "T": 3000, "C": 150, "hip_flex": 600, "knee": 60},
        on="chair"),
    "hands_knees": Position(
        "On hands and knees: hands under the shoulders, knees under the hips, the tops of the "
        "feet on the floor, the back long, looking at the floor.",
        quat_from_axes(-Z, Y, X),  # face down, head toward +x
        {"hip_flex_L": 90, "hip_flex_R": 90, "knee_L": 90, "knee_R": 90, "ankle_L": 50, "ankle_R": 50,
         "shoulder_flex_L": 82, "shoulder_flex_R": 82, "shoulder_abd_L": -5, "shoulder_abd_R": -5,
         "elbow_L": 0, "elbow_R": 0, "wrist_flex_L": -70, "wrist_flex_R": -70,
         "pronation_L": 90, "pronation_R": 90},
        {"L": 3000, "T": 3000, "C": 150, "hip": 300, "knee": 100, "shoulder": 300, "elbow": 300,
         "sc_": 300, "ac_": 300, "wrist": 30}),
}


def apply_tone(model, position):
    """Point the tone springs at this position's pose and add its tone stiffness."""
    pos = POSITIONS[position]
    for name, deg in pos.joints.items():
        model.qpos_spring[model.joint(name).qposadr[0]] = np.radians(deg)
    for j in range(model.njnt):
        name = model.joint(j).name
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        for prefix, k in pos.tone.items():
            if name.startswith(prefix) and (prefix != "L" and prefix != "T" and prefix != "C"
                                            or name[1:2].isdigit()):
                model.jnt_stiffness[j] += k
