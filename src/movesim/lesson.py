"""Movement scripts: what a lesson asks the body to do, played through the physics.

A script starts in one of the starting positions (movesim.positions) and is a list of steps.
Each step has the teacher's words ("say") and the movements that go with them. A movement is
one of three kinds:

- a shorthand (knees tilting, the head rolling, ...): named paths for points of the body,
- a reach: any point of the body (a handle) goes toward another point or in a direction,
- a joint intention: a joint is turned by an amount (the hip turns in, the elbow bends); for
  the head and spine the turn is shared over the levels in proportion to their range.

Points are guided by invisible hands on springs; joint intentions by a soft pull toward the
wanted angle. Parts can also be held still (HOLDS). Everything else is passive: physics decides
where the rest goes.

    script = load_script("lessons/knees.json")
    play(script, "web/lessons/knees")   # writes motion.bin and meta.json for the web player
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from movesim.bones import MESH_BODIES
from movesim.positions import POSITIONS
from movesim.sim import hands_to_knees, load_settled

FPS = 30
UP = np.array([0, 0, 1.0])
ON_BACK = ("hook_lying", "lying_long")
HOOK = ("hook_lying",)
ANY = tuple(POSITIONS)
SIDES = ("left", "right", "both")


@dataclass
class Move:
    help: str      # what it means, for people and for the translator
    unit: str      # "deg" or "cm"
    default: float
    sides: tuple   # which values of "side" make sense
    positions: tuple = ANY
    joint: tuple = ()  # for joint intentions: (joint name pattern or spine region, sign)


MOVES = {
    # Shorthands for lying on the back.
    "knees_tilt": Move("Both knees tip together to one side over the feet (side = direction).",
                       "deg", 20, ("left", "right"), HOOK),
    "knee_open": Move("A knee falls outward, away from the midline.", "deg", 20, SIDES, HOOK),
    "knee_close": Move("A knee moves inward, across toward the other side.", "deg", 15, SIDES, HOOK),
    "foot_press": Move("The foot presses into the floor: that side of the pelvis lifts and is pushed a "
                       "little toward the head (amount = how far the pelvis lifts).", "cm", 3, SIDES, HOOK),
    "knee_to_chest": Move("The knee comes toward the chest; the foot leaves the floor.", "deg", 30, SIDES, ON_BACK),
    "head_roll": Move("The head rolls on the floor to one side, nose turning that way (side = direction).",
                      "deg", 30, ("left", "right"), ON_BACK),
    "head_lift": Move("The head lifts from the floor, chin toward the chest.", "deg", 20, ("none",), ON_BACK),
    "pelvis_flatten": Move("The pelvis tilts so the lower back comes toward the floor.", "deg", 8, ("none",), HOOK),
    "pelvis_arch": Move("The pelvis tilts so the lower back arches away from the floor.", "deg", 8, ("none",), HOOK),
    "pelvis_lift": Move("The pelvis lifts from the floor, as in a small bridge.", "cm", 5, ("none",), HOOK),
    "arm_lift": Move("A straight arm lifts from the floor toward the ceiling (90 = hand above the shoulder).",
                     "deg", 90, SIDES, ON_BACK),
    "shoulder_lift": Move("A shoulder lifts from the floor toward the ceiling, the arm following.",
                          "cm", 4, SIDES, ON_BACK),
    # Reach: any point toward another point or in a direction (see REACH_POINTS, DIRECTIONS).
    "reach": Move("A point of the body ('point') moves toward another point or in a direction ('toward'); "
                  "amount = how far in cm, 0 = all the way to the other point.", "cm", 10, ("none",)),
    # Joint intentions.
    "hip_bend": Move("The thigh comes up toward the belly (the hip bends).", "deg", 30, SIDES, joint=("hip_flex", 1)),
    "hip_straighten": Move("The hip straightens.", "deg", 20, SIDES, joint=("hip_flex", -1)),
    "hip_turn_in": Move("The leg turns in at the hip.", "deg", 20, SIDES, joint=("hip_inrot", 1)),
    "hip_turn_out": Move("The leg turns out at the hip.", "deg", 20, SIDES, joint=("hip_inrot", -1)),
    "hip_open": Move("The leg moves out to the side at the hip.", "deg", 20, SIDES, joint=("hip_abd", 1)),
    "hip_close": Move("The leg moves in toward the other leg at the hip.", "deg", 10, SIDES, joint=("hip_abd", -1)),
    "knee_bend": Move("The knee bends.", "deg", 30, SIDES, joint=("knee", 1)),
    "knee_straighten": Move("The knee straightens.", "deg", 30, SIDES, joint=("knee", -1)),
    "foot_flex": Move("The toes come up toward the shin (ankle bends).", "deg", 20, SIDES, joint=("ankle", -1)),
    "foot_point": Move("The foot points away from the shin.", "deg", 20, SIDES, joint=("ankle", 1)),
    "foot_roll_out": Move("The sole turns outward, standing on the inner edge.", "deg", 10, SIDES,
                          joint=("subtalar", 1)),
    "foot_roll_in": Move("The sole turns inward, standing on the outer edge.", "deg", 15, SIDES,
                         joint=("subtalar", -1)),
    "arm_raise": Move("The arm moves forward and up from the shoulder (180 = overhead).", "deg", 90, SIDES,
                      joint=("shoulder_flex", 1)),
    "arm_out": Move("The arm moves out to the side from the shoulder.", "deg", 45, SIDES, joint=("shoulder_abd", 1)),
    "arm_turn_in": Move("The arm turns in at the shoulder.", "deg", 30, SIDES, joint=("shoulder_inrot", 1)),
    "arm_turn_out": Move("The arm turns out at the shoulder.", "deg", 30, SIDES, joint=("shoulder_inrot", -1)),
    "elbow_bend": Move("The elbow bends.", "deg", 60, SIDES, joint=("elbow", 1)),
    "elbow_straighten": Move("The elbow straightens.", "deg", 30, SIDES, joint=("elbow", -1)),
    "palm_turn_down": Move("The forearm turns the palm down (from palm up; 90 = thumb up).", "deg", 90, SIDES,
                           joint=("pronation", 1)),
    "palm_turn_up": Move("The forearm turns the palm up.", "deg", 90, SIDES, joint=("pronation", -1)),
    "wrist_bend": Move("The hand bends toward the palm side.", "deg", 30, SIDES, joint=("wrist_flex", 1)),
    "wrist_back": Move("The hand bends back.", "deg", 30, SIDES, joint=("wrist_flex", -1)),
    "head_turn": Move("The head turns, nose toward one side, in the neck (side = direction).", "deg", 30,
                      ("left", "right"), joint=("neck_twist", -1)),
    "head_tilt": Move("The head tilts, ear toward one shoulder (side = direction).", "deg", 15,
                      ("left", "right"), joint=("neck_side", 1)),
    "head_nod": Move("The chin comes toward the chest, in the neck.", "deg", 15, ("none",), joint=("neck_bend", 1)),
    "head_back": Move("The head tips back, chin away from the chest.", "deg", 15, ("none",), joint=("neck_bend", -1)),
    "spine_twist": Move("The chest turns on the pelvis, the spine twisting (side = where the front of the "
                        "chest turns).", "deg", 20, ("left", "right"), joint=("trunk_twist", -1)),
    "spine_side_bend": Move("The spine bends sideways (side = toward which side).", "deg", 10, ("left", "right"),
                            joint=("trunk_side", 1)),
    "spine_round": Move("The spine rounds, bending forward.", "deg", 15, ("none",), joint=("trunk_bend", 1)),
    "spine_arch": Move("The spine arches, bending back.", "deg", 10, ("none",), joint=("trunk_bend", -1)),
}

REACH_POINTS = {  # handles a reach can move or aim at
    "knee_L": "left knee", "knee_R": "right knee", "ankle_L": "left ankle", "ankle_R": "right ankle",
    "toes_L": "left toes", "toes_R": "right toes", "palm_L": "left hand", "palm_R": "right hand",
    "elbow_L": "left elbow", "elbow_R": "right elbow", "shoulder_L": "left shoulder", "shoulder_R": "right shoulder",
    "forehead": "forehead", "chin": "chin", "sternum": "breastbone", "pubis": "pubic bone",
    "asis_L": "left front hip bone", "asis_R": "right front hip bone",
}
DIRECTIONS = ("up", "down", "head", "feet", "front", "back", "left", "right", "inward", "outward")

HOLDS = {  # what can be asked to stay still, and the points that hold it
    "head": ["forehead", "chin", "temple"], "pelvis": ["asis_L", "asis_R"], "chest": ["sternum"],
    "knees": ["knee_L", "knee_R"], "feet": ["ankle_L", "ankle_R"],
    "left_arm": ["palm_L"], "right_arm": ["palm_R"], "shoulders": ["shoulder_L", "shoulder_R"],
    "left_leg": ["knee_L", "ankle_L"], "right_leg": ["knee_R", "ankle_R"], "elbows": ["elbow_L", "elbow_R"],
}

STEP_DEFAULTS = {"moves": [], "hold": [], "seconds": 3.0, "stay": 1.0, "return": True, "repeat": 1,
                 "rest_after": 1.0, "release": False}

# How firmly an intention turns a joint, N·m/rad (per level for the spine and neck).
INTENT_STIFFNESS = {"hip": 800, "knee": 400, "ankle": 80, "subtalar": 40, "shoulder": 200, "elbow": 50,
                    "pronation": 20, "wrist": 15, "L": 2000, "T": 2000, "C": 150}


# ------------------------------------------------------------------ the script

def load_script(path):
    return check(json.loads(Path(path).read_text()))


def check(script):
    """Fill defaults and reject what the body can't do; returns the script."""
    start = script.setdefault("start", "hook_lying")
    if start not in POSITIONS:
        raise ValueError(f"unknown starting position {start!r}; choose from {sorted(POSITIONS)}")
    for i, step in enumerate(script["steps"], 1):
        for key, value in STEP_DEFAULTS.items():
            step.setdefault(key, value)
        for m in step["moves"]:
            spec = MOVES.get(m["move"])
            if spec is None:
                raise ValueError(f"step {i}: unknown move {m['move']!r}")
            if start not in spec.positions:
                raise ValueError(f"step {i}: {m['move']} needs one of {spec.positions}, not {start}")
            if m.get("side") in (None, "none") or m["side"] not in spec.sides:
                if m.get("side") not in (None, "none") and m["side"] not in spec.sides:
                    raise ValueError(f"step {i}: {m['move']} takes side {spec.sides}, not {m['side']!r}")
                m["side"] = spec.sides[0]
            if not m.get("amount"):
                m["amount"] = 0 if m["move"] == "reach" else spec.default
            if m["move"] == "reach":
                if m.get("point") not in REACH_POINTS:
                    raise ValueError(f"step {i}: reach needs a point from {sorted(REACH_POINTS)}")
                if m.get("toward") not in REACH_POINTS and m.get("toward") not in DIRECTIONS:
                    raise ValueError(f"step {i}: reach needs toward = a point or one of {DIRECTIONS}")
        for h in step["hold"]:
            if h not in HOLDS:
                raise ValueError(f"step {i}: can't hold {h!r}; choose from {sorted(HOLDS)}")
    return script


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "lesson"


# ------------------------------------------------------------------ paths for the handles

def rotate(p, axis, pivot, deg):
    a = axis / np.linalg.norm(axis)
    t = np.radians(deg)
    v = p - pivot
    return pivot + v * np.cos(t) + np.cross(a, v) * np.sin(t) + a * (a @ v) * (1 - np.cos(t))


def arc(p, axis, pivot, deg, toward):
    """A path rotating p about axis by deg, in whichever sense moves it along `toward`."""
    sign = 1 if (rotate(p, axis, pivot, deg) - p) @ toward >= 0 else -1
    return lambda f: rotate(p, axis, pivot, sign * deg * f)


def rolling(p, center, radius, deg, toward, axis):
    """A path for a point on a ball (the head) rolling on the floor about `axis`."""
    def at(sign, f):
        t = np.radians(sign * deg * f)
        shift = np.cross(axis * t, radius * UP)  # rolling without slipping moves the centre
        return rotate(p, axis, center, sign * deg * f) + shift
    sign = 1 if (at(1, 1) - p) @ toward >= 0 else -1
    return lambda f: at(sign, f)


def line(p, direction, cm):
    d = direction / np.linalg.norm(direction)
    return lambda f: p + d * cm / 100 * f


class Frame:
    """Directions of the body right now: the person's head, right and front, from the pelvis."""

    def __init__(self, data):
        r = data.body("pelvis").xmat.reshape(3, 3)
        self.front, self.left, self.head = r[:, 0].copy(), r[:, 1].copy(), r[:, 2].copy()
        self.right = -self.left

    def direction(self, name, point=""):
        side = {"L": self.left, "R": self.right}.get(point[-1:], np.zeros(3)) if point[-2:-1] == "_" else np.zeros(3)
        return {"up": UP, "down": -UP, "head": self.head, "feet": -self.head, "front": self.front,
                "back": -self.front, "left": self.left, "right": self.right,
                "inward": -side, "outward": side}[name]


def paths(model, data, move):
    """For a point-guiding movement: {handle site: path(f)} with f from 0 (start) to 1 (there)."""
    name, side, amount = move["move"], move["side"], move["amount"]
    sides = {"left": ["L"], "right": ["R"], "both": ["L", "R"]}.get(side, [])
    site = lambda s: data.site(s).xpos.copy()
    body = lambda b: data.body(b).xpos.copy()
    b = Frame(data)
    toward_side = b.right if side == "right" else b.left
    out = {}
    if name == "reach":
        p = site(move["point"])
        if move["toward"] in REACH_POINTS:
            gap = site(move["toward"]) - p
            dist = max(np.linalg.norm(gap) - 0.04, 0)  # stop just short of the other point
            cm = min(amount, dist * 100) if amount else dist * 100
            out[move["point"]] = line(p, gap, cm)
        else:
            out[move["point"]] = line(p, b.direction(move["toward"], move["point"]), amount)
    elif name == "knees_tilt":
        pivot = 0.5 * (body("foot_L") + body("foot_R"))
        for s in "LR":
            out[f"knee_{s}"] = arc(site(f"knee_{s}"), b.head, pivot, amount, toward_side)
    elif name in ("knee_open", "knee_close"):
        for s in sides:
            outward = b.right if s == "R" else b.left
            toward = outward if name == "knee_open" else -outward
            out[f"knee_{s}"] = arc(site(f"knee_{s}"), b.head, body(f"foot_{s}"), amount, toward)
    elif name == "foot_press":
        # Pressing a planted foot pushes up through the thigh: that side of the pelvis lifts and
        # moves a little toward the head. The knee and foot follow on their own.
        for s in sides:
            out[f"asis_{s}"] = line(site(f"asis_{s}"), UP + 0.4 * b.head, amount * np.hypot(1, 0.4))
    elif name == "knee_to_chest":
        for s in sides:
            out[f"knee_{s}"] = arc(site(f"knee_{s}"), b.right, body(f"thigh_{s}"), amount, b.head)
    elif name == "head_roll":
        head = data.geom("occiput")
        out["forehead"] = rolling(site("forehead"), head.xpos.copy(), model.geom("occiput").size[0],
                                  amount, toward_side, b.head)
    elif name == "head_lift":
        out["forehead"] = arc(site("forehead"), b.right, body("C4"), amount, UP)
    elif name in ("pelvis_flatten", "pelvis_arch"):
        pivot = 0.5 * (body("thigh_L") + body("thigh_R"))
        toward = b.head if name == "pelvis_flatten" else -b.head
        for s in "LR":
            out[f"asis_{s}"] = arc(site(f"asis_{s}"), b.right, pivot, amount, toward)
    elif name == "pelvis_lift":
        for s in "LR":
            out[f"asis_{s}"] = line(site(f"asis_{s}"), UP, amount)
    elif name == "arm_lift":
        for s in sides:
            out[f"palm_{s}"] = arc(site(f"palm_{s}"), b.right, body(f"humerus_{s}"), amount, UP)
    elif name == "shoulder_lift":
        for s in sides:
            out[f"shoulder_{s}"] = line(site(f"shoulder_{s}"), UP, amount)
    return out


# ------------------------------------------------------------------ joint intentions

NECK = ["C0_C1", "C1_C2", "C2_C3", "C3_C4", "C4_C5", "C5_C6", "C6_C7", "C7_T1"]
TRUNK = (["L5_S1", "L4_L5", "L3_L4", "L2_L3", "L1_L2", "T12_L1"]
         + [f"T{n}_T{n + 1}" for n in range(11, 0, -1)])


def intent_stiffness(joint):
    for prefix, k in INTENT_STIFFNESS.items():
        if joint.startswith(prefix) and (len(prefix) > 1 or joint[1:2].isdigit()):
            return k
    return 50


def joint_targets(model, data, move):
    """For a joint intention: {dof: (start angle, change, stiffness)}."""
    spec = MOVES[move["move"]]
    pattern, sign = spec.joint
    amount = np.radians(move["amount"]) * sign
    out = {}
    if pattern.startswith(("neck_", "trunk_")):
        group, plane = pattern.split("_")
        levels = NECK if group == "neck" else TRUNK
        change = amount * (-1 if move["side"] == "left" else 1)  # signs are given for "right"
        joints = [model.joint(f"{lv}_{plane}") for lv in levels]
        ranges = [j.range[1] if change > 0 else -j.range[0] for j in joints]
        for j, r in zip(joints, ranges):
            out[j.dofadr[0]] = (data.qpos[j.qposadr[0]], change * r / sum(ranges), intent_stiffness(j.name))
    else:
        for s in {"left": "L", "right": "R", "both": "LR"}.get(move["side"], "LR"):
            j = model.joint(f"{pattern}_{s}")
            out[j.dofadr[0]] = (data.qpos[j.qposadr[0]], amount, intent_stiffness(j.name))
    return out


# ------------------------------------------------------------------ playing a script

def ease(x):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))


class Player:
    """Steps the physics through a script and records what the body does."""

    def __init__(self, position="hook_lying"):
        self.position = position
        self.model, self.data = load_settled(position)
        m = self.model
        self.handles = [m.body(b).name[len("guide_"):] for b in range(m.nbody)
                        if m.body(b).name.startswith("guide_")]
        self.mocap = {h: m.body(f"guide_{h}").mocapid[0] for h in self.handles}
        self.bodies = list(MESH_BODIES)
        self.body_ids = [m.body(b).id for b in self.bodies]
        self.dof_qpos = {m.jnt_dofadr[j]: m.jnt_qposadr[j] for j in range(m.njnt)}
        self.spring0 = m.qpos_spring.copy()  # where the position's tone pulls when nothing is intended
        self.kept = {}         # handle -> position it keeps after a step that did not return
        self.kept_joints = {}  # dof -> (angle, stiffness) kept after a step that did not return
        self.frames, self.active, self.captions = [], [], []
        self.next_frame = 0.0

    def step_physics(self, targets, joints):
        """One physics step: targeted handles go to their targets, the rest follow their points;
        intended joints are pulled toward their angles."""
        m, d = self.model, self.data
        hands_to_knees(m, d)
        for h, p in {**self.kept, **targets}.items():
            d.mocap_pos[self.mocap[h]] = p
        d.qfrc_applied[:] = 0
        m.qpos_spring[:] = self.spring0
        for dof, (angle, k) in {**self.kept_joints, **joints}.items():
            # The tone that holds the position follows the intention instead of resisting it.
            m.qpos_spring[self.dof_qpos[dof]] = angle
            # A pull toward the wanted angle; the joint's own damping (handled implicitly by the
            # integrator) calms it, so none is added here, where it would be explicit and unstable.
            d.qfrc_applied[dof] = k * (angle - d.qpos[self.dof_qpos[dof]])
        mujoco.mj_step(m, d)
        if d.time >= self.next_frame:
            self.next_frame += 1 / FPS
            self.frames.append(np.concatenate([d.xpos[self.body_ids], d.xquat[self.body_ids]], axis=1))
            held = {**self.kept, **targets}
            self.active.append([[*(d.mocap_pos[self.mocap[h]]), 1.0 if h in held else 0.0]
                                for h in self.handles])

    def run_for(self, seconds, targets_at):
        end = self.data.time + seconds
        start = self.data.time
        while self.data.time < end:
            self.step_physics(*targets_at((self.data.time - start) / max(seconds, 1e-9)))

    def play_step(self, step):
        if step["release"]:
            self.kept, self.kept_joints = {}, {}
        t0 = self.data.time
        holds = {h: self.data.site(h).xpos.copy() for name in step["hold"] for h in HOLDS[name]}
        moves, joints = {}, {}
        for m in step["moves"]:
            if MOVES[m["move"]].joint:
                joints.update(joint_targets(self.model, self.data, m))
            else:
                moves.update(paths(self.model, self.data, m))
        for h in moves:
            self.kept.pop(h, None)
        for dof in joints:
            self.kept_joints.pop(dof, None)

        def at(f):
            return ({**holds, **{h: path(f) for h, path in moves.items()}},
                    {dof: (q0 + dq * f, k) for dof, (q0, dq, k) in joints.items()})

        for r in range(int(step["repeat"])):
            self.run_for(step["seconds"], lambda x: at(ease(x)))
            self.run_for(step["stay"], lambda x: at(1.0))
            if step["return"]:
                self.run_for(step["seconds"], lambda x: at(1 - ease(x)))
                if r < step["repeat"] - 1:
                    self.run_for(0.5, lambda x: at(0.0))
        if not step["return"]:
            self.kept.update({h: path(1.0) for h, path in moves.items()})
            self.kept_joints.update({dof: (q0 + dq, k) for dof, (q0, dq, k) in joints.items()})
        self.run_for(step["rest_after"], lambda x: ({}, {}))
        self.captions.append({"start": round(t0, 2), "end": round(self.data.time, 2), "text": step["say"]})

    def props(self):
        """Boxes for the props this position uses (pillow, chair), for the web player."""
        out = []
        for name in ("pillow", "chair"):
            b = self.model.body(name)
            if self.data.xpos[b.id][2] < -1:
                continue
            for g in range(b.geomadr[0], b.geomadr[0] + b.geomnum[0]):
                out.append({"pos": self.data.geom_xpos[g].round(4).tolist(),
                            "size": self.model.geom_size[g].round(4).tolist(),
                            "color": self.model.geom_rgba[g][:3].round(3).tolist()})
        return out


def play(script, out_dir, progress=print):
    """Simulate a checked script and write motion.bin + meta.json for the web player."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    player = Player(script["start"])
    player.run_for(0.5, lambda x: ({}, {}))
    for i, step in enumerate(script["steps"], 1):
        progress(f"  step {i}/{len(script['steps'])}: {step['say'][:70]}")
        player.play_step(step)
    frames = np.asarray(player.frames, dtype=np.float32)            # (frames, bodies, 7)
    handles = np.asarray(player.active, dtype=np.float32)           # (frames, handles, 4)
    (out / "motion.bin").write_bytes(frames.tobytes() + handles.tobytes())
    meta = {"title": script.get("title", ""), "start": script["start"], "fps": FPS, "frames": len(frames),
            "bodies": player.bodies, "handles": player.handles, "captions": player.captions,
            "props": player.props(), "seconds": round(player.data.time, 2)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "script.json").write_text(json.dumps(script, indent=2))
    return meta


def write_index(lessons_dir):
    """List the played lessons for the web player's menu, newest first."""
    lessons_dir = Path(lessons_dir)
    entries = []
    for meta_path in lessons_dir.glob("*/meta.json"):
        if meta_path.parent.name.startswith("test-"):
            continue  # tests stay out of the public menu
        meta = json.loads(meta_path.read_text())
        entries.append({"slug": meta_path.parent.name, "title": meta.get("title") or meta_path.parent.name,
                        "mtime": meta_path.stat().st_mtime})
    entries.sort(key=lambda e: -e["mtime"])
    (lessons_dir / "index.json").write_text(json.dumps(
        [{"slug": e["slug"], "title": e["title"]} for e in entries], indent=1))
