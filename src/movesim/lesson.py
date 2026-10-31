"""Movement scripts: what a lesson asks the body to do, played through the physics.

A script is a list of steps. Each step has the teacher's words ("say") and the movements that
go with them, in a small vocabulary (MOVES). A movement takes hold of one or more points of
the body (handles: knees, forehead, front hip bones, breastbone, palms, shoulders, ankles) and
guides them along a path on a soft spring, the way an intention leads a part. Points can also
be held still (HOLDS). Everything else is passive: physics decides where the rest goes.

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
from movesim.sim import hands_to_knees, load_settled

FPS = 30
RIGHT = np.array([0, 1.0, 0])  # the person's right is world +y (head toward +x, face up)
TOWARD_HEAD = np.array([1.0, 0, 0])
UP = np.array([0, 0, 1.0])


@dataclass
class Move:
    help: str      # what it means, for people and for the translator
    unit: str      # "deg" or "cm"
    default: float
    sides: tuple   # which values of "side" make sense


MOVES = {
    "knees_tilt": Move("Both knees tip together to one side over the feet (side = direction).",
                       "deg", 20, ("left", "right")),
    "knee_open": Move("One knee (or both) falls outward, away from the midline.", "deg", 20,
                      ("left", "right", "both")),
    "knee_close": Move("One knee (or both) moves inward, across toward the other side.", "deg", 15,
                       ("left", "right", "both")),
    "foot_press": Move("The foot presses into the floor: that side of the pelvis lifts and is pushed a "
                       "little toward the head (amount = how far the pelvis lifts).", "cm", 3,
                       ("left", "right", "both")),
    "knee_to_chest": Move("The knee comes toward the chest; the foot leaves the floor.", "deg", 30,
                          ("left", "right", "both")),
    "head_roll": Move("The head rolls to one side, nose turning toward that side (side = direction).",
                      "deg", 30, ("left", "right")),
    "head_lift": Move("The head lifts from the floor, chin toward the chest.", "deg", 20, ("none",)),
    "pelvis_flatten": Move("The pelvis tilts so the lower back comes toward the floor "
                           "(pubic bone lifts slightly).", "deg", 8, ("none",)),
    "pelvis_arch": Move("The pelvis tilts so the lower back arches away from the floor "
                        "(tailbone presses down).", "deg", 8, ("none",)),
    "pelvis_lift": Move("The pelvis lifts from the floor, as in a small bridge.", "cm", 5, ("none",)),
    "arm_lift": Move("The arm lifts from the floor toward the ceiling, straight, from the shoulder "
                     "(90 = hand above the shoulder).", "deg", 90, ("left", "right", "both")),
    "shoulder_lift": Move("The shoulder lifts from the floor toward the ceiling, the arm following.",
                          "cm", 4, ("left", "right", "both")),
}

HOLDS = {  # what can be asked to stay still, and the points that hold it
    "head": ["forehead"], "pelvis": ["asis_L", "asis_R"], "chest": ["sternum"],
    "knees": ["knee_L", "knee_R"], "feet": ["ankle_L", "ankle_R"],
    "left_arm": ["palm_L"], "right_arm": ["palm_R"], "shoulders": ["shoulder_L", "shoulder_R"],
}

STEP_DEFAULTS = {"moves": [], "hold": [], "seconds": 3.0, "stay": 1.0, "return": True, "repeat": 1,
                 "rest_after": 1.0, "release": False}


# ------------------------------------------------------------------ the script

def load_script(path):
    return check(json.loads(Path(path).read_text()))


def check(script):
    """Fill defaults and reject what the body can't do; returns the script."""
    if script.get("start", "hook_lying") != "hook_lying":
        raise ValueError(f"unknown starting position {script['start']!r}; only hook_lying for now")
    for i, step in enumerate(script["steps"], 1):
        for key, value in STEP_DEFAULTS.items():
            step.setdefault(key, value)
        for m in step["moves"]:
            spec = MOVES.get(m["move"])
            if spec is None:
                raise ValueError(f"step {i}: unknown move {m['move']!r}")
            m.setdefault("side", spec.sides[0])
            if m["side"] not in spec.sides:
                raise ValueError(f"step {i}: {m['move']} takes side {spec.sides}, not {m['side']!r}")
            if not m.get("amount"):
                m["amount"] = spec.default
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


def rolling(p, center, radius, deg, toward):
    """A path for a point on a ball (the head) rolling on the floor about the head-to-foot axis."""
    def at(sign, f):
        t = np.radians(sign * deg * f)
        shift = np.cross(TOWARD_HEAD * t, radius * UP)  # rolling without slipping moves the centre
        return rotate(p, TOWARD_HEAD, center, sign * deg * f) + shift
    sign = 1 if (at(1, 1) - p) @ toward >= 0 else -1
    return lambda f: at(sign, f)


def line(p, direction, cm):
    d = direction / np.linalg.norm(direction)
    return lambda f: p + d * cm / 100 * f


def paths(model, data, move):
    """For one movement: {handle site: path(f)} with f from 0 (start) to 1 (there)."""
    name, side, amount = move["move"], move["side"], move["amount"]
    sides = {"left": ["L"], "right": ["R"], "both": ["L", "R"]}.get(side, [])
    site = lambda s: data.site(s).xpos.copy()
    body = lambda b: data.body(b).xpos.copy()
    out = {}
    if name == "knees_tilt":
        pivot = 0.5 * (body("foot_L") + body("foot_R"))
        toward = RIGHT if side == "right" else -RIGHT
        for s in "LR":
            out[f"knee_{s}"] = arc(site(f"knee_{s}"), TOWARD_HEAD, pivot, amount, toward)
    elif name in ("knee_open", "knee_close"):
        for s in sides:
            outward = RIGHT if s == "R" else -RIGHT
            toward = outward if name == "knee_open" else -outward
            out[f"knee_{s}"] = arc(site(f"knee_{s}"), TOWARD_HEAD, body(f"foot_{s}"), amount, toward)
    elif name == "foot_press":
        # Pressing a planted foot pushes up through the thigh: that side of the pelvis lifts and
        # moves a little toward the head. The knee and foot follow on their own.
        for s in sides:
            out[f"asis_{s}"] = line(site(f"asis_{s}"), UP + 0.4 * TOWARD_HEAD, amount * np.hypot(1, 0.4))
    elif name == "knee_to_chest":
        for s in sides:
            out[f"knee_{s}"] = arc(site(f"knee_{s}"), RIGHT, body(f"thigh_{s}"), amount, TOWARD_HEAD)
    elif name == "head_roll":
        toward = RIGHT if side == "right" else -RIGHT
        head = data.geom("occiput")
        out["forehead"] = rolling(site("forehead"), head.xpos.copy(), model.geom("occiput").size[0], amount, toward)
    elif name == "head_lift":
        out["forehead"] = arc(site("forehead"), RIGHT, body("C4"), amount, UP)
    elif name in ("pelvis_flatten", "pelvis_arch"):
        pivot = 0.5 * (body("thigh_L") + body("thigh_R"))
        toward = TOWARD_HEAD if name == "pelvis_flatten" else -TOWARD_HEAD
        for s in "LR":
            out[f"asis_{s}"] = arc(site(f"asis_{s}"), RIGHT, pivot, amount, toward)
    elif name == "pelvis_lift":
        for s in "LR":
            out[f"asis_{s}"] = line(site(f"asis_{s}"), UP, amount)
    elif name == "arm_lift":
        for s in sides:
            out[f"palm_{s}"] = arc(site(f"palm_{s}"), RIGHT, body(f"humerus_{s}"), amount, UP)
    elif name == "shoulder_lift":
        for s in sides:
            out[f"shoulder_{s}"] = line(site(f"shoulder_{s}"), UP, amount)
    return out


# ------------------------------------------------------------------ playing a script

def ease(x):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))


class Player:
    """Steps the physics through a script and records what the body does."""

    def __init__(self):
        self.model, self.data = load_settled()
        m = self.model
        self.handles = [m.body(b).name[len("guide_"):] for b in range(m.nbody)
                        if m.body(b).name.startswith("guide_")]
        self.mocap = {h: m.body(f"guide_{h}").mocapid[0] for h in self.handles}
        self.bodies = [b for b in MESH_BODIES]
        self.body_ids = [m.body(b).id for b in self.bodies]
        self.kept = {}  # handle -> position it keeps after a step that did not return
        self.frames, self.active, self.captions = [], [], []
        self.next_frame = 0.0

    def step_physics(self, targets):
        """One physics step: targeted handles go to their targets, the rest follow their points."""
        hands_to_knees(self.model, self.data)
        for h, p in {**self.kept, **targets}.items():
            self.data.mocap_pos[self.mocap[h]] = p
        mujoco.mj_step(self.model, self.data)
        if self.data.time >= self.next_frame:
            self.next_frame += 1 / FPS
            d = self.data
            self.frames.append(np.concatenate([d.xpos[self.body_ids], d.xquat[self.body_ids]], axis=1))
            held = {**self.kept, **targets}
            self.active.append([[*(d.mocap_pos[self.mocap[h]]), 1.0 if h in held else 0.0]
                                for h in self.handles])

    def run_for(self, seconds, targets_at):
        end = self.data.time + seconds
        start = self.data.time
        while self.data.time < end:
            self.step_physics(targets_at((self.data.time - start) / max(seconds, 1e-9)))

    def play_step(self, step):
        if step["release"]:
            self.kept = {}
        t0 = self.data.time
        holds = {h: self.data.site(h).xpos.copy() for name in step["hold"] for h in HOLDS[name]}
        moves = {}
        for m in step["moves"]:
            moves.update(paths(self.model, self.data, m))
        for h in moves:
            self.kept.pop(h, None)
        at = lambda f: {**holds, **{h: path(f) for h, path in moves.items()}}
        for r in range(int(step["repeat"])):
            self.run_for(step["seconds"], lambda x: at(ease(x)))
            self.run_for(step["stay"], lambda x: at(1.0))
            if step["return"]:
                self.run_for(step["seconds"], lambda x: at(1 - ease(x)))
                if r < step["repeat"] - 1:
                    self.run_for(0.5, lambda x: at(0.0))
        if not step["return"]:
            self.kept.update({h: path(1.0) for h, path in moves.items()})
        self.run_for(step["rest_after"], lambda x: {})
        self.captions.append({"start": round(t0, 2), "end": round(self.data.time, 2), "text": step["say"]})


def play(script, out_dir, progress=print):
    """Simulate a checked script and write motion.bin + meta.json for the web player."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    player = Player()
    player.run_for(0.5, lambda x: {})
    for i, step in enumerate(script["steps"], 1):
        progress(f"  step {i}/{len(script['steps'])}: {step['say'][:70]}")
        player.play_step(step)
    frames = np.asarray(player.frames, dtype=np.float32)            # (frames, bodies, 7)
    handles = np.asarray(player.active, dtype=np.float32)           # (frames, handles, 4)
    (out / "motion.bin").write_bytes(frames.tobytes() + handles.tobytes())
    meta = {"title": script.get("title", ""), "fps": FPS, "frames": len(frames),
            "bodies": player.bodies, "handles": player.handles, "captions": player.captions,
            "seconds": round(player.data.time, 2)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "script.json").write_text(json.dumps(script, indent=2))
    return meta


def write_index(lessons_dir):
    """List the played lessons for the web player's menu, newest first."""
    lessons_dir = Path(lessons_dir)
    entries = []
    for meta_path in lessons_dir.glob("*/meta.json"):
        meta = json.loads(meta_path.read_text())
        entries.append({"slug": meta_path.parent.name, "title": meta.get("title") or meta_path.parent.name,
                        "mtime": meta_path.stat().st_mtime})
    entries.sort(key=lambda e: -e["mtime"])
    (lessons_dir / "index.json").write_text(json.dumps(
        [{"slug": e["slug"], "title": e["title"]} for e in entries], indent=1))
