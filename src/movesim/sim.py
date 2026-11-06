"""Load the skeleton, put it in a starting position, and let it settle under gravity."""

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "model" / "skeleton.xml"
KEY_PATH = ROOT / "model" / "keyframes.xml"

# Segments whose rotation we follow, from the pelvis up to the skull.
SPINE = (["pelvis"] + [f"L{i}" for i in range(5, 0, -1)] + [f"T{i}" for i in range(12, 0, -1)]
         + [f"C{i}" for i in range(7, 0, -1)] + ["skull"])

SOLE = ("heel", "met1", "met5", "midfoot", "bigtoe", "smalltoes")


def load(position="hook_lying"):
    """The model with this position's tone and its tissue (ligaments) switched on."""
    from movesim import ligaments
    from movesim.positions import apply_tone

    mujoco.set_mjcb_passive(None)  # a hook left from an earlier model must not run while compiling
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    apply_tone(model, position)
    ligaments.install(model)
    return model, mujoco.MjData(model)


def hands_to_knees(model, data):
    """Put every guide hand exactly on its point of the body, so the springs pull with no force."""
    for b in range(model.nbody):
        name = model.body(b).name
        if name.startswith("guide_"):
            data.mocap_pos[model.body_mocapid[b]] = data.site(name[len("guide_"):]).xpos


def lowest_point(model, data, geoms):
    """Lowest point (world z) of these sphere/ellipsoid geoms."""
    lowest = np.inf
    for g in geoms:
        size = model.geom_size[g]
        xmat = data.geom_xmat[g].reshape(3, 3)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = data.geom_xpos[g][2] - size[0]
        else:  # ellipsoid: support point along -z
            local = xmat.T @ np.array([0, 0, -1.0])
            p = size**2 * local / np.sqrt(np.sum((size * local) ** 2))
            z = (data.geom_xpos[g] + xmat @ p)[2]
        lowest = min(lowest, z)
    return lowest


def flesh(model, *prefixes):
    return [g for g in range(model.ngeom) if model.geom_group[g] == 1
            and (not prefixes or model.geom(g).name.startswith(prefixes))]


def set_start_pose(model, data, position):
    """Put the body in the position's pose, just touching the floor (and props) under it."""
    from movesim.positions import POSITIONS

    pos = POSITIONS[position]
    mujoco.mj_resetData(model, data)
    hinges = model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
    adr = model.jnt_qposadr[hinges]
    data.qpos[adr] = model.qpos_spring[adr]
    data.qpos[3:7] = pos.pelvis
    data.qpos[:3] = [0, 0, 1.0]
    mujoco.mj_forward(model, data)
    if pos.on == "chair":
        data.qpos[2] -= lowest_point(model, data, flesh(model, *SOLE)) - 0.002
        mujoco.mj_forward(model, data)
        seat = lowest_point(model, data, flesh(model, "glute", "thigh")) - 0.003
        hips = 0.5 * (data.body("thigh_L").xpos + data.body("thigh_R").xpos)
        # Sitting on the front half: the sitting bones (under the glutes, ~5 cm behind the hip
        # joints) rest about halfway between the seat's centre and its front edge.
        data.mocap_pos[model.body("chair").mocapid[0]] = [hips[0] - 0.15, hips[1], seat]
    else:
        data.qpos[2] -= lowest_point(model, data, flesh(model)) - 0.002
        mujoco.mj_forward(model, data)
    if pos.pillow:
        top = lowest_point(model, data, flesh(model, "head_side")) - 0.002
        g = model.body("pillow").geomadr[0]
        model.geom_size[g][2] = top / 2  # a pillow just thick enough for the head
        head = data.geom("occiput").xpos
        data.mocap_pos[model.body("pillow").mocapid[0]] = [head[0], head[1], top / 2]
    mujoco.mj_forward(model, data)
    hands_to_knees(model, data)


def settle(model, data, max_time=15.0, tol=1e-3):
    """Let the body come to rest, then let it ease out sideways shear.

    Dropping a body onto sticky contacts locks in shear that a real body would let go of by
    shifting a little. So everything gets a short spell of low friction: the legs then find
    the pose where their tone carries them without pushing the feet along the floor. (Props
    keep their grip: MuJoCo uses the larger friction of the two surfaces.)
    """
    still(model, data, data.time + max_time / 3, tol)
    friction = model.geom_friction.copy()
    props = [model.body("pillow").geomadr[0]] + list(range(model.body("chair").geomadr[0],
                                                            model.body("chair").geomadr[0]
                                                            + model.body("chair").geomnum[0]))
    low = [g for g in range(model.ngeom) if g not in props]
    model.geom_friction[low, 0] *= 0.03
    still(model, data, data.time + 3.0, tol=0)  # a fixed spell: slow sliding is what we want
    model.geom_friction[:] = friction
    return still(model, data, data.time + max_time / 3, tol)


def still(model, data, max_time, tol):
    """Step with the hands following their points until the body is still."""
    still_since = None
    while data.time < max_time:
        hands_to_knees(model, data)
        mujoco.mj_step(model, data)
        speed = np.abs(data.qvel).max()
        if speed < tol:
            still_since = still_since if still_since is not None else data.time
            if data.time - still_since > 0.5:
                break
        else:
            still_since = None
    speed = np.abs(data.qvel).max()
    data.qvel[:] = 0
    hands_to_knees(model, data)
    mujoco.mj_forward(model, data)
    return data.time, speed


def write_keyframes(states, path=KEY_PATH):
    """states: {position: (qpos, mocap_pos, mocap_quat, pillow_half_height)}"""
    fmt = lambda a: " ".join(f"{v:.6g}" for v in np.ravel(a))
    keys = "".join(
        f'    <key name="{name}"\n         qpos="{fmt(q)}"\n         mpos="{fmt(mp)}"\n'
        f'         mquat="{fmt(mq)}"/>\n' for name, (q, mp, mq, _) in states.items())
    pillows = "".join(f"<!-- pillow {name} {h:.4f} -->\n" for name, (_, _, _, h) in states.items())
    path.write_text("<!-- Written by `uv run movesim build`: each starting position, settled under gravity. -->\n"
                    + pillows + "<mujoco>\n  <keyframe>\n" + keys + "  </keyframe>\n</mujoco>\n")


def pillow_height(position):
    import re

    m = re.search(rf"<!-- pillow {position} ([\d.]+) -->", KEY_PATH.read_text())
    return float(m.group(1)) if m else 0.05


def load_settled(position="hook_lying"):
    """The model in its stored, settled starting position, ready to step."""
    model, data = load(position)
    model.geom_size[model.body("pillow").geomadr[0]][2] = pillow_height(position)
    mujoco.mj_resetDataKeyframe(model, data, model.key(position).id)
    mujoco.mj_forward(model, data)
    return model, data


def main(positions=None):
    from movesim.positions import POSITIONS

    states = {}
    for name in positions or POSITIONS:
        model, data = load(name)
        set_start_pose(model, data, name)
        t, speed = settle(model, data)
        h = model.geom_size[model.body("pillow").geomadr[0]][2]
        states[name] = (data.qpos.copy(), data.mocap_pos.copy(), data.mocap_quat.copy(), h)
        print(f"  {name}: settled after {t:.1f} s (max joint speed {speed:.1e})")
    write_keyframes(states)
    print(f"Wrote {KEY_PATH.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
