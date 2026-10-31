"""Load the skeleton, put it in hook lying, and let it settle under gravity."""

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "model" / "skeleton.xml"
KEY_PATH = ROOT / "model" / "hook_lying_key.xml"

# Segments whose rotation we follow, from the pelvis up to the skull.
SPINE = (["pelvis"] + [f"L{i}" for i in range(5, 0, -1)] + [f"T{i}" for i in range(12, 0, -1)]
         + [f"C{i}" for i in range(7, 0, -1)] + ["skull"])


def load():
    """The model, with its ligaments switched on."""
    from movesim import ligaments

    mujoco.set_mjcb_passive(None)  # a hook left from an earlier model must not run while compiling
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    ligaments.install(model)
    return model, mujoco.MjData(model)


def hands_to_knees(model, data):
    """Put every guide hand exactly on its point of the body, so the springs pull with no force."""
    for b in range(model.nbody):
        name = model.body(b).name
        if name.startswith("guide_"):
            data.mocap_pos[model.body_mocapid[b]] = data.site(name[len("guide_"):]).xpos


def set_hook_pose(model, data):
    """Start every joint at its resting-tone angle: hook lying, arms by the sides."""
    mujoco.mj_resetData(model, data)
    hinges = model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
    adr = model.jnt_qposadr[hinges]
    data.qpos[adr] = model.qpos_spring[adr]
    mujoco.mj_forward(model, data)
    # Lower the body so the lowest contact surface just touches the floor.
    data.qpos[2] -= lowest_flesh_point(model, data) - 0.002
    mujoco.mj_forward(model, data)
    hands_to_knees(model, data)


def lowest_flesh_point(model, data):
    lowest = np.inf
    for g in range(model.ngeom):
        if model.geom_group[g] != 1:
            continue
        # Bottom of the geom's bounding sphere is too crude for ellipsoids;
        # sample the surface along world -z instead.
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


def settle(model, data, max_time=15.0, tol=1e-3):
    """Let the body come to rest, then let it ease out sideways shear.

    Dropping a body onto sticky contacts locks in shear that a real body would let go of by
    shifting a little. So everything gets a short spell of low friction: the legs then find
    the pose where their tone carries them without pushing the feet along the floor.
    """
    still(model, data, data.time + max_time / 3, tol)
    friction = model.geom_friction.copy()
    model.geom_friction[:, 0] *= 0.03
    still(model, data, data.time + 3.0, tol=0)  # a fixed spell: slow sliding is what we want
    model.geom_friction[:] = friction
    return still(model, data, data.time + max_time / 3, tol)


def still(model, data, max_time, tol):
    """Step with the hands following the knees until the body is still."""
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


def write_keyframe(model, data, path=KEY_PATH):
    fmt = lambda a: " ".join(f"{v:.6g}" for v in np.ravel(a))
    path.write_text(
        "<!-- Written by `uv run python -m movesim.sim`: hook lying, settled under gravity. -->\n"
        "<mujoco>\n  <keyframe>\n"
        f'    <key name="hook_lying"\n'
        f'         qpos="{fmt(data.qpos)}"\n'
        f'         mpos="{fmt(data.mocap_pos)}"\n'
        f'         mquat="{fmt(data.mocap_quat)}"/>\n'
        "  </keyframe>\n</mujoco>\n"
    )


def load_settled():
    """The model in its stored hook-lying keyframe, ready to step."""
    model, data = load()
    mujoco.mj_resetDataKeyframe(model, data, model.key("hook_lying").id)
    mujoco.mj_forward(model, data)
    return model, data


def main():
    model, data = load()
    set_hook_pose(model, data)
    t, speed = settle(model, data)
    write_keyframe(model, data)
    print(f"Settled after {t:.2f} s (max joint speed {speed:.1e}). Wrote {KEY_PATH.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
