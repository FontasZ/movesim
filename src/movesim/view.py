"""Open the skeleton in MuJoCo's viewer, lying in hook lying and ready to be moved.

Run it with plain python, also on macOS:  uv run python -m movesim.view
(mjpython is only for the passive viewer; with launch() it fails on macOS.)
"""

import mujoco
import mujoco.viewer

from movesim.sim import load_settled

# How quickly a hand that is let go drifts back onto its knee, so it stops pulling (seconds).
LET_GO_TIME = 0.3


def hands_let_go(model, data):
    """Each step, ease every guide hand toward its point on the body.

    While you drag a hand, the viewer puts it back under the mouse before every step, so the
    drift does not show. Once you let go, the hand settles onto its point, the spring goes
    slack, and the body sinks back under its own weight.
    """
    alpha = model.opt.timestep / LET_GO_TIME
    for b in range(model.nbody):
        name = model.body(b).name
        if name.startswith("guide_"):
            mocap = model.body_mocapid[b]
            point = data.site(name[len("guide_"):]).xpos
            data.mocap_pos[mocap] += alpha * (point - data.mocap_pos[mocap])


def main():
    model, data = load_settled()
    mujoco.set_mjcb_control(hands_let_go)
    try:
        mujoco.viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
