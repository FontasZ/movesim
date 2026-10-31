"""Ligaments: resistance that grows steeply toward the end of each joint's range.

Real joints are loose around their resting position (the neutral zone) and stiffen gradually as
ligaments, capsules and discs tighten. The model's joint springs give the gentle, even
resistance of the neutral zone; this adds the tightening.

For each hinge joint the ligaments start to pull halfway from the resting angle to the end of
the range, and their moment grows exponentially until it reaches the joint's end moment (the
user value in the model, N·m) at the end of the range. The joint limit stays as a backstop
just beyond.

MuJoCo has no built-in nonlinear joint spring, so this runs as a passive-force callback:
everything that loads the model through movesim.sim gets it. The bare XML in another viewer
does not.
"""

import mujoco
import numpy as np

ELASTIC_START = 0.5  # how far from the resting angle to the end of range the tightening begins
SHARPNESS = 3.0      # how steeply it grows: e-foldings across that stretch
OVERSHOOT = 1.5      # cap the growth a little past the end of range; the joint limit takes over


class Ligaments:
    def __init__(self, model):
        joints = np.flatnonzero((model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
                                & (model.jnt_limited == 1) & (model.jnt_user[:, 0] > 0))
        self.names = [model.joint(j).name for j in joints]
        self.qadr = model.jnt_qposadr[joints]
        self.dadr = model.jnt_dofadr[joints]
        self.moment = model.jnt_user[joints, 0].copy()
        rest = model.qpos_spring[self.qadr]
        lo, hi = model.jnt_range[joints].T
        self.hi_start = rest + ELASTIC_START * (hi - rest)
        self.lo_start = rest + ELASTIC_START * (lo - rest)
        self.hi_scale = (hi - self.hi_start) / SHARPNESS
        self.lo_scale = (self.lo_start - lo) / SHARPNESS
        self.norm = np.expm1(SHARPNESS)

    def torque(self, q):
        """Ligament moment on each joint at angles q (rad), restoring toward rest."""
        cap = SHARPNESS * OVERSHOOT
        up = np.clip((q - self.hi_start) / self.hi_scale, 0, cap)
        down = np.clip((self.lo_start - q) / self.lo_scale, 0, cap)
        return self.moment * (np.expm1(down) - np.expm1(up)) / self.norm

    def __call__(self, model, data):
        data.qfrc_passive[self.dadr] += self.torque(data.qpos[self.qadr])


def install(model):
    """Make MuJoCo add the ligament moments for this model on every step."""
    ligaments = Ligaments(model)
    mujoco.set_mjcb_passive(ligaments)
    return ligaments
