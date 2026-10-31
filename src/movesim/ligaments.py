"""Passive tissue: resistance that grows toward the end of each joint's range.

Real joints give way easily near rest and stiffen as ligaments, capsules, discs and resting
muscle tighten. For each hinge joint the moment resisting a displacement d from rest is

    c · (exp(a · u) − 1),   u = (|d| − s·r) / ((1 − s)·r), from 0 where tightening starts,

with r the range on that side of rest and c, a, s the joint's user values (c is given for each
side). The spine starts tightening at rest (s = 0) and follows in-vivo measurements; the limbs
keep their gentle joint spring up to halfway (s = 0.5) and tighten from there. The joint limit
stays as a backstop just beyond.

MuJoCo has no built-in nonlinear joint spring, so this runs as a passive-force callback:
everything that loads the model through movesim.sim gets it. The bare XML in another viewer
does not.
"""

import mujoco
import numpy as np

OVERSHOOT = 1.3  # cap the growth a little past the end of range; the joint limit takes over


class Ligaments:
    def __init__(self, model):
        joints = np.flatnonzero((model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
                                & (model.jnt_limited == 1) & (model.jnt_user[:, 0] > 0))
        self.names = [model.joint(j).name for j in joints]
        self.qadr = model.jnt_qposadr[joints]
        self.dadr = model.jnt_dofadr[joints]
        self.c_lo, self.c_hi, self.a, self.s = model.jnt_user[joints, :4].T.copy()
        self.rest = model.qpos_spring[self.qadr]
        lo, hi = model.jnt_range[joints].T
        self.r_lo = np.maximum(self.rest - lo, 1e-6)
        self.r_hi = np.maximum(hi - self.rest, 1e-6)

    def torque(self, q):
        """Passive moment on each joint at angles q (rad), restoring toward rest."""
        d = q - self.rest
        r = np.where(d > 0, self.r_hi, self.r_lo)
        c = np.where(d > 0, self.c_hi, self.c_lo)
        u = np.clip((np.abs(d) - self.s * r) / ((1 - self.s) * r), 0, OVERSHOOT)
        return -np.sign(d) * c * np.expm1(self.a * u)

    def __call__(self, model, data):
        data.qfrc_passive[self.dadr] += self.torque(data.qpos[self.qadr])


def install(model):
    """Make MuJoCo add the passive tissue moments for this model on every step."""
    ligaments = Ligaments(model)
    mujoco.set_mjcb_passive(ligaments)
    return ligaments
