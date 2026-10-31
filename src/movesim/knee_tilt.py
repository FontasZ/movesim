"""Knees to the right: guide both knees about 20 degrees right and back, everything else passive.

Writes out/knee_tilt/{data.csv, report.md, strip_above.png, strip_feet.png, knee_tilt.mp4}.
"""

import csv
from pathlib import Path

import mujoco
import numpy as np

from movesim.sim import ROOT, SPINE, hands_to_knees, load_settled

OUT = ROOT / "out" / "knee_tilt"

TILT_DEG = 20.0
# Phases (s): rest, knees go right, stay, knees come back, rest.
T_REST, T_GO, T_HOLD, T_BACK, T_END = 1.0, 4.0, 5.0, 8.0, 9.0

FOOT_PARTS = ("heel", "met1", "met5", "midfoot", "bigtoe", "smalltoes")
FOOT_POINTS = [f"{p}_{s}" for s in "LR" for p in FOOT_PARTS]
REGIONS = ["pelvis", "lumbar back", "lower ribs (T8-T12)", "upper back (T1-T7)", "shoulder blades",
           "head and neck", "arms", "feet"]


def region(geom):
    """Which part of the body a contact surface belongs to."""
    base = geom.rsplit("_", 1)[0] if geom.endswith(("_L", "_R")) else geom
    if base in ("sacrum", "glute"):
        return "pelvis"
    if base.startswith("L"):
        return "lumbar back"
    if base.startswith("T"):
        return "lower ribs (T8-T12)" if int(base[1:].split("_")[0]) >= 8 else "upper back (T1-T7)"
    if base == "scapula":
        return "shoulder blades"
    if base.startswith("C") or base == "occiput":
        return "head and neck"
    if base in ("arm", "elbow", "forearm", "hand"):
        return "arms"
    return "feet"


def side(geom):
    return {"L": "left", "R": "right"}.get(geom[-1], "centre") if geom[-2] == "_" else "centre"


def tilt_fraction(t):
    """0 at rest, 1 at full tilt, with smooth (cosine) starts and stops."""
    ease = lambda x: 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))
    if t < T_GO:
        return ease((t - T_REST) / (T_GO - T_REST))
    if t < T_HOLD:
        return 1.0
    return 1.0 - ease((t - T_HOLD) / (T_BACK - T_HOLD))


def rot_x(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


class Guide:
    """Moves the two hands along the arc the knees take when they tip over the feet.

    The arc turns about the head-to-foot line through the middle of the ankles.
    The person's right is world +y, so tipping right turns about world x by a negative angle.
    """

    def __init__(self, model, data):
        self.mocap = [model.body(f"guide_knee_{s}").mocapid[0] for s in "LR"]
        self.start = data.mocap_pos[self.mocap].copy()
        self.pivot = 0.5 * (data.body("foot_L").xpos + data.body("foot_R").xpos)

    def move(self, data, fraction):
        r = rot_x(-TILT_DEG * fraction)
        data.mocap_pos[self.mocap] = (self.start - self.pivot) @ r.T + self.pivot


def roll_right_deg(rel_quat):
    """Rotation about the head-to-foot axis, positive when the front turns to the right."""
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, rel_quat, 1.0)
    return -np.degrees(vel[0])


def relative_quat(q_now, q_ref):
    """World-frame rotation taking q_ref to q_now."""
    inv, out = np.zeros(4), np.zeros(4)
    mujoco.mju_negQuat(inv, q_ref)
    mujoco.mju_mulQuat(out, q_now, inv)
    return out


def accumulate_slip(model, data, foot_geoms, slip):
    """Add up how far each touching foot point slides along the floor (rolling does not count)."""
    jacp = np.zeros((3, model.nv))
    for i in range(data.ncon):
        c = data.contact[i]
        g = c.geom1 if c.geom1 in foot_geoms else c.geom2 if c.geom2 in foot_geoms else None
        if g is None:
            continue
        mujoco.mj_jac(model, data, jacp, None, c.pos, model.geom_bodyid[g])
        v = jacp @ data.qvel
        slip[foot_geoms[g]] += np.linalg.norm(v[:2]) * model.opt.timestep


def contact_forces(model, data):
    """Normal force (N) on each floor-touching geom, and the force-weighted centre of pressure."""
    forces = {}
    wrench = np.zeros(6)
    cop, total = np.zeros(3), 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        g = c.geom1 if model.geom_group[c.geom1] == 1 else c.geom2
        mujoco.mj_contactForce(model, data, i, wrench)
        name = model.geom(g).name
        forces[name] = forces.get(name, 0.0) + wrench[0]
        if not name.startswith(FOOT_PARTS):
            cop += wrench[0] * c.pos
            total += wrench[0]
    return forces, (cop / total if total > 0 else cop)


def run(render=True):
    model, data = load_settled()
    guide = Guide(model, data)
    flesh = [model.geom(g).name for g in range(model.ngeom) if model.geom_group[g] == 1]
    back = [g for g in flesh if region(g) not in ("arms", "feet", "head and neck")]
    rest_height = {g: data.geom(g).xpos[2] for g in back}
    ref_quat = {b: data.body(b).xquat.copy() for b in SPINE}
    foot_geoms = {model.geom(p).id: p for p in FOOT_POINTS}
    slip = dict.fromkeys(FOOT_POINTS, 0.0)

    renderer = Renderer(model) if render else None
    rows, frames = [], {"above": [], "feet": []}
    sample_every, frame_every = 20, 33  # 50 Hz data, 30 fps video
    step = 0
    while data.time < T_END:
        hands_to_knees(model, data)  # every other handle follows its point and pulls with no force
        guide.move(data, tilt_fraction(data.time))
        mujoco.mj_step(model, data)
        accumulate_slip(model, data, foot_geoms, slip)
        step += 1
        if step % sample_every == 0:
            forces, cop = contact_forces(model, data)
            row = {"t": round(data.time, 3), "knee_tilt_target_deg": round(TILT_DEG * tilt_fraction(data.time), 2)}
            knees = 0.5 * (data.site("knee_L").xpos + data.site("knee_R").xpos)
            pivot = guide.pivot
            row["knee_tilt_actual_deg"] = round(np.degrees(np.arctan2(knees[1] - pivot[1], knees[2] - pivot[2])), 2)
            for b in SPINE:
                row[f"roll_{b}"] = round(roll_right_deg(relative_quat(data.body(b).xquat, ref_quat[b])), 3)
            for g in flesh:
                row[f"force_{g}"] = round(forces.get(g, 0.0), 2)
            for g in back:
                row[f"lift_{g}_mm"] = round(1000 * (data.geom(g).xpos[2] - rest_height[g]), 2)
            row["cop_back_y_mm"] = round(1000 * cop[1], 2)
            for p in FOOT_POINTS:
                row[f"slip_{p}_mm"] = round(1000 * slip[p], 2)
            row["guide_force_N"] = round(sum(data.ten_length[model.tendon(f"guide_knee_{s}").id] * model.tendon_stiffness[model.tendon(f"guide_knee_{s}").id] for s in "LR"), 2)
            rows.append(row)
        if renderer and step % frame_every == 0:
            for view in frames:
                frames[view].append(renderer.render(data, view))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "report.md").write_text(report(rows, flesh))
    if renderer:
        save_media(frames)
    return rows


# ---------------------------------------------------------------- rendering

class Renderer:
    def __init__(self, model, width=640, height=480):
        self.model = model
        self.r = mujoco.Renderer(model, height, width)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]  # bones and the guide; hide flesh and mass
        self.opt.sitegroup[:] = 1
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
        self.cams = {}
        for view, (az, el, dist) in {"above": (90, -89.9, 1.7), "feet": (0, -12, 1.35)}.items():
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.azimuth, cam.elevation, cam.distance = az, el, dist
            self.cams[view] = cam

    def render(self, data, view):
        cam = self.cams[view]
        if view == "above":
            cam.lookat[:] = [-0.05, 0.0, 0.15]
        else:  # from beyond the feet, looking up the body: the person's right is on the image's left
            cam.lookat[:] = [-0.15, 0.0, 0.22]
        self.r.update_scene(data, cam, self.opt)
        return self.r.render().copy()


def save_media(frames):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    fps = 30
    both = [np.concatenate([a, b], axis=1) for a, b in zip(frames["above"], frames["feet"])]
    imageio.mimsave(OUT / "knee_tilt.mp4", both, fps=fps, macro_block_size=8)
    # Strips: rest, halfway, full tilt, halfway back, rest again.
    times = [T_REST, (T_REST + T_GO) / 2, (T_GO + T_HOLD) / 2, (T_HOLD + T_BACK) / 2, T_END - 0.05]
    for view, imgs in frames.items():
        picks = [imgs[min(int(t * fps), len(imgs) - 1)] for t in times]
        strip = Image.fromarray(np.concatenate(picks, axis=1))
        draw = ImageDraw.Draw(strip)
        for i, t in enumerate(times):
            draw.text((i * picks[0].shape[1] + 10, 10), f"t={t:.1f}s  tilt {TILT_DEG * tilt_fraction(t):.0f}°",
                      fill=(0, 0, 0))
        strip.save(OUT / f"strip_{view}.png")


# ---------------------------------------------------------------- report

def report(rows, flesh):
    peak = max(rows, key=lambda r: abs(r["knee_tilt_actual_deg"]))
    rest, end = rows[0], rows[-1]
    lines = [
        "# Knees to the right: what the skeleton did",
        "",
        f"Lying on the back, knees bent, feet flat, arms by the sides. A guide's hands moved both knees "
        f"{TILT_DEG:.0f}° to the right over {T_GO - T_REST:.0f} s, held for {T_HOLD - T_GO:.0f} s, and brought "
        f"them back over {T_BACK - T_HOLD:.0f} s. Everything else was passive. "
        f"At the peak (t = {peak['t']:.1f} s) the knees had tipped {peak['knee_tilt_actual_deg']:.1f}° "
        f"(the hands asked for {peak['knee_tilt_target_deg']:.1f}°).",
        "",
        "## How far each part rolled to the right at the peak",
        "",
        "| part | roll right (°) | share of pelvis roll |",
        "|---|---:|---:|",
    ]
    pelvis = peak["roll_pelvis"]
    for b in SPINE:
        share = peak[f"roll_{b}"] / pelvis if abs(pelvis) > 1e-6 else 0
        lines.append(f"| {b} | {peak[f'roll_{b}']:.2f} | {100 * share:.0f}% |")

    lines += ["", "## Floor contact (newtons) at rest and at the peak", "",
              "| region | side | at rest | at peak | change |", "|---|---|---:|---:|---:|"]
    totals = {}
    for g in flesh:
        key = (region(g), side(g))
        a, b = totals.get(key, (0.0, 0.0))
        totals[key] = (a + rest[f"force_{g}"], b + peak[f"force_{g}"])
    for reg in REGIONS:
        for sd in ("left", "centre", "right"):
            if (reg, sd) in totals and max(totals[reg, sd]) >= 0.5:
                a, b = totals[reg, sd]
                lines.append(f"| {reg} | {sd} | {a:.0f} | {b:.0f} | {b - a:+.0f} |")

    lifted = [g for g in flesh if rest[f"force_{g}"] > 1 and peak[f"force_{g}"] < 0.5]
    landed = [g for g in flesh if rest[f"force_{g}"] < 0.5 and peak[f"force_{g}"] > 1]
    rises = sorted(((peak[k], k[5:-3]) for k in peak if k.startswith("lift_")), reverse=True)
    max_slide = max(max(r[f"slip_{p}_mm"] for r in rows) for p in FOOT_POINTS)
    cop_shift = peak["cop_back_y_mm"] - rest["cop_back_y_mm"]
    left = sum(peak[f"force_{g}"] - rest[f"force_{g}"] for g in flesh
               if side(g) == "left" and region(g) not in ("arms", "feet"))
    right = sum(peak[f"force_{g}"] - rest[f"force_{g}"] for g in flesh
                if side(g) == "right" and region(g) not in ("arms", "feet"))
    skull = max(abs(r["roll_skull"]) for r in rows)
    first_small = next((b for b in SPINE[1:] if abs(peak[f"roll_{b}"]) < 0.1 * abs(pelvis)), None)
    half = next((b for b in SPINE[1:] if abs(peak[f"roll_{b}"]) < 0.5 * abs(pelvis)), None)

    lines += ["", "## In plain words", ""]
    lines.append(f"- The pelvis rolled {pelvis:.1f}° to the right"
                 + (" (to the right, as expected)." if pelvis > 0 else " (the wrong way)."))
    if first_small:
        lines.append(f"- The twist faded going up the spine: under half the pelvis roll by {half}, "
                     f"under a tenth by {first_small}.")
    else:
        lines.append("- The twist did not fade below a tenth of the pelvis roll anywhere up the spine.")
    lines.append(f"- T6, in the middle of the chest, rolled {peak['roll_T6']:.2f}°; the skull at most {skull:.2f}°.")
    lines.append(f"- Back pressure moved {abs(cop_shift):.0f} mm to the {'right' if cop_shift > 0 else 'left'}: "
                 f"the left side of the back unloaded by {-left:.0f} N and the right side took {right:+.0f} N.")
    up = [f"{g} {mm:.0f} mm" for mm, g in rises if mm >= 1 and g.endswith("_L")][:6]
    if up:
        lines.append(f"- The left side of the back lifted: {', '.join(up)}.")
    if lifted:
        lines.append(f"- Off the floor at the peak: {', '.join(lifted)}.")
    if landed:
        lines.append(f"- Newly touching the floor at the peak: {', '.join(landed)}.")
    lines.append(f"- The feet stayed planted: no point of contact slid more than {max_slide:.1f} mm in total."
                 if max_slide < 5 else f"- The feet slid: up to {max_slide:.1f} mm in total.")
    lines.append(f"- After returning, the pelvis was {end['roll_pelvis']:+.2f}° from where it started.")
    lines.append("")
    return "\n".join(lines)


def main():
    rows = run()
    print((OUT / "report.md").read_text())


if __name__ == "__main__":
    main()
