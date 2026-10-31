"""Fit real bone shapes from BodyParts3D onto the model's segments.

    uv run python -m movesim.bones /path/to/partof_BP3D_4.0_obj_99 /path/to/partof_element_parts.txt

BodyParts3D is a whole-body model of one adult man, built from CT scans (Database Center for
Life Science, Japan; CC BY 4.0). For each of our segments this takes the matching bones,
finds joint landmarks on them (disc centres, the femoral head, the knee, the ankle, ...),
places and scales them so those landmarks land on our joints, simplifies them, and writes one
mesh per segment to model/meshes/<segment>.stl in that segment's own frame.

Run it once; the build then picks the meshes up. Only the processed meshes are kept.
"""

import struct
import sys
from pathlib import Path

import fast_simplification
import mujoco
import numpy as np

from movesim.sim import ROOT

MESH_DIR = ROOT / "model" / "meshes"

VERTEBRAE = ([f"L{i}" for i in range(5, 0, -1)] + [f"T{i}" for i in range(12, 0, -1)]
             + [f"C{i}" for i in range(7, 0, -1)])
ORDINAL = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
           "tenth", "eleventh", "twelfth"]
REGION = {"L": "lumbar", "T": "thoracic", "C": "cervical"}
FOOT = ["talus", "calcaneus", "navicular bone of {s} foot", "{s} cuboid bone", "{s} medial cuneiform bone",
        "{s} intermediate cuneiform bone", "{s} lateral cuneiform bone"] + \
       [f"{{s}} {o} metatarsal bone" for o in ORDINAL[:5]]
TOES = ["proximal phalanx of {s} big toe", "distal phalanx of {s} big toe"] + [
    f"{p} phalanx of {{s}} {t} toe" for t in ("second", "third", "fourth", "little")
    for p in ("proximal", "middle", "distal")]
HAND = [f"{{s}} {b}" for b in ("scaphoid", "lunate", "triquetral", "pisiform", "trapezium", "trapezoid",
                                "capitate", "hamate")] + \
       [f"{{s}} {o} metacarpal bone" for o in ORDINAL[:5]] + \
       ["proximal phalanx of {s} thumb", "distal phalanx of {s} thumb"] + [
    f"{p} phalanx of {{s}} {f} finger" for f in ("index", "middle", "ring", "little")
    for p in ("proximal", "middle", "distal")]

# Faces to keep per segment after simplification.
FACES = {"skull": 20000, "pelvis": 5000, "thigh": 2500, "shin": 2500, "foot": 3000, "toes": 1500,
         "clavicle": 600, "scapula": 1800, "humerus": 2000, "forearm": 2500, "hand": 3500,
         "lumbar": 1500, "thoracic": 2400, "cervical": 900}


class Atlas:
    """The BodyParts3D bones, in our standing axes (x forward, y left, z up) and metres."""

    def __init__(self, obj_dir, element_parts):
        self.dir = Path(obj_dir)
        self.elements = {}
        for line in Path(element_parts).read_text().splitlines()[1:]:
            _, name, fid = line.split("\t")
            self.elements.setdefault(name, []).append(fid)

    def has(self, name):
        return name in self.elements

    def mesh(self, *names):
        """Vertices and triangles of the named parts, merged."""
        verts, faces = [], []
        for name in names:
            for fid in self.elements[name]:
                v, f = read_obj(self.dir / f"{fid}.obj")
                faces.append(f + sum(len(x) for x in verts))
                verts.append(v)
        v = np.concatenate(verts)
        # BodyParts3D: x = the person's left, -y = forward, z = up, millimetres.
        return np.column_stack([-v[:, 1], v[:, 0], v[:, 2]]) / 1000, np.concatenate(faces)

    def points(self, *names):
        return self.mesh(*names)[0]


def read_obj(path):
    verts, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
            faces += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    return np.array(verts), np.array(faces, dtype=np.int64)


def near(points, axis, end, within):
    """Centroid of the points within `within` metres of the max (end=+1) or min (end=-1) along axis."""
    c = points @ axis
    edge = c.max() if end > 0 else c.min()
    return points[np.abs(c - edge) < within].mean(axis=0)


# ------------------------------------------------------------------ landmarks

def vertebra_name(v):
    return {"C1": "atlas", "C2": "axis"}.get(v, f"{ORDINAL[int(v[1:]) - 1]} {REGION[v[0]]} vertebra")


def landmarks(atlas):
    """Joint centres on the real skeleton (our standing axes, metres)."""
    up, left, fwd = np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])
    lm = {}
    # Disc centres: the disc below each vertebra (FMA names a disc after the vertebra above it).
    disc = {}
    for v in VERTEBRAE:
        name = ("intervertebral disk of axis" if v == "C2"
                else f"intervertebral disk of {ORDINAL[int(v[1:]) - 1]} {REGION[v[0]]} vertebra")
        if atlas.has(name):
            disc[v] = atlas.points(name).mean(axis=0)
    # A disc missing from the data (below T12): halfway between the two vertebrae, corrected by
    # how far the disc sits from that halfway point one level up.
    centroid = lambda v: atlas.points(vertebra_name(v)).mean(axis=0)
    for i, v in enumerate(VERTEBRAE[:-2]):
        if v not in disc:
            below = VERTEBRAE[i - 1] if i > 0 else None
            above = VERTEBRAE[i + 1]
            mid = lambda a, b: 0.5 * (centroid(a) + centroid(b))
            disc[v] = mid(v, below) + (disc[above] - mid(above, v))
    atlas_c = atlas.points("atlas").mean(axis=0)
    disc["C1"] = atlas_c - 0.006 * up  # atlas on axis, just below the ring of the atlas
    lm["skull"] = atlas_c + 0.008 * up  # occipital condyles on the atlas
    for v in VERTEBRAE:
        lm[v] = disc[v]
    lm["pelvis"] = disc["L5"]
    lm["vertex"] = near(atlas.points("skull"), up, +1, 0.01)
    for s, side in (("L", "left"), ("R", "right")):
        femur = atlas.points(f"{side} femur")
        medial = femur[np.abs(femur[:, 1]) < np.abs(femur[:, 1].mean())]
        lm[f"thigh_{s}"] = near(medial, up, +1, 0.045)               # femoral head centre
        lm[f"shin_{s}"] = near(atlas.points(f"{side} tibia"), up, +1, 0.012)  # knee
        lm[f"foot_{s}"] = atlas.points(f"{side} talus").mean(axis=0)  # ankle
        lm[f"toes_{s}"] = near(atlas.points(f"{side} second metatarsal bone"), fwd, +1, 0.008)
        clav = atlas.points(f"{side} clavicle")
        out = left if s == "L" else -left
        lm[f"clavicle_{s}"] = near(clav, out, -1, 0.01)              # sternoclavicular joint
        lm[f"scapula_{s}"] = near(clav, out, +1, 0.01)               # acromioclavicular joint
        lm[f"humerus_{s}"] = near(atlas.points(f"{side} humerus"), up, +1, 0.04)
        lm[f"forearm_{s}"] = near(atlas.points(f"{side} humerus"), up, -1, 0.012)  # elbow
        lm[f"hand_{s}"] = near(atlas.points(f"{side} radius", f"{side} ulna"), up, -1, 0.012)  # wrist
    return lm


# ------------------------------------------------------------------ which bones go where

def parts(body):
    s = {"L": "left", "R": "right"}.get(body[-1])
    base = body[:-2] if s else body
    fill = lambda names: [n.format(s=s) if "{s}" in n else f"{s} {n}" for n in names]
    if body == "pelvis":
        return ["left hip bone", "right hip bone", "sacrum", "coccyx"]
    if body == "skull":
        return ["skull", "mandible"]
    if body in VERTEBRAE:
        n = int(body[1:])
        if body == "C1":
            return ["atlas"]
        if body == "C2":
            return ["axis"]
        names = [vertebra_name(body)]
        if body[0] == "T":
            names += [f"{side} {ORDINAL[n - 1]} rib" for side in ("left", "right")]
            names += [f"{side} {ORDINAL[n - 1]} costal cartilage" for side in ("left", "right")]
            if body == "T4":
                names.append("sternum")
        return names
    return {
        "thigh": fill(["femur", "patella"]),
        "shin": fill(["tibia", "fibula"]),
        "foot": fill(FOOT),
        "toes": fill(TOES),
        "clavicle": fill(["clavicle"]),
        "scapula": fill(["scapula"]),
        "humerus": fill(["humerus"]),
        "forearm": fill(["radius", "ulna"]),
        "hand": fill(HAND),
    }[base]


# Segments whose bones ride on the parent's fit, offset to the child's own origin.
RIDES_ON = {"scapula": "clavicle", "toes": "foot", "hand": "forearm"}


def face_budget(body):
    base = body[:-2] if body[-2:] in ("_L", "_R") else body
    if body in VERTEBRAE:
        return FACES[REGION[body[0]]]
    return FACES[base]


# ------------------------------------------------------------------ fitting


def frame(a, ref):
    a = a / np.linalg.norm(a)
    b = ref - (ref @ a) * a
    b /= np.linalg.norm(b)
    return np.column_stack([a, b, np.cross(a, b)])


def standing_model():
    """Our model in its standing reference pose, without meshes."""
    from movesim import build_model

    xml = build_model.build(meshes=False)
    model = mujoco.MjModel.from_xml_string(xml, {"keyframes.xml": b"<mujoco><keyframe/></mujoco>"})
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    data.qpos[3] = 1  # pelvis upright, as if standing
    mujoco.mj_kinematics(model, data)
    return model, data


def fit(model, data, body, lm):
    """Similarity transform (scale, rotation, origin) taking real-skeleton points into body's frame."""
    b = model.body(body)
    rot = data.xmat[b.id].reshape(3, 3)  # body frame -> standing axes
    children = [model.body(i) for i in range(model.nbody) if model.body_parentid[i] == b.id]
    left, fwd = np.array([0, 1.0, 0]), np.array([1.0, 0, 0])
    ref = fwd if body.startswith("clavicle") else left
    if body == "pelvis":
        target_src = 0.5 * (lm["thigh_L"] + lm["thigh_R"])
        target_body = 0.5 * (model.body("thigh_L").pos + model.body("thigh_R").pos)
    elif body == "skull":
        target_src, target_body = lm["vertex"], np.array([0.01, 0, 0.14])
    else:
        child = next(c for c in children if c.name in lm and not c.name.startswith(("clavicle", "thigh")))
        target_src, target_body = lm[child.name], child.pos.copy()
    src = frame(target_src - lm[body], ref)
    dst = frame(target_body, rot.T @ ref)
    r = dst @ src.T
    scale = np.linalg.norm(target_body) / np.linalg.norm(target_src - lm[body])
    return scale, r, lm[body]


def simplify(verts, faces, keep):
    if len(faces) <= keep:
        return verts, faces
    v, f = fast_simplification.simplify(verts.astype(np.float32), faces.astype(np.int32), 1 - keep / len(faces))
    return v, f


def write_stl(path, verts, faces):
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    with open(path, "wb") as f:
        f.write(b"movesim bone, from BodyParts3D (DBCLS, CC BY 4.0)".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = normals, tri
        f.write(rec.tobytes())


MESH_BODIES = (["pelvis"] + VERTEBRAE + ["skull"]
               + [f"{b}_{s}" for s in "LR" for b in
                  ("thigh", "shin", "foot", "toes", "clavicle", "scapula", "humerus", "forearm", "hand")])


def main(obj_dir, element_parts):
    atlas = Atlas(obj_dir, element_parts)
    lm = landmarks(atlas)
    model, data = standing_model()
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    fits, total = {}, 0
    for body in MESH_BODIES:
        base = body[:-2] if body[-2:] in ("_L", "_R") else body
        if base in RIDES_ON:
            scale, r, origin = fits[f"{RIDES_ON[base]}{body[-2:]}"]
            offset = model.body(body).pos
        else:
            scale, r, origin = fits[body] = fit(model, data, body, lm)
            offset = np.zeros(3)
        names = [n for n in parts(body) if atlas.has(n)]
        verts, faces = atlas.mesh(*names)
        verts = scale * (verts - origin) @ r.T - offset
        verts, faces = simplify(verts, faces, face_budget(body))
        write_stl(MESH_DIR / f"{body}.stl", verts, faces)
        total += len(faces)
        print(f"{body:12s} {len(names):2d} bones, {len(faces):5d} faces, scale {scale:.2f}")
    print(f"{len(MESH_BODIES)} meshes, {total} faces, in {MESH_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
