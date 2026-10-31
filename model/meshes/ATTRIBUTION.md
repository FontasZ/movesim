# Bone meshes

The bone shapes in this folder are derived from **BodyParts3D**,
© The Database Center for Life Science (DBCLS), Japan.
<https://dbarchive.biosciencedbc.jp/en/bodyparts3d/>

BodyParts3D is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0),
per its README of 2025-02-27. The original OBJ files still carry the earlier notice
"BodyParts3D, (c) The Database Center for Life Science licensed under CC Attribution-Share
Alike 2.1 Japan". To respect both, these derived meshes are shared under
**CC BY-SA 4.0**.

Source: `partof_BP3D_4.0_obj_99.zip` (BodyParts3D 4.0, part-of tree, 99% polygon reduction).

Changes made by `src/movesim/bones.py`: selected the bones of each body segment, converted
millimetres to metres and to movesim's axes, placed and scaled each segment's bones so their
joint landmarks meet movesim's joints (scale factors 0.93–1.07), merged them per segment and
simplified them.
