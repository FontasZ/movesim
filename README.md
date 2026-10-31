# movesim

A passive human skeleton in MuJoCo for composing and checking Feldenkrais-style movement lessons.
You say what moves on purpose (the driver) and what stays still, and physics decides where the
rest of the body goes. A lesson in words plays as 3D motion in a web page, step by step, with
the instructions as captions. The pictures are for whoever writes or reviews lessons; students
never watch a screen.

The body is meant to be a good-enough reference adult, not a perfect one: rigid bones, no
muscles; joint limits, gentle springs and ligaments that stiffen toward the end of each range
stand in for soft tissue; gravity and a floor with friction.

## Setup

You need [uv](https://docs.astral.sh/uv/). It brings its own Python 3.12 and leaves the system
Python alone.

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh   # installs to ~/.local/bin
cd ~/Documents/Projects/movesim
uv sync
```

To turn lesson text into movement you need a Claude API key: copy `.env.example` to `.env` and
fill in `ANTHROPIC_API_KEY`.

## Play a lesson

```bash
uv run movesim lesson "Slowly let both knees go to the right, and come back. Then roll the head to the left."
uv run movesim lesson lessons/knees-and-head.txt      # or a whole lesson from a text file
uv run movesim serve                                   # then open http://127.0.0.1:8765/web/
```

`movesim lesson` asks Claude (`claude-opus-5-5`) to turn the text into a movement script, saves
the script in `lessons/<name>.json`, simulates it, and adds it to the web player. It also prints
what the text asks for that the body can't show yet (other positions, breathing, the eyes...).

The script is plain JSON, so you can edit it and play it again without Claude:

```bash
uv run movesim play lessons/knees-and-head.json
```

Each step has the teacher's words (`say`), the movements done together (`moves`), what stays
still (`hold`), and timing (`seconds` to go there, `stay`, `return`, `repeat`, `rest_after`,
`release` to let go of parts left in place). The movements:

| move | what it does | amount |
|---|---|---|
| `knees_tilt` | both knees tip to one side over the feet | degrees |
| `knee_open` / `knee_close` | one or both knees fall outward / move across | degrees |
| `foot_press` | a foot presses; that side of the pelvis lifts | cm |
| `knee_to_chest` | a knee comes toward the chest, the foot leaves the floor | degrees |
| `head_roll` | the head rolls to one side | degrees |
| `head_lift` | the head lifts, chin toward the chest | degrees |
| `pelvis_flatten` / `pelvis_arch` | the pelvis tilts: lower back toward / away from the floor | degrees |
| `pelvis_lift` | a small bridge | cm |
| `arm_lift` | an arm lifts toward the ceiling | degrees |
| `shoulder_lift` | a shoulder lifts from the floor | cm |

Holds: `head`, `pelvis`, `chest`, `knees`, `feet`, `left_arm`, `right_arm`, `shoulders`.

Each movement takes hold of a point of the body (a knee, the forehead, the front hip bones, the
breastbone, a palm, a shoulder, an ankle) with an invisible hand on a spring and guides it; the
green dots in the player show which hands are working. Only lying on the back with the knees bent
is available as a starting position so far.

## The web player

`uv run movesim serve`, then <http://127.0.0.1:8765/web/>. Pick a lesson at the top; play,
pause (space), scrub, change speed; click a step to jump to it; switch between side, above,
from the feet and from the head, or drag to turn the view.

## Move the body yourself

```bash
uv run movesim view
```

MuJoCo's viewer opens with the body in hook lying. Double-click a green ball (a guide's hand)
and drag it with Ctrl + right mouse button; the point it holds follows on a spring, and when you
let go the hand eases back and stops pulling. Double-click any bone and Ctrl + right drag to push
it directly. Simulation → Key → `hook_lying` → Load key starts again. On macOS use this command
(plain `python`), not `mjpython`.

## The knee-tilt test

```bash
uv run python -m movesim.knee_tilt
```

Knees 20° to the right and back, with a plain-language report of what each vertebra did, which
parts of the back lifted and how the pressure moved: `out/knee_tilt/report.md`, plus a CSV,
frame strips and an MP4.

## The body

About 70 kg and 1.76 m, 44 rigid segments, 118 joints.

- **Spine:** all 24 vertebrae and the skull, each joined to the one below by three hinges (bend,
  side-bend, twist) with its own range of motion (White & Panjabi) and resting curve.
- **Ribs** on every thoracic vertebra, so the ribcage bends and twists with the spine.
- **Shoulders and arms:** collarbones, shoulder blades, upper arms, forearms that turn the palm,
  hands. Arms lie by the sides, palms up.
- **Legs:** hips with three hinges, knees that bend and let the shin turn, slanted ankle and
  subtalar axes, toes.
- **Bone shapes and proportions** come from BodyParts3D, a CT-based skeleton of one adult man
  (© DBCLS, CC BY 4.0; see `model/meshes/ATTRIBUTION.md`), scaled to 1.76 m. Masses follow
  de Leva (1996).
- **Ligaments:** each joint is loose around rest and stiffens steeply toward the end of its
  range (`src/movesim/ligaments.py`).
- **Contact:** soft surfaces for the back of the pelvis, both sides of each vertebra's back, the
  shoulder blades, the back of the head, arms, heels, balls of the feet and toes.
- **Resting tone:** gentle springs hold the legs in hook lying without pushing the feet along
  the floor.

Not modelled: muscles (by design), discs that compress, breathing, the shoulder blade gliding
over the ribs.

## Files

- `src/movesim/build_model.py`: the anatomy tables and the model builder. Edit here, then
  `uv run movesim build`.
- `src/movesim/bones.py`: fits BodyParts3D bones onto the segments (run once; needs the
  BodyParts3D download, see its docstring).
- `src/movesim/lesson.py`: movement scripts, the movement vocabulary, and playback.
- `src/movesim/translate.py`: lesson text to script with Claude.
- `src/movesim/cli.py`: the `movesim` command.
- `src/movesim/ligaments.py`, `sim.py`, `view.py`, `knee_tilt.py`.
- `model/`: the generated model, settled keyframe and bone meshes.
- `lessons/`: lesson texts and scripts. `web/`: the player and played lessons.
- `docs/knee-tilt-tuning.md`: what was tuned along the way and why.
