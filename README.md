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

Each script starts in one position (`start`), and each step has the teacher's words (`say`),
the movements done together (`moves`), what stays still (`hold`), and timing (`seconds` to go
there, `stay`, `return`, `repeat`, `rest_after`, `release` to let go of parts left in place).

**Starting positions:** `hook_lying` (on the back, knees bent, feet standing), `lying_long`
(legs long), `side_lying_left` / `side_lying_right` (knees bent and stacked, head on a pillow),
`sitting_chair` (front half of a chair, feet on the floor). Each comes with the small muscle
tone that holds it, kept separate from the passive tissue: lying with the knees standing needs a
little hip tone or the knees fall open, and sitting upright needs a lot of trunk tone. While a
joint is turned on purpose, its tone follows the intention instead of resisting it.

**Movements** come in three kinds (the full list with help texts is `MOVES` in
`src/movesim/lesson.py`; `side` is left, right or both):

- *Joint intentions*, which turn a joint (the head and spine share the turn over their levels):
  `hip_bend`/`hip_straighten`, `hip_turn_in`/`out`, `hip_open`/`close`, `knee_bend`/`straighten`,
  `foot_flex`/`point`, `foot_roll_in`/`out`, `arm_raise`, `arm_out`, `arm_turn_in`/`out`,
  `elbow_bend`/`straighten`, `palm_turn_down`/`up`, `wrist_bend`/`back`, `head_turn`,
  `head_tilt`, `head_nod`, `head_back`, `spine_twist`, `spine_side_bend`, `spine_round`,
  `spine_arch` (degrees).
- *Reach*: a point of the body (`point`) goes toward another point or in a direction (`toward`),
  by `amount` cm, or all the way to the other point with 0. Points: knees, ankles, toes, hands,
  elbows, shoulders, forehead, chin, breastbone, pubic bone, front hip bones. Directions: up,
  down, head, feet, front, back, left, right, inward, outward (the person's own).
- *Shorthands* for lying on the back: `knees_tilt`, `knee_open`/`close`, `foot_press` (the pelvis
  lifts on that side), `knee_to_chest`, `head_roll`, `head_lift`, `pelvis_flatten`/`arch`,
  `pelvis_lift`, `arm_lift`, `shoulder_lift`.

Holds: `head`, `pelvis`, `chest`, `knees`, `feet`, `left_arm`, `right_arm`, `shoulders`,
`left_leg`, `right_leg`, `elbows`.

Points are guided by invisible hands on springs (the green dots in the player show which are
working); joint intentions by a pull toward the wanted angle. Everything else is passive. Moving
from one position to another (rolling over, coming to sit) is not available yet.

Sample lessons in `lessons/`: knees and head, elbow to knee (lying), turning to look behind
(sitting), the top arm slides (side-lying), and a test of the building blocks.

## The website

`uv run movesim serve`, then <http://127.0.0.1:8765/web/>:

- `web/index.html`: the landing page for practitioners (what it is, a live skeleton, movements to
  try, how to use it, the call to send movements and feedback, who is behind it).
- `web/player.html`: the player. Pick a movement; play, pause (space), scrub, change speed; click
  an instruction to jump to it; drag to turn the view. "How to use" explains it, and opens on the
  first visit. `player.html?lesson=<name>&embed=1&view=feet&from=4.5&to=26` shows only the moving
  body, looping, as on the landing page.
- `web/science.html`: what the body is made of, where its numbers come from, the validation
  checks (from `web/validation.json`, written by `movesim validate`), and references.

Lessons whose name starts with `test-` stay out of the public menu.

## Move the body yourself

```bash
uv run movesim view
```

MuJoCo's viewer opens with the body in hook lying. Double-click a green ball (a guide's hand)
and drag it with Ctrl + right mouse button; the point it holds follows on a spring, and when you
let go the hand eases back and stops pulling. Double-click any bone and Ctrl + right drag to push
it directly. Simulation → Key → `hook_lying` → Load key starts again. On macOS use this command
(plain `python`), not `mjpython`.

## Validation

```bash
uv run movesim validate
```

Compares the body with measurements of living people and writes `docs/validation.md`: lumbar
passive stiffness through the range in three planes (Watt et al. 2022), neck rotation stiffness
(De Boe et al.), hip rotation stiffness, and how much each lumbar level twists when the trunk is
turned 45° (Fujii et al. 2007). All 15 checks pass. The thoracic spine has no in-vivo data and is
assumed to resist like the lumbar levels. `docs/knee-tilt-tuning.md` tells the story of what was
tuned and why.

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
- **Passive tissue:** the spine stiffens steadily from rest toward the end of each range, as
  measured in vivo; the limbs are loose to halfway and then tighten
  (`src/movesim/ligaments.py`, parameters in `build_model.py`).
- **Contact:** soft surfaces for the back of the pelvis, both sides of each vertebra's back, the
  sides of the chest, the shoulder blades and shoulders, the back of the head, arms, thighs,
  calves, ankles, heels, balls of the feet and toes; the legs can rest on each other. Props: a
  pillow and a chair.
- **Tone:** per starting position, the least muscle tone that holds it (see above).

Not modelled: muscles (by design), discs that compress, breathing, the shoulder blade gliding
over the ribs.

## Files

- `src/movesim/build_model.py`: the anatomy tables and the model builder. Edit here, then
  `uv run movesim build`.
- `src/movesim/bones.py`: fits BodyParts3D bones onto the segments (run once; needs the
  BodyParts3D download, see its docstring).
- `src/movesim/lesson.py`: movement scripts, the movement vocabulary, and playback.
- `src/movesim/positions.py`: starting positions, their tone and props.
- `src/movesim/validate.py`: the checks against published measurements.
- `src/movesim/translate.py`: lesson text to script with Claude.
- `src/movesim/cli.py`: the `movesim` command.
- `src/movesim/ligaments.py`, `sim.py`, `view.py`, `knee_tilt.py`.
- `model/`: the generated model, settled keyframe and bone meshes.
- `lessons/`: lesson texts and scripts. `web/`: the player and played lessons.
- `docs/knee-tilt-tuning.md`: what was tuned along the way and why.
