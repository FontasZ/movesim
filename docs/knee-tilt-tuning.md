# Knee-tilt test: what went wrong and what was tuned

First test: hook lying, both knees guided 20° to the right and back. The aim was what a
Feldenkrais teacher would expect. The pelvis rolls right, the twist fades going up the spine,
the left side of the back lifts a little, the feet stay planted and the head stays nearly still.

## Problems along the way

1. **The head rolled 50° and stayed there.** The back of the head was a small sphere below the
   head's centre of mass, so the head sat on it like a ball balanced on a point. When it tipped,
   gravity carried it further. Fix: the contact sphere for the back of the head is now centred on
   the head's mass (radius 11 cm), so the head rolls like a real one and the neck brings it back.
2. **The feet crept along the floor even at rest (about 1.7 mm/s).** Soft contacts let constant
   loads slip slowly. Fix: elliptic friction cones, `impratio=10` and `noslip_iterations=5`.
3. **The pelvis hardly moved (1.9°) and the knees lagged the hands.** Fix: hip side-to-side and
   rotation stiffness from 30 to 60 N·m/rad. Guide springs from 400 to 1000 N/m.
4. **The feet still slipped and twisted (3–4° at the peak, about 1–2 cm of total slip).**
   - The leg springs aimed at a slightly different pose from the one the body settled into, so
     they pushed the feet across the floor all the time (shear at the friction limit). Fix: the
     springs' rest angles (hip 38.6°, knee 103.6°, ankle 30.2°) are now set to the settled pose.
   - Dropping the body onto sticky contacts locked in shear under the back. Fix: during settling,
     the back and head get 2 s of very low friction. MuJoCo uses the larger friction of the two
     touching surfaces, so the floor has to go low too. The soles keep their high friction.
   - The main cause was that the knee was a pure hinge. With the knee bent, the shin can turn
     about its own length (tibial rotation). Without it, every bit of hip rotation as the knees
     tip went into twisting the foot on the floor. Fix: a tibial rotation hinge (±25°, gentle
     spring). Slip fell to under 0.5 mm at five of the six foot points and 4 mm at the sixth.
   - The soles also got skin-on-mat friction (1.2) with some grip against pivoting.

## Result

The pelvis rolls about 9° right. The twist fades through the lumbar spine: L5 95%, L3 72%,
L1 32%, T9-10 11%, and nothing above. The left buttock comes off the floor, the left ribs unload
a little and the pressure under the back moves about 5 cm to the right. The feet roll onto their
edges (right foot onto its outer edge, left onto its inner edge) without sliding. The head does
not move. Everything returns to where it started.

## Known limits of this first model

- **The ribcage is fully rigid and fixed to T7-8.** Its broad back carries the upper body, so
  the twist stops sharply at T7-8, and the lower ribs on the left barely lift (about 8 N
  unloaded, no visible lift). A real ribcage is jointed at every thoracic level and would let a
  little more twist through. The next step could be to split the ribcage into a lower and upper
  part, or to tie it to several thoracic segments with soft constraints.
- The spine's neutral shape is straight, with no built-in lordosis or kyphosis. The lumbar back
  barely touches the floor.
- No arms. Their weight would mostly rest on the floor beside the body anyway.
- The stiffness and damping values are reasoned guesses, not measured.

## v0.2: the detailed skeleton

The model is now generated from anatomy tables (`src/movesim/build_model.py`): all 24
vertebrae with per-level ranges and built-in curves, a pair of ribs on every thoracic vertebra,
arms with collarbones and shoulder blades, slanted ankle and subtalar axes, and toes.

What needed tuning:

1. **The whole upper back rested on one patch (T7, 228 N), the shoulder blades on almost
   nothing.** The contact surfaces were too hard to spread the load over the curve of the back.
   Fix: softer flesh that gives way over about 2 cm, a slightly flatter lying thoracic curve
   (32° instead of 40°), and shoulder-blade surfaces 1 cm further back. Now T7–T9 and both
   shoulder blades share the load.
2. **Shear at the feet, again.** The settling spell of low friction now covers the feet too,
   so the legs find the pose where their tone carries them without pushing the feet along the
   floor. Their rest angles (hip 49.4°, knee 112.5°) are set so that this pose is hip 45°,
   knee 100°. A little toe-down ankle tone (40°) puts weight on the ball of the foot.

Result: the pelvis rolls 8.5° right. With a jointed ribcage the twist now travels further up:
L1 56%, T12 44%, T11 32%, T10 16%, then it stops at T9, where the floor holds the back of the
ribs. The left buttock lifts 2 cm and the left lower back 0.5–1.2 cm; the left ribs unload a
little and the right lower ribs take more. The feet roll onto their edges and slip at most
0.5 mm. The head and arms stay still. Everything returns to where it started.
