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

## Ligaments that stiffen toward the end of range

Joints used to have an even spring plus a fairly sudden limit. Now each joint is loose around
its resting angle and, from halfway to the end of its range, resistance grows exponentially to
an end moment (`src/movesim/ligaments.py`; end moments in the tables in `build_model.py`).
For L4-L5 twist: 0.9 N·m at 1°, 4.8 N·m at 1.7°, 11.7 N·m at the 2° limit.

At 20° of knee tilt this barely matters: each lumbar level twists 0.6–0.8°, below where its
ligaments start to tighten. The twist reaches a little further (T10 16% → 22% of the pelvis
roll, T9 2% → 5%); everything else is within a newton or a millimetre.

At 45° it shows. Levels near their limits now hand the twist on gradually instead of stopping
it at a wall, so more of it reaches the chest (roll in degrees, ligaments off → on):
pelvis 31.7 → 30.9, T10 15.6 → 17.6, T8 8.9 → 11.6, T6 4.7 → 7.5, T4 3.2 → 5.1.

## Validation against published measurements (`uv run movesim validate`)

The stiffness values were my estimates until here. They now come from measurements of living
people where those exist (all 15 checks in `docs/validation.md` pass):

- **Lumbar spine:** passive stiffness per quartile of range in flexion, side-bending and axial
  rotation, fitted to the in-vivo meta-analysis of Watt, Callaway & Williams (2022). Each level
  resists with c·(exp(a·d/r) − 1): it stiffens steadily from rest, roughly doubling every
  quarter of its range, instead of being loose to halfway.
- **Thoracic spine:** no in-vivo data found; assumed to resist like the lumbar levels. Its twist
  ranges are widened to ±5° (T1-T9) after Wilke et al. 2017, so that turning the trunk 45° on a
  held pelvis leaves each lumbar level the 1.2-1.7° measured by MRI (Fujii et al. 2007); the
  body gives 1.5°. L5-S1 twist is ±2° (in vivo at least 1.2°; White & Panjabi give 1°).
- **Neck:** axial rotation averages 0.07 N·m/° over its range (De Boe et al.). The old neck was
  much softer near rest.
- **Hips:** rotation stiffness 8 N·m/rad from two in-vivo studies (5.6 and 10.1), where the
  body had 60. Lying with the knees standing needs a little muscle tone as well, without which
  the knees fall open; that tone (8 N·m/rad, about the least that holds them) is now explicit
  and separate.

The body became less stiff where it matters most for these lessons, at the hips. Knees tipped
to the right now move in the hips first and the pelvis follows later, the twist travelling
higher as the movement grows: knees 20° → pelvis 0.7°; 30° → 5° (twist to T10); 45° → 15° (T8);
60° → 28° (T6, head turning slightly the other way). With the old, six-times-stiffer hips,
20° already rolled the pelvis 8°.

## Positions and building blocks

- **Tone per position.** Lying with the knees standing needs hip tone of about 7 N·m/rad or
  the knees fall open (8 used). Side-lying needs some hip-flexion tone to keep the knees bent.
  Sitting upright needs about 3000 N·m/rad per spinal level: below that the trunk folds forward
  over the knees, because 17 levels in series divide the stiffness.
- **The tone follows the intention.** Turning the head or chest while sitting first reached
  only half the asked angle: the posture's tone pulled back toward neutral. A person intending
  to turn lets that tone go in the direction of the turn, so while a joint is turned on purpose
  its tone now holds the new angle. Head turn 50° → 46°; chest and head together → the head
  turns about 65° and the upper chest 28°, while the pelvis stays on the chair.
- **Joint intentions** pull toward the wanted angle with the joint's own (implicit) damping.
  Adding explicit damping made the light spinal segments unstable.
- **Holding the head** takes three points (forehead, chin, temple); one point leaves it free to
  turn about that point.
