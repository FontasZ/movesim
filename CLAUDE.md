# movesim

A simulated skeleton that shows where a movement goes, for Feldenkrais practitioners: to look, to
explain, to compose lessons. Public site: https://fontasz.github.io/movesim/ (published from `web/`
by `.github/workflows/pages.yml` on every push to `master`, so a push is a publish).

The README describes how to run everything. `docs/knee-tilt-tuning.md` tells what was tuned and why.

## How to work on it

- The goal is a good-enough reference body, not a perfect one. Sources over guesses, but don't chase
  precision for its own sake.
- Edit the anatomy tables in `src/movesim/build_model.py`, not the generated `model/skeleton.xml`.
  After changing the body: `uv run movesim build`, `uv run movesim validate`, then replay the lessons
  (`uv run movesim play lessons/<name>.json`), since the recordings in `web/lessons/` are generated.
- Measure a movement from the recording rather than trusting how it looks; several bugs only
  showed up in the numbers.
- Movements are guided by handles and joint intentions; no explicit damping in joint intentions
  (it made light spinal segments unstable).

## The website

- Audience is practitioners, not academics. `index.html` and `model.html` stay in plain language and
  short; the academic text belongs in `researchers.html`.
- The site is about movesim only; don't mention the author's other projects.
- The landing page has no "who is behind it" or "built on research" section, by the author's
  choice (they read as sales). Don't add them back, or name the author on the site, without asking.
- Don't transcribe published Feldenkrais lessons; lessons here are written from scratch. Say
  "Feldenkrais-style" and keep the trademark notice.
- Bone meshes are CC BY-SA 4.0 with credit to BodyParts3D; the code is MIT.

## Open

- The two Tally form links go in `web/forms.js` (buttons say "coming soon" until then); add a privacy
  line to the footer at the same time.
- Translating lesson text with Claude (`movesim lesson`) has never run: no API key is set yet
  (`.env`, see `.env.example`). The sample lessons were translated by hand.
- Some references on `researchers.html` are incomplete or unverified (author lists, Manual Therapy
  study, page numbers).
- At the pelvic clock's 12 and 6 the model's chin direction hasn't been checked against a source or a
  practitioner.
- Model gaps: arms pass through the trunk and each other; opening the top arm in side-lying turns the
  chest only about 4°; a held head still turns somewhat; no transitions between positions.
- Maybe a gentler public name and a domain of its own.
