"""Turn lesson text into a movement script with Claude.

The script vocabulary (movements, holds) comes from movesim.lesson, so the translator always
knows exactly what the body can be asked to do. The reply is constrained to the script's JSON
schema; whatever the text asks for that the body can't show goes into "not_shown".
"""

import json

import anthropic

from movesim.lesson import DIRECTIONS, HOLDS, MOVES, REACH_POINTS, check
from movesim.positions import POSITIONS

MODEL = "claude-opus-5-5"

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "start": {"type": "string", "enum": list(POSITIONS)},
        "steps": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "say": {"type": "string"},
                "moves": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "move": {"type": "string", "enum": list(MOVES)},
                        "side": {"type": "string", "enum": ["left", "right", "both", "none"]},
                        "amount": {"type": "number"},
                        "point": {"type": "string", "enum": list(REACH_POINTS) + ["none"]},
                        "toward": {"type": "string", "enum": list(REACH_POINTS) + list(DIRECTIONS) + ["none"]},
                    },
                    "required": ["move", "side", "amount", "point", "toward"],
                    "additionalProperties": False,
                }},
                "hold": {"type": "array", "items": {"type": "string", "enum": list(HOLDS)}},
                "seconds": {"type": "number"},
                "stay": {"type": "number"},
                "return": {"type": "boolean"},
                "repeat": {"type": "integer"},
                "rest_after": {"type": "number"},
                "release": {"type": "boolean"},
            },
            "required": ["say", "moves", "hold", "seconds", "stay", "return", "repeat", "rest_after", "release"],
            "additionalProperties": False,
        }},
        "not_shown": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "start", "steps", "not_shown"],
    "additionalProperties": False,
}


def system_prompt():
    positions = "\n".join(f"- {name}: {p.help}" for name, p in POSITIONS.items())
    moves = "\n".join(
        f"- {name} ({m.unit}, typical {m.default:g}; side: {'/'.join(m.sides)}"
        + ("" if len(m.positions) == len(POSITIONS) else f"; only in {', '.join(m.positions)}")
        + f"): {m.help}" for name, m in MOVES.items())
    points = ", ".join(f"{k} ({v})" for k, v in REACH_POINTS.items())
    return f"""You turn Feldenkrais-style movement lessons into movement scripts for a physics simulation of the human skeleton. The script plays in a 3D view so that a person composing or reviewing a lesson can see what each instruction does in the body.

The body is passive: a movement takes hold of a point of the body and guides it, or turns a joint, and physics decides where the rest of the body goes. So describe only what the lesson moves on purpose, never its consequences (do not add a pelvis roll for "knees to the right"; the simulation shows that).

Starting positions ("start"; choose the one the lesson begins in):
{positions}

Movements (amount in the unit given; use the typical value unless the text says small, large, or how far). Prefer the most direct one: a shorthand when it fits, a joint movement for actions of one joint ("turn the leg in", "bend the elbow", "turn the head"), a reach for a point going somewhere ("bring the elbow toward the knee", "slide the hand along the floor toward the feet"). Several movements in one step happen together.
{moves}

For "reach": "point" is the part that moves and "toward" is another point or a direction; for every other movement set both to "none". Points: {points}. Directions: {", ".join(DIRECTIONS)} (head/feet/front/back/left/right are the person's own; up/down are toward the ceiling/floor; inward/outward toward/away from the midline).

Parts that can be asked to stay still ("hold"): {", ".join(HOLDS)}. Use hold only when the text says to keep a part still.

Each step:
- say: the teacher's words for this step, short (one or two sentences, close to the text).
- moves: the movements done together in this step; empty for rests and for instructions about attention, sensing or breathing.
- seconds: time to go there (Feldenkrais pace: slow, usually 2-4 s); stay: time to remain there; return: whether to come back to where it started (false leaves the part there for the next steps, as in "let the knees stay to the right"); repeat: how many times (show at most 3 even if the text says more, and say so in "say"); rest_after: pause after the step (rests: 5-10 s); release: true to let go of everything left in place by earlier steps.

Keep the order of the lesson. The script has one starting position: if the lesson changes position (rolls to the side, sits up), script the part in the starting position and list the rest in "not_shown". Also list there what no movement here can show (eyes, tongue, breathing, imagined movement), keeping its words as a step with no moves when it belongs in the flow. Title: a short name for the lesson."""


def translate(text, model=MODEL):
    """Lesson text -> checked script dict."""
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=system_prompt(),
        messages=[{"role": "user", "content": text}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to translate this text.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The lesson is too long to translate in one go; try it in parts.")
    reply = next(b.text for b in response.content if b.type == "text")
    script = json.loads(reply)
    script["source"] = text
    return check(script)
