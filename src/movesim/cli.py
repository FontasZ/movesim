"""movesim: play Feldenkrais lessons and movement instructions on a physics skeleton.

    uv run movesim lesson "Slowly let both knees go to the right, and come back."
    uv run movesim lesson path/to/lesson.txt
    uv run movesim play lessons/some-lesson.json      # a script you wrote or edited
    uv run movesim serve                              # the web player
    uv run movesim view                               # MuJoCo's interactive viewer
    uv run movesim validate                           # check the body against published data
"""

import argparse
import functools
import http.server
import json
from pathlib import Path

from movesim.sim import ROOT

SCRIPTS = ROOT / "lessons"
WEB_LESSONS = ROOT / "web" / "lessons"
PORT = 8765


def url(slug, port=PORT):
    return f"http://127.0.0.1:{port}/web/?lesson={slug}"


def play_script(script, slug):
    from movesim.lesson import play, write_index

    print(f"Simulating {len(script['steps'])} steps…")
    meta = play(script, WEB_LESSONS / slug)
    write_index(WEB_LESSONS)
    print(f"Done: {meta['seconds']:.0f} s of movement. Watch it at {url(slug)}  (run `uv run movesim serve` first)")


def cmd_lesson(args):
    from dotenv import load_dotenv

    from movesim.lesson import slug as make_slug
    from movesim.translate import translate

    load_dotenv(ROOT / ".env")
    source = Path(args.text)
    text = source.read_text() if source.suffix in (".txt", ".md") and source.exists() else args.text
    print("Translating with Claude…")
    script = translate(text)
    name = args.name or make_slug(script["title"])
    SCRIPTS.mkdir(exist_ok=True)
    path = SCRIPTS / f"{name}.json"
    path.write_text(json.dumps(script, indent=2, ensure_ascii=False))
    print(f"Script: {path.relative_to(ROOT)} ({len(script['steps'])} steps)")
    for item in script.get("not_shown", []):
        print(f"  not shown: {item}")
    play_script(script, name)


def cmd_play(args):
    from movesim.lesson import load_script

    path = Path(args.script)
    play_script(load_script(path), args.name or path.stem)


def cmd_serve(args):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Web player at http://127.0.0.1:{args.port}/web/  (Ctrl+C to stop)")
        server.serve_forever()


def cmd_view(args):
    from movesim import view

    view.main()


def cmd_validate(args):
    from movesim import validate

    validate.main()


def cmd_build(args):
    from movesim import build_model

    build_model.main()


def main():
    parser = argparse.ArgumentParser(prog="movesim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("lesson", help="translate lesson text with Claude and simulate it")
    p.add_argument("text", help="the lesson or instruction, or a .txt/.md file holding it")
    p.add_argument("--name", help="file name for the script and the player (default: from the title)")
    p.set_defaults(func=cmd_lesson)
    p = sub.add_parser("play", help="simulate a movement script (JSON)")
    p.add_argument("script")
    p.add_argument("--name")
    p.set_defaults(func=cmd_play)
    p = sub.add_parser("serve", help="serve the web player")
    p.add_argument("--port", type=int, default=PORT)
    p.set_defaults(func=cmd_serve)
    sub.add_parser("view", help="open MuJoCo's interactive viewer").set_defaults(func=cmd_view)
    sub.add_parser("validate", help="check the body against published measurements").set_defaults(func=cmd_validate)
    sub.add_parser("build", help="rebuild the model from the anatomy tables and settle it").set_defaults(func=cmd_build)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
