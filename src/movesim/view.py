"""Open the skeleton in MuJoCo's viewer, lying in hook lying and ready to be moved.

Run it with plain python, also on macOS:  uv run python -m movesim.view
(mjpython is only for the passive viewer; with launch() it fails on macOS.)
"""

import mujoco.viewer

from movesim.sim import load_settled


def main():
    model, data = load_settled()
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
