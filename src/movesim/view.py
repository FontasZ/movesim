"""Open the skeleton in MuJoCo's viewer, lying in hook lying and ready to be moved.

On macOS run it with mjpython:  uv run mjpython -m movesim.view
"""

import mujoco.viewer

from movesim.sim import load_settled


def main():
    model, data = load_settled()
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
