#!/usr/bin/env python3
"""
Render a local Orbit Wars match to an HTML replay file.

Examples:
  python render_local.py
  python render_local.py --agent main.py --opponent random
  python render_local.py --output replay_vs_random.html
"""

import argparse
from pathlib import Path

from kaggle_environments import make


def parse_args():
    parser = argparse.ArgumentParser(description="Render a local Orbit Wars replay.")
    parser.add_argument(
        "--agent",
        default="main.py",
        help="Path to your agent file. Default: main.py",
    )
    parser.add_argument(
        "--opponent",
        default="random",
        help='Opponent agent path or built-in name such as "random". Default: random',
    )
    parser.add_argument(
        "--output",
        default="replay.html",
        help="Output HTML file path. Default: replay.html",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=900,
        help="Render width. Default: 900",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=700,
        help="Render height. Default: 700",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Create the environment with debug=True.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = make("orbit_wars", debug=args.debug)
    env.run([args.agent, args.opponent])

    html = env.render(mode="html", width=args.width, height=args.height)
    output_path = Path(args.output)
    output_path.write_text(html, encoding="utf-8")

    final = env.steps[-1]
    print(f"Saved replay to {output_path.resolve()}")
    for i, state in enumerate(final):
        print(f"Player {i}: reward={state.reward}, status={state.status}")


if __name__ == "__main__":
    main()
