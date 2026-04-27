#!/usr/bin/env python3
"""
Run local Orbit Wars matches for quick benchmarking.

Examples:
  python run_local.py
  python run_local.py --games 20 --opponent random
  python run_local.py --games 30 --opponent ./agent_v1.py
"""

import argparse
from statistics import mean

from kaggle_environments import make


def parse_args():
    parser = argparse.ArgumentParser(description="Run local Orbit Wars matches.")
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
        "--games",
        type=int,
        default=10,
        help="Number of games to run. Default: 10",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Create environments with debug=True.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be greater than 0")

    wins = 0
    draws = 0
    losses = 0
    rewards = []

    for game_index in range(1, args.games + 1):
        env = make("orbit_wars", debug=args.debug)
        env.run([args.agent, args.opponent])
        final = env.steps[-1]

        my_reward = final[0].reward
        opp_reward = final[1].reward
        rewards.append(my_reward)

        if my_reward > opp_reward:
            wins += 1
            result = "WIN"
        elif my_reward < opp_reward:
            losses += 1
            result = "LOSS"
        else:
            draws += 1
            result = "DRAW"

        print(
            f"Game {game_index:>3}: {result:<4} "
            f"reward={my_reward} opponent_reward={opp_reward} "
            f"status={final[0].status}"
        )

    print()
    print(f"Agent     : {args.agent}")
    print(f"Opponent  : {args.opponent}")
    print(f"Games     : {args.games}")
    print(f"Record    : {wins}W {draws}D {losses}L")
    print(f"Win Rate  : {wins / args.games:.1%}")
    print(f"Avg Reward: {mean(rewards):.3f}")


if __name__ == "__main__":
    main()
