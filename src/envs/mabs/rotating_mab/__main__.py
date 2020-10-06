#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Play the Rotating MAB game.

From the terminal:

    python nmrl_wa/envs/rotating_mab [--seed SEED]

The available commands are: press left or press right.
"""

import argparse

from src.envs.mabs.rotating_mab import RotatingMAB


def parse_args():
    parser = argparse.ArgumentParser(
        "rotating_mab", description="Play the RotMAB game. Commands: Left/Right."
    )
    parser.add_argument(
        "--probs",
        nargs="+",
        default=(0.9, 0.1),
        help="The winning probabilities for each arm.",
    )
    parser.add_argument("--seed", default=42, help="The random seed.")
    arguments = parser.parse_args()
    return arguments


def _print_arguments(arguments):
    print(", ".join(map(lambda x: x[0] + "=" + str(x[1]), vars(arguments).items())))


def main():
    # we put imports here due to problems while testing:
    # https://github.com/moses-palmer/pynput/issues/6
    import pynput
    from pynput.keyboard import Key

    arguments = parse_args()

    _print_arguments(arguments)
    env = RotatingMAB(winning_probs=arguments.probs, seed=arguments.seed)
    initial_state = env.reset()
    done = False
    total_reward = 0
    num_left = 0
    num_right = 0
    print("Initial state = ", initial_state)
    while not done:
        # The event listener will be running in this block
        with pynput.keyboard.Events() as events:
            # Block at most one second
            event = events.get()
            if event is None or type(event) != pynput.keyboard.Events.Press:
                continue
            if event.key == Key.left:
                action = 0
                num_left += 1
            elif event.key == Key.right:
                action = 1
                num_right += 1
            else:
                continue
            observation, reward, done, info = env.step(action)
            total_reward += reward
            print(
                "Obs={}, reward={:04.2f}, total_reward={: 4}, num_left={: 4}, num_right={: 4}".format(
                    observation, reward, total_reward, num_left, num_right
                )
            )
            if reward == 1.0:
                print(env.current_winning_probs)


if __name__ == "__main__":
    main()
