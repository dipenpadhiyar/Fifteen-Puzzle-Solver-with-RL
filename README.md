# Fifteen Puzzle Solver

This project provides a Python-based solver for the classic Fifteen Puzzle using reinforcement learning techniques. The solver attempts to find an optimal solution to the puzzle by learning the best actions to take in different states of the game.

## Overview

The Fifteen Puzzle, also known as the "Game of Fifteen," is a sliding puzzle that consists of a 4x4 grid with numbered tiles (from 1 to 15) and one blank space. 
The goal is to rearrange the tiles from a random initial configuration to reach a specified goal state by sliding tiles into the blank space.

This solver uses Q-learning, a type of reinforcement learning, to determine the best moves in different states of the puzzle. The solver learns from the game states and rewards associated with each move to gradually improve its strategy for solving the puzzle.

## Features

- **Fifteen Puzzle Solver:** A Python implementation of a solver for the Fifteen Puzzle.
- **Reinforcement Learning:** Uses Q-learning to estimate the values of actions in different states.
- **Optimal Policy Extraction:** Derives the optimal policy for solving the puzzle.
- **State Representation:** Defines states, actions, and terminal conditions for the game.
- **Visualization:** Provides a display of the game states during play.

## Usage

To run the Fifteen Puzzle Solver:

1. Clone this repository to your local machine.
2. Ensure you have Python installed (Python 3.x recommended).
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Run the main script: `python fifteen_solver.py`.

If the trained model is available (`trained_model.pkl`), the solver will load it and attempt to solve random instances of the Fifteen Puzzle. If the model is not found, the solver will start training and then proceed to solve the puzzle.

## Files

- `fifteen_solver.py`: Main script containing the solver implementation.
- `requirements.txt`: File specifying the required Python dependencies.
- `trained_model.pkl`: Pre-trained model containing learned states and policies.

## Contribution

Contributions to this project are welcome! If you have suggestions, feature requests, or found a bug, please open an issue or create a pull request.
