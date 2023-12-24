import sys
import random
import math
from collections import defaultdict
from tqdm import tqdm
import time
import pickle

class FifteenState:
    NUM_CELLS = 16
    FIRST_ROW_SOLVED_STATE = [1, 2, 3, 4, NUM_CELLS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    SECOND_ROW_SOLVED_STATE = [1, 2, 3, 4, 5, 6, 7, 8, NUM_CELLS, 0, 0, 0, 0, 0, 0, 0]
    ALL_ROWS_SOLVED_STATE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, NUM_CELLS]

    def __init__(self, numbers):
        if len(numbers) != self.NUM_CELLS:
            raise ValueError("Invalid size of numbers for state")
        self.numbers = numbers
        self.rows_solved = self.count_rows_solved()
        self.terminal = self.rows_solved == 4 or \
                        (self.rows_solved == 1 and self.count_masked() == 11) or \
                        (self.rows_solved == 2 and self.count_masked() == 7)
        self.hash_val = self.generate_hash()
        self.empty_cell_index = self.numbers.index(self.NUM_CELLS)
        self.value = 0.0
        self.action_state_dict = {}

    def mask(self, upper):
        return [0 if a > upper and a != self.NUM_CELLS else a for a in self.numbers]

    def generate_hash(self):
        if self.rows_solved == 0:
            return ','.join(map(str, self.mask(4)))
        elif self.rows_solved == 1:
            return ','.join(map(str, self.mask(8)))
        else:
            return ','.join(map(str, self.mask(15)))

    def count_rows_solved(self):
        for i in range(self.NUM_CELLS):
            if self.numbers[i] != i + 1:
                if i < 4:
                    return 0
                elif i < 8:
                    return 1
                else:
                    return 2
        return 4

    def count_masked(self):
        return self.numbers.count(0)

    def next_state(self, action):
        ints = self.numbers[:]
        action_cell = ints[action]
        ints[action] = self.NUM_CELLS
        ints[self.empty_cell_index] = action_cell
        return FifteenState(ints)

    def get_value(self):
        return self.value

    def set_value(self, value):
        if self.terminal:
            raise ValueError("Cannot set value for terminal state")
        self.value = value

    def get_hash(self):
        return self.hash_val

    def is_terminal(self):
        return self.terminal

    def is_solved(self):
        return self.rows_solved == 4

    def possible_actions(self):
        width = int(math.sqrt(self.NUM_CELLS))
        row = self.empty_cell_index // width
        col = self.empty_cell_index % width

        possible_actions = []
        if row - 1 >= 0:
            possible_actions.append(width * (row - 1) + col)
        if row + 1 < width:
            possible_actions.append(width * (row + 1) + col)
        if col - 1 >= 0:
            possible_actions.append(width * row + col - 1)
        if col + 1 < width:
            possible_actions.append(width * row + col + 1)

        return possible_actions

    def action_state(self):
        return self.action_state_dict
    
    def __str__(self):
        result = ""
        width = int(math.sqrt(self.NUM_CELLS))
        for i in range(self.NUM_CELLS):
            if self.numbers[i] == self.NUM_CELLS:
                result += "   |" if self.numbers[i] < 10 else "    |"
            else:
                result += f" {self.numbers[i]} |" if self.numbers[i] < 10 else f" {self.numbers[i]} |"
            if (i + 1) % width == 0:
                result += "\n"
        return result


class FifteenPuzzleSolver:
    def __init__(self):
        self.states = defaultdict(FifteenState)
        self.policy = {}
        self.f_discount_rate = 0.9
        self.f_theta = 0.01
        self.f_trials = 10000
        self.parse_command_line_args()

    def parse_command_line_args(self):
        args = sys.argv[1:]
        if len(args) != 4 or args[0] != '-th' or args[2] != '-g':
            print("Usage: python fifteen_solver.py -th <value> -g <value>")
            sys.exit(1)

        try:
            self.f_theta = float(args[1])
            self.f_discount_rate = float(args[3])
        except ValueError:
            print("Error: Invalid argument values.")
            sys.exit(1)

    def next_state(self, s, action):
        next_state = s.next_state(action)
        return self.states.get(next_state.get_hash(), None)

    def action_value(self, s):
        def func(action):
            next_state = self.next_state(s, action)
            return (
                self.reward(s, next_state)
                + self.f_discount_rate * next_state.get_value()
                if next_state
                else 0.0
            )

        return func

    def reward(self, current_state, next_state):
        if next_state.is_solved():
            return 100.0
        return 1.0 * (next_state.rows_solved - current_state.rows_solved)

    def run(self):
        self.generate_states_from(FifteenState.FIRST_ROW_SOLVED_STATE)
        self.generate_states_from(FifteenState.SECOND_ROW_SOLVED_STATE)
        self.generate_states_from(FifteenState.ALL_ROWS_SOLVED_STATE)
        self.calculate_optimal_values()
        self.calculate_optimal_policy()
        self.print_bad_states()
        self.save_model()  # Save the trained model
        print("Done!")
        
        self.play()

    def solve(self, current_state):
        t = 0
        while True:
            if current_state.is_solved():
                return t
            t += 1
            action = self.policy.get(current_state.get_hash())
            current_state = current_state.next_state(action)

    def play(self):
        current_state = self.generate_random_state()
        t = 0
        while True:
            print(f"T = {t}")
            print("-----------------")
            print(current_state)
            if current_state.is_solved():
                return
            t += 1
            action = self.policy.get(current_state.get_hash())
            current_state = current_state.next_state(action)

    def generate_random_state(self):
        s = FifteenState(FifteenState.ALL_ROWS_SOLVED_STATE)
        t = 0
        while t < 10000:
            actions = s.possible_actions()
            if not actions:
                break  # Prevent infinite loops
            s = s.next_state(random.choice(actions))
            t += 1
        return s

    def print_bad_states(self):
        bad_states = [k for k, v in self.states.items() if v.get_value() < 0]
        print(f"Bad States: {len(bad_states)}")

    def calculate_optimal_values(self):
        states = list(self.states.values())
        max_iterations = 1000  # Limit the number of iterations
        for _ in tqdm(range(max_iterations)):
            delta = 0
            for s in states:
                if not s.is_terminal():  # Skip terminal states
                    v = s.get_value()
                    s.set_value(max(map(self.action_value(s), s.possible_actions())))
                    delta = max(delta, abs(v - s.get_value()))
            if delta < self.f_theta:
                break


    def calculate_optimal_policy(self):
        for s in self.states.values():
            best_action = max(s.possible_actions(), key=self.action_value(s))
            self.policy[s.get_hash()] = best_action

    def generate_states_from(self, start_state):
        state = FifteenState(start_state)
        q = [state]
        while q:
            s = q.pop()
            if s.get_hash() not in self.states:
                self.states[s.get_hash()] = s
                for action in s.possible_actions():
                    next_state = s.next_state(action)
                    if next_state.get_hash() not in self.states:
                        q.append(next_state)
    
    def save_model(self, filename='trained_model.pkl'):
        # Save the trained model (states and policy) to a file
        with open(filename, 'wb') as file:
            pickle.dump(self.states, file)
            pickle.dump(self.policy, file)
        print(f"Model saved as {filename}")

    def load_model(self, filename='trained_model.pkl'):
        # Load the trained model from the file
        with open(filename, 'rb') as file:
            self.states = pickle.load(file)
            self.policy = pickle.load(file)
        print(f"Model loaded from {filename}")

if __name__ == "__main__":
    fifteen_solver = FifteenPuzzleSolver()
    try:
        fifteen_solver.load_model()
        print("Model loaded successfully!")
        fifteen_solver.play()
    except FileNotFoundError:
        print("Model not found. Training the model...")
        fifteen_solver.run()
