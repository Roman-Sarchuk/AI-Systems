import API
import numpy as np
import random
import sys
import os
import argparse

# params
# CMD: --winp 7,7;7,8;8,7;8,8
win_position = [(4,4)]
# win_position = [(7,7),(7,8),(8,7),(8,8)]
START_POSITION = (0, 0)
Q_FILE = "q_table.npy"
NUM_WIN_STRIKES = 10

# Environment size
MAZE_WIDTH = API.mazeWidth()
MAZE_HEIGHT = API.mazeHeight()

NUM_ORIENTATIONS = 4  # North, East, South, West
NUM_IS_WALL = 2  # 0 or 1 (no wall or wall)
# Number of states and actions
# state = (x, y, orientation, is_front_wall)
NUM_STATES = MAZE_WIDTH * MAZE_HEIGHT * NUM_ORIENTATIONS * NUM_IS_WALL
NUM_ACTIONS = 3  # 0 - move forward, 1 - turn left, 2 - turn right
Q = np.zeros((NUM_STATES, NUM_ACTIONS))


# Q-learning parameters
ALPHA = 0.1      # learning rate
GAMMA = 0.9      # discount factor
EPSILON = 0.1    # exploration rate
EPISODES = 200
MAX_STEPS = 200




def save_q_table():
    np.save(Q_FILE, Q)
    API.log("Q-table saved.")

def load_q_table():
    global Q
    if os.path.exists(Q_FILE):
        Q = np.load(Q_FILE)
        API.log("Q-table loaded.")
    else:
        API.log("No saved Q-table found.")

def clear_q_table():
    if os.path.exists(Q_FILE):
        os.remove(Q_FILE)
        API.log("Saved Q-table deleted.")
    else:
        API.log("No saved Q-table to delete.")

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-S", action="store_true", help="Save Q-table")
    parser.add_argument("-R", action="store_true", help="Read Q-table")
    parser.add_argument("-C", action="store_true", help="Clear saved Q-table")

    parser.add_argument(
        "--winp",
        type=str,
        help="Winning positions as x,y;x,y;..."
    )

    args = parser.parse_args()

    win_positions = []
    
    if args.winp:
        pairs = args.winp.split(";")
        for pair in pairs:
            x_str, y_str = pair.split(",")
            win_positions.append((int(x_str), int(y_str)))

    return args.S, args.R, args.C, win_positions


def get_next_position(x, y, orientation):
    if orientation == 0:  # North
        return x, y + 1
    elif orientation == 1:  # East
        return x + 1, y
    elif orientation == 2:  # South
        return x, y - 1
    elif orientation == 3:  # West
        return x - 1, y


def get_state_index(x, y, orientation, is_wall):
    return (
        x * MAZE_HEIGHT * NUM_ORIENTATIONS * NUM_IS_WALL +
        y * NUM_ORIENTATIONS * NUM_IS_WALL +
        orientation * NUM_IS_WALL +
        is_wall
    )

def choose_action(state, epsilon=EPSILON):
    """
    Choose action using ε-greedy strategy.
    """
    if random.uniform(0, 1) < epsilon:
        # exploration: random action
        API.log("Exploring: choosing random action, e={:.2f}".format(epsilon))
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        # exploitation: best known action
        return int(np.argmax(Q[state]))

def main():
    API.log("Running...")

    for pos in win_position:
        if pos[0] >= MAZE_WIDTH or pos[1] >= MAZE_HEIGHT:
            API.log(f"Error: WIN_POSITION {pos} is out of maze bounds.")
            return

    API.setColor(START_POSITION[0], START_POSITION[1], "B")
    API.setText(START_POSITION[0], START_POSITION[1], "start")
    for pos in win_position:
        API.setColor(pos[0], pos[1], "G")
        API.setText(pos[0], pos[1], "win0")
    
    prev_win_strikes = 0
    win_strikes = 0

    for episode in range(EPISODES):
        # starting position
        API.ackReset()
        x, y, orientation = 0, 0, 0
        done = False
        steps = 0
        epsilon = EPSILON

        if prev_win_strikes != win_strikes:
            for pos in win_position:
                API.setText(pos[0], pos[1], f"win{win_strikes}")

        prev_win_strikes = win_strikes

        while not done and steps < MAX_STEPS and win_strikes < NUM_WIN_STRIKES:
            epsilon = max(0.01, epsilon * 0.99)  # decreasing ε over time
            state = get_state_index(x, y, orientation, API.wallFront())
            action = choose_action(state, epsilon)
            
            # Execute action
            try:
                if action == 0:  # move forward
                    API.moveForward()
                    x, y = get_next_position(x, y, orientation) 
                    reward = -0.1 
                elif action == 1:  # turn left
                    API.turnLeft()
                    orientation = (orientation - 1) % NUM_ORIENTATIONS
                    reward = -0.1
                elif action == 2:  # turn right
                    API.turnRight()
                    orientation = (orientation + 1) % NUM_ORIENTATIONS
                    reward = -0.1
                # reward = -0.1  # small penalty for each step to encourage faster goal achievement

                # check if goal reached
                if (x, y) in win_position:
                    reward = 1000
                    win_strikes += 1
                    done = True  # don't end episode if goal reached
                    API.log(f"Mouse reached win#{win_strikes}! R: +++++++++1000\n\n")
            except API.MouseCrashedError:
                API.setWall(x, y, API.DIRECTIONS[orientation])
                reward = -100
                win_strikes = 0
                done = True
                API.log("Mouse crashed! R: ---------100\n\n")

            # new state after action
            new_state = get_state_index(x, y, orientation, API.wallFront())
            Q[state, action] = Q[state, action] + ALPHA * (reward + GAMMA * np.max(Q[new_state]) - Q[state, action])
            steps += 1

if __name__ == "__main__":
    # -S -R -C --winp 7,7;7,8;8,7;8,8
    save_flag, read_flag, clear_flag, win_position_arg = parse_arguments()
    API.log(f"Flags: {save_flag and 'S' or ''} {read_flag and 'R' or ''} {clear_flag and 'C' or ''}")
    
    if win_position_arg:
        win_position = win_position_arg
        API.log(f"Load winning positions: {win_position}")

    if clear_flag:
        clear_q_table()

    if read_flag:
        load_q_table()

    main()

    if save_flag:
        save_q_table()