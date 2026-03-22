import API
import numpy as np
import random
import sys
import os
import json
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

# params
# Example win positions: "4,4" or "7,7;7,8;8,7;8,8"
win_position = [(4, 4)]
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


TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), "params.json")

DEFAULT_CONFIG = {
    "win_position": win_position,
    "start_position": START_POSITION,
    "q_file": Q_FILE,
    "num_win_strikes": NUM_WIN_STRIKES,
    "alpha": ALPHA,
    "gamma": GAMMA,
    "epsilon": EPSILON,
    "episodes": EPISODES,
    "max_steps": MAX_STEPS,
    "save_q": False,
    "load_q": False,
    "clear_q": False,
}


def parse_single_position(text):
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("Position must have format x,y")
    return int(parts[0].strip()), int(parts[1].strip())


def parse_win_positions(text):
    raw = text.strip()
    if not raw:
        raise ValueError("win_position cannot be empty")

    positions = []
    for token in raw.split(";"):
        token = token.strip()
        if token:
            positions.append(parse_single_position(token))

    if not positions:
        raise ValueError("win_position cannot be empty")
    return positions


def validate_positions(win_positions, start_position):
    sx, sy = start_position
    if sx < 0 or sy < 0 or sx >= MAZE_WIDTH or sy >= MAZE_HEIGHT:
        raise ValueError(
            f"START_POSITION {start_position} is out of maze bounds {MAZE_WIDTH}x{MAZE_HEIGHT}"
        )

    for pos in win_positions:
        x, y = pos
        if x < 0 or y < 0 or x >= MAZE_WIDTH or y >= MAZE_HEIGHT:
            raise ValueError(
                f"WIN_POSITION {pos} is out of maze bounds {MAZE_WIDTH}x{MAZE_HEIGHT}"
            )


def positions_to_text(positions):
    return ";".join(f"{x},{y}" for x, y in positions)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result_config = None
        self.setWindowTitle("Micromouse Simulator - Algorithm Settings")
        self.setMinimumWidth(520)
        self.setFont(QFont("Segoe UI", 9))
        self.setStyleSheet(
            """
    QDialog { 
        background-color: #f0f0f0; 
    }
    QGroupBox {
        font-weight: bold;
        border: 1px solid #c0c0c0;
        margin-top: 10px;
        padding-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 3px;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        padding: 4px;
        selection-background-color: #0078d7;
    }
    QPushButton {
        background-color: #e1e1e1;
        border: 1px solid #adadad;
        min-width: 80px;
        padding: 5px;
    }
    QPushButton:hover {
        background-color: #e5f1fb;
        border: 1px solid #0078d7;
    }
    QPushButton:pressed {
        background-color: #cce4f7;
    }
    QLabel {
        color: #333333;
    }
    """
        )
        self._build_ui()
        self._load_config_to_widgets(DEFAULT_CONFIG)

    def _build_ui(self):
        root = QVBoxLayout(self)

        params_group = QGroupBox("Parameters")
        params_form = QFormLayout()

        self.winp_edit = QLineEdit()
        self.winp_edit.setPlaceholderText("4,4 or 7,7;7,8;8,7;8,8")
        params_form.addRow("win_position", self.winp_edit)

        self.start_edit = QLineEdit()
        self.start_edit.setPlaceholderText("0,0")
        params_form.addRow("START_POSITION", self.start_edit)

        self.qfile_edit = QLineEdit()
        params_form.addRow("Q_FILE", self.qfile_edit)

        self.strikes_spin = QSpinBox()
        self.strikes_spin.setRange(1, 100000)
        params_form.addRow("NUM_WIN_STRIKES", self.strikes_spin)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setDecimals(4)
        self.alpha_spin.setSingleStep(0.01)
        params_form.addRow("ALPHA", self.alpha_spin)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0, 1.0)
        self.gamma_spin.setDecimals(4)
        self.gamma_spin.setSingleStep(0.01)
        params_form.addRow("GAMMA", self.gamma_spin)

        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0, 1.0)
        self.epsilon_spin.setDecimals(4)
        self.epsilon_spin.setSingleStep(0.01)
        params_form.addRow("EPSILON", self.epsilon_spin)

        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 10000000)
        params_form.addRow("EPISODES", self.episodes_spin)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(1, 10000000)
        params_form.addRow("MAX_STEPS", self.max_steps_spin)

        params_group.setLayout(params_form)
        root.addWidget(params_group)

        actions_group = QGroupBox("Actions")
        actions_layout = QGridLayout()
        self.save_check = QCheckBox("Save Q-table after run")
        self.load_check = QCheckBox("Load Q-table on start")
        self.clear_check = QCheckBox("Clear Q-table before start")
        actions_layout.addWidget(self.save_check, 0, 0)
        actions_layout.addWidget(self.load_check, 0, 1)
        actions_layout.addWidget(self.clear_check, 1, 0)
        actions_group.setLayout(actions_layout)
        root.addWidget(actions_group)

        buttons = QHBoxLayout()
        self.load_template_btn = QPushButton("Load Template")
        self.save_template_btn = QPushButton("Save Template")
        self.apply_btn = QPushButton("Apply/Run")
        self.cancel_btn = QPushButton("Cancel")

        self.load_template_btn.clicked.connect(self._load_template)
        self.save_template_btn.clicked.connect(self._save_template)
        self.apply_btn.clicked.connect(self._apply)
        self.cancel_btn.clicked.connect(self.reject)

        buttons.addWidget(self.load_template_btn)
        buttons.addWidget(self.save_template_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.apply_btn)
        root.addLayout(buttons)

        footer = QLabel("© Roman (PP-34)")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(footer)

    def _collect_config_from_widgets(self):
        config = {
            "win_position": parse_win_positions(self.winp_edit.text()),
            "start_position": parse_single_position(self.start_edit.text().strip()),
            "q_file": self.qfile_edit.text().strip(),
            "num_win_strikes": int(self.strikes_spin.value()),
            "alpha": float(self.alpha_spin.value()),
            "gamma": float(self.gamma_spin.value()),
            "epsilon": float(self.epsilon_spin.value()),
            "episodes": int(self.episodes_spin.value()),
            "max_steps": int(self.max_steps_spin.value()),
            "save_q": self.save_check.isChecked(),
            "load_q": self.load_check.isChecked(),
            "clear_q": self.clear_check.isChecked(),
        }

        if not config["q_file"]:
            raise ValueError("Q_FILE cannot be empty")

        validate_positions(config["win_position"], config["start_position"])
        return config

    def _load_config_to_widgets(self, config):
        self.winp_edit.setText(positions_to_text(config["win_position"]))
        self.start_edit.setText(
            f"{config['start_position'][0]},{config['start_position'][1]}"
        )
        self.qfile_edit.setText(config["q_file"])
        self.strikes_spin.setValue(config["num_win_strikes"])
        self.alpha_spin.setValue(config["alpha"])
        self.gamma_spin.setValue(config["gamma"])
        self.epsilon_spin.setValue(config["epsilon"])
        self.episodes_spin.setValue(config["episodes"])
        self.max_steps_spin.setValue(config["max_steps"])
        self.save_check.setChecked(config["save_q"])
        self.load_check.setChecked(config["load_q"])
        self.clear_check.setChecked(config["clear_q"])

    def _show_error(self, message):
        QMessageBox.critical(self, "Invalid settings", str(message))

    def _save_template(self):
        try:
            config = self._collect_config_from_widgets()
            serializable = {
                **config,
                "win_position": [list(p) for p in config["win_position"]],
                "start_position": list(config["start_position"]),
            }
            with open(TEMPLATE_FILE, "w", encoding="utf-8") as fh:
                json.dump(serializable, fh, indent=2)
            QMessageBox.information(
                self, "Template saved", f"Saved to {TEMPLATE_FILE}")
        except Exception as exc:
            self._show_error(exc)

    def _load_template(self):
        if not os.path.exists(TEMPLATE_FILE):
            QMessageBox.information(
                self, "Template not found", "params.json does not exist yet.")
            return

        try:
            with open(TEMPLATE_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            config = {
                "win_position": [tuple(p) for p in data["win_position"]],
                "start_position": tuple(data["start_position"]),
                "q_file": data["q_file"],
                "num_win_strikes": int(data["num_win_strikes"]),
                "alpha": float(data["alpha"]),
                "gamma": float(data["gamma"]),
                "epsilon": float(data["epsilon"]),
                "episodes": int(data["episodes"]),
                "max_steps": int(data["max_steps"]),
                "save_q": bool(data["save_q"]),
                "load_q": bool(data["load_q"]),
                "clear_q": bool(data["clear_q"]),
            }

            validate_positions(config["win_position"],
                               config["start_position"])
            self._load_config_to_widgets(config)
        except Exception as exc:
            self._show_error(f"Failed to load template: {exc}")

    def _apply(self):
        try:
            self.result_config = self._collect_config_from_widgets()
            self.accept()
        except Exception as exc:
            self._show_error(exc)


def save_q_table(q_file):
    np.save(q_file, Q)
    API.log(f"Q-table saved to {q_file}")


def load_q_table(q_file):
    global Q
    if os.path.exists(q_file):
        try:
            loaded_q = np.load(q_file)
            if loaded_q.shape == Q.shape:
                Q = loaded_q
                API.log("Q-table loaded successfully.")
            else:
                API.log(
                    f"Shape mismatch: {loaded_q.shape} vs {Q.shape}. New table created.")
        except Exception as e:
            API.log(f"Error loading Q-table: {e}")
    else:
        API.log("No saved Q-table found.")


def clear_q_table(q_file):
    if os.path.exists(q_file):
        os.remove(q_file)
        API.log("Q-table file deleted.")
    else:
        API.log("Nothing to delete.")


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


def choose_action(state, epsilon):
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


def main(config):
    global Q

    win_positions = config["win_position"]
    start_position = config["start_position"]
    q_file = config["q_file"]
    num_win_strikes_target = config["num_win_strikes"]
    alpha = config["alpha"]
    gamma = config["gamma"]
    initial_epsilon = config["epsilon"]
    episodes = config["episodes"]
    max_steps = config["max_steps"]

    Q = np.zeros((NUM_STATES, NUM_ACTIONS))

    if config["clear_q"]:
        clear_q_table(q_file)

    if config["load_q"]:
        load_q_table(q_file)

    validate_positions(win_positions, start_position)

    API.log("Running with selected parameters:")
    API.log(f"win_position={win_positions}")
    API.log(f"START_POSITION={start_position}")
    API.log(f"Q_FILE={q_file}")
    API.log(f"NUM_WIN_STRIKES={num_win_strikes_target}")
    API.log(
        f"ALPHA={alpha}, GAMMA={gamma}, EPSILON={initial_epsilon}, EPISODES={episodes}, MAX_STEPS={max_steps}"
    )
    API.log(
        f"Actions: save_q={config['save_q']} load_q={config['load_q']} clear_q={config['clear_q']}"
    )
    API.log("Running...")

    API.setColor(start_position[0], start_position[1], "B")
    API.setText(start_position[0], start_position[1], "start")
    for pos in win_positions:
        API.setColor(pos[0], pos[1], "G")
        API.setText(pos[0], pos[1], "win0")

    prev_win_strikes = 0
    win_strikes = 0

    for episode in range(episodes):
        # starting position
        API.ackReset()
        x, y, orientation = start_position[0], start_position[1], 0
        done = False
        steps = 0
        epsilon = initial_epsilon

        if prev_win_strikes != win_strikes:
            for pos in win_positions:
                API.setText(pos[0], pos[1], f"win{win_strikes}")

        prev_win_strikes = win_strikes

        while not done and steps < max_steps and win_strikes < num_win_strikes_target:
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
                if (x, y) in win_positions:
                    reward = 1000
                    win_strikes += 1
                    done = True  # don't end episode if goal reached
                    API.log(
                        f"Mouse reached win#{win_strikes}! R: +++++++++1000\n\n")
            except API.MouseCrashedError:
                API.setWall(x, y, API.DIRECTIONS[orientation])
                reward = -100
                win_strikes = 0
                done = True
                API.log("Mouse crashed! R: ---------100\n\n")

            # new state after action
            new_state = get_state_index(x, y, orientation, API.wallFront())
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )
            steps += 1

    if config["save_q"]:
        save_q_table(q_file)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = SettingsDialog()

    if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_config is not None:
        main(dialog.result_config)
