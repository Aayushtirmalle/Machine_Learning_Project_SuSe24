import numpy as np
import pickle
import os
import events as e
from .callbacks import *

def setup_training(self):
    global q_table
    if os.path.isfile("my-saved-q-table.pkl"):
        with open("my-saved-q-table.pkl", "rb") as file:
            q_table = pickle.load(file)
    else:
        q_table = np.zeros((state_size, len(ACTIONS)))
    logger.info("Training setup complete.")

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or new_game_state is None:
        logger.error("Received None for old_game_state or new_game_state.")
        return

    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    reward = 0
    for event in events:
        if event == e.COIN_COLLECTED:
            reward += 10
        elif event == e.CRATE_DESTROYED:
            reward += 5
        elif event == e.KILLED_OPPONENT:
            reward += 50
        elif event == e.GOT_KILLED:
            reward -= 100

    update_q_table(old_state, self_action, reward, new_state, alpha=0.1, gamma=0.9)

def end_of_round(self, last_game_state, last_action, events):
    if last_game_state is None:
        logger.error("Received None for last_game_state in end_of_round.")
        return

    with open("my-saved-q-table.pkl", "wb") as file:
        pickle.dump(q_table, file)
    logger.info("Q-table saved.")
