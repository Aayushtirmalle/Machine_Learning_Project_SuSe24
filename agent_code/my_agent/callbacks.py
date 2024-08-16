import logging
import numpy as np
import os
import pickle
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Define the dimensions of the field
FIELD_WIDTH = 17
FIELD_HEIGHT = 17
# Define the available actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
state_size = 10000  # Example size, adjust based on actual state representation
q_table = np.zeros((state_size, len(ACTIONS)))

# Setup logging
logger = logging.getLogger(__name__)

def setup(self):
    global q_table
    if not os.path.isfile("my-saved-q-table.pkl"):
        q_table = np.zeros((state_size, len(ACTIONS)))
        logger.info("Q-table initialized.")
    else:
        with open("my-saved-q-table.pkl", "rb") as file:
            q_table = pickle.load(file)
        logger.info("Q-table loaded from file.")

def state_to_features(game_state):
    if game_state is None:
        logger.error("Received None for game_state.")
        return 0  # Fallback to a default state

    try:
        # Validate and extract self position
        self_info = game_state.get('self')
        if self_info is None or len(self_info) < 4:
            raise ValueError("Self information is missing or incomplete in game state.")
        
        own_position = self_info[3]  # Expecting (x, y) tuple here
        if not isinstance(own_position, tuple) or len(own_position) != 2:
            raise ValueError(f"Expected own position as (x, y) tuple but got {own_position}")
        
        # Extract other elements
        coins = game_state.get('coins', [])  # List of (x, y) tuples of coins
        field = game_state.get('field')  # 2D numpy array of the field
        opponents = [opponent[3] for opponent in game_state.get('others', [])]  # List of (x, y) tuples of opponents

        # Check if field is a valid numpy array
        if not isinstance(field, np.ndarray):
            raise ValueError("Field is not a numpy array.")
        
        # Initialize the feature vector
        field_width, field_height = field.shape
        features = np.zeros(field_width * field_height + len(coins) * 2 + len(opponents) * 2)
        idx = 0

        # Add own position (check if within bounds)
        if 0 <= own_position[0] < field_width and 0 <= own_position[1] < field_height:
            features[idx] = own_position[0]  # x-coordinate
            idx += 1
            features[idx] = own_position[1]  # y-coordinate
            idx += 1
        else:
            logger.error(f"Own position out of bounds: {own_position}")
            return 0  # Fallback to a default state

        # Add coin positions
        for coin in coins:
            if isinstance(coin, tuple) and len(coin) == 2 and 0 <= coin[0] < field_width and 0 <= coin[1] < field_height:
                features[idx] = coin[0]  # x-coordinate of coin
                idx += 1
                features[idx] = coin[1]  # y-coordinate of coin
                idx += 1
            else:
                logger.error(f"Coin position out of bounds or invalid: {coin}")

        # Add opponent positions
        for opponent in opponents:
            if isinstance(opponent, tuple) and len(opponent) == 2 and 0 <= opponent[0] < field_width and 0 <= opponent[1] < field_height:
                features[idx] = opponent[0]  # x-coordinate of opponent
                idx += 1
                features[idx] = opponent[1]  # y-coordinate of opponent
                idx += 1
            else:
                logger.error(f"Opponent position out of bounds or invalid: {opponent}")

        # Add flattened field information
        flattened_field = field.flatten()
        for x in range(field_width):
            for y in range(field_height):
                if idx < len(features):
                    features[idx] = flattened_field[y * field_width + x]
                    idx += 1
                else:
                    logger.error("Feature vector index out of bounds during field processing")
                    break

        # Return a hash of the features to create a unique state representation
        state_index = int(hash(features.tobytes()) % (2 ** 31 - 1))
        return min(state_index, 9999)  # Assuming Q-table size is 10000

    except Exception as e:
        logger.error(f"Error in state_to_features: {e}")
        return 0  # Fallback to a default state

        
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return np.random.choice(ACTIONS)
    else:
        return ACTIONS[np.argmax(q_table[state])]

def update_q_table(old_state, action, reward, new_state, alpha=0.1, gamma=0.9):
    global q_table

    try:
        # Convert state to feature index
        old_state_index = state_to_features(old_state)
        new_state_index = state_to_features(new_state)

        # Ensure the state indices are within bounds
        if not (0 <= old_state_index < q_table.shape[0]):
            logger.error(f"Old state index {old_state_index} is out of bounds.")
            old_state_index = 0  # Fallback to default
        if not (0 <= new_state_index < q_table.shape[0]):
            logger.error(f"New state index {new_state_index} is out of bounds.")
            new_state_index = 0  # Fallback to default

        # Map action to index
        action_index = ACTIONS.index(action)
        if action_index is None:
            logger.error(f"Invalid action: {action}")
            return

        # Update Q-value using Q-learning update rule
        old_q_value = q_table[old_state_index, action_index]
        future_q_value = np.max(q_table[new_state_index])
        new_q_value = old_q_value + alpha * (reward + gamma * future_q_value - old_q_value)
        q_table[old_state_index, action_index] = new_q_value

    except Exception as e:
        logger.error(f"Error in update_q_table: {e}")


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    if game_state is None:
        self.logger.error("Received None for game_state in act.")
        return 'WAIT'  # Default action to prevent errors

    action = game_state.get('user_input', 'WAIT')  # Default to 'WAIT' if no user input is provided
    if action not in ACTIONS:
        self.logger.error(f"Invalid action received: {action}. Defaulting to 'WAIT'.")
        action = 'WAIT'
    return action
