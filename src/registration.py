from gymnax import register

def register_custom_env():
    register(
        id="CustomEnv-v0",  # Unique ID for your environment
        entry_point="path_to_your_file:CustomEnv",  # Path to your environment class
        max_episode_steps=1000,  # Maximum number of steps per episode
        reward_threshold=1.0  # Reward threshold for early stopping
    )

# Call the register function
register_custom_env()
