import os
import pickle

import jax
import jax.numpy as jnp

from match_three_env.env import EnvParams, EnvState, MatchThree
from match_three_env.game_grid import MatchThreeGameGridStruct


def evaluate_greedy_agent(
    num_episodes: int = 100, max_steps_in_episode: int = 100, seed: int = 42
) -> tuple[float, float]:
    rng = jax.random.PRNGKey(seed)

    env_params = EnvParams(max_steps_in_episode=max_steps_in_episode)
    env = MatchThree(env_params)

    def eval_action(key, grid, time, action) -> int:
        state = EnvState(grid=MatchThreeGameGridStruct(grid=grid), time=time)
        _, _, _, _, info = env.step_env(key, state, action, env_params)
        matches = info["matches"]
        return matches.matches[0]

    eval_action_vmap = jax.jit(jax.vmap(eval_action, in_axes=0))

    def run_episode(carry, _):
        """Run a single episode with a greedy agent."""
        rng, episode_return = carry
        rng, reset_key, action_key = jax.random.split(rng, 3)

        obs, state = env.reset(reset_key, env_params)
        episode_return = 0.0
        done = jnp.array(False)

        def step_loop(carry, _):
            """Inner loop for one episode's steps."""
            rng, obs, state, episode_return, done = carry
            rng, *action_keys = jax.random.split(rng, 145)
            action_keys = jnp.array(action_keys)

            grids = jnp.tile(state.grid.grid[jnp.newaxis], (144, 1, 1))
            times = jnp.tile(state.time, (144))
            actions = jnp.array(jnp.arange(144, dtype=int))
            matches = eval_action_vmap(
                action_keys,
                grids,
                times,
                actions,
            )

            action = jnp.argmax(matches)
            obs, next_state, reward, done, _ = env.step_env(
                action_key, state, action, env_params
            )
            episode_return += reward

            return (rng, obs, next_state, episode_return, done), None

        # Run steps until episode terminates or max_steps reached
        (rng, _, state, episode_return, done), _ = jax.lax.scan(
            step_loop,
            (rng, obs, state, episode_return, done),
            None,
            length=max_steps_in_episode,
            unroll=1,
        )

        return (rng, episode_return), None

    run_episode_vmap = jax.jit(jax.vmap(run_episode, in_axes=(0, None)))

    rng, episode_key = jax.random.split(rng)
    episode_keys = jax.random.split(episode_key, num_episodes)
    initial_returns = jnp.zeros(num_episodes)

    (_, episode_returns), _ = run_episode_vmap((episode_keys, initial_returns), None)

    return episode_returns


def main():
    num_episodes = 100
    max_steps = 100
    seed = 42

    episode_returns = evaluate_greedy_agent(num_episodes, max_steps, seed)

    os.makedirs("output", exist_ok=True)
    with open("output/returns_greedy.pkl", mode="wb") as f:
        pickle.dump(episode_returns, f)

    mean_return = float(jnp.mean(episode_returns))
    std_return = float(jnp.std(episode_returns))
    print(f"\nNumber of episodes: {num_episodes}")
    print(f"Steps in episode: {max_steps}")
    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")


if __name__ == "__main__":
    main()
