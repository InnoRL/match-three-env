import jax
import jax.numpy as jnp
from match3_env.env import EnvParams, EnvState, MatchThree
from match3_env.game_grid import MatchThreeGameGridStruct


class GameService:
    def __init__(self):
        self.env_params = EnvParams()
        self.env = MatchThree(self.env_params)

        self.step_env_jit = jax.jit(self.env.step_env)
        self.reset_env_jit = jax.jit(self.env.reset)

        self.reset()

    def _tiles_to_action(self, tile1, tile2):
        grid_h, grid_w = self.env_params.grid_size
        if tile1[0] == tile2[0]:
            # Horizontal swap.
            i = tile1[0]
            j = min(tile1[1], tile2[1])
            action = i * (grid_w - 1) + j
        elif tile1[1] == tile2[1]:
            # Vertical swap.
            i = min(tile1[0], tile2[0])
            j = tile1[1]
            action = i + j * (grid_h - 1) + grid_h * (grid_w - 1)
        return action

    def _action_to_tiles(self, action):
        grid_h, grid_w = self.env_params.grid_size
        if action < (grid_w - 1) * grid_h:
            # Horizontal swap.
            tile1 = [int(action // (grid_w - 1)), int(action % (grid_w - 1))]
            tile2 = [int(action // (grid_w - 1)), int(action % (grid_w - 1) + 1)]
        else:
            # Vertical swap.
            action -= (grid_w - 1) * grid_h
            tile1 = [int(action % (grid_h - 1)), int(action // (grid_h - 1))]
            tile2 = [int(action % (grid_h - 1) + 1), int(action // (grid_h - 1))]
        print(tile1, tile2)
        return tile1, tile2

    def reset(self):
        self.rng = jax.random.PRNGKey(42)
        self.rng, reset_key = jax.random.split(self.rng)
        _, self.state = self.reset_env_jit(reset_key, self.env_params)
        self.episode_return = 0

        return {
            "board": self.state.grid.grid.tolist(),
            "score": self.episode_return,
            "moves_left": self.env_params.max_steps_in_episode,
        }

    def swap(self, tile1, tile2):
        action = self._tiles_to_action(tile1, tile2)

        self.rng, action_key = jax.random.split(self.rng)
        _, self.state, reward, _, info = self.step_env_jit(
            action_key, self.state, action, self.env_params
        )
        self.episode_return += float(reward)

        return {
            "board": self.state.grid.grid.tolist(),
            "score": self.episode_return,
            "moves_left": int(self.env_params.max_steps_in_episode - self.state.time),
        }

    def swap_random(self):
        self.rng, action_key = jax.random.split(self.rng)
        action = self.env.action_space(self.env_params).sample(action_key)
        _, self.state, reward, _, info = self.step_env_jit(
            action_key, self.state, action, self.env_params
        )
        self.episode_return += float(reward)

        tile1, tile2 = self._action_to_tiles(action)
        return {
            "tile1": tile1,
            "tile2": tile2,
            "board": self.state.grid.grid.tolist(),
            "score": self.episode_return,
            "moves_left": int(self.env_params.max_steps_in_episode - self.state.time),
        }

    def swap_greedy(self):
        def _eval_action(key, grid, time, action) -> int:
            state = EnvState(grid=MatchThreeGameGridStruct(grid=grid), time=time)
            _, _, _, _, info = self.step_env_jit(key, state, action, self.env_params)
            matches = info["matches"]
            return matches.matches[0]

        _eval_action_vmap = jax.jit(jax.vmap(_eval_action, in_axes=0))

        self.rng, *action_keys = jax.random.split(self.rng, 145)
        action_keys = jnp.array(action_keys)

        grids = jnp.tile(self.state.grid.grid[jnp.newaxis], (144, 1, 1))
        times = jnp.tile(self.state.time, (144))
        actions = jnp.array(jnp.arange(144, dtype=int))
        matches = _eval_action_vmap(
            action_keys,
            grids,
            times,
            actions,
        )

        action = jnp.argmax(matches)
        self.rng, action_key = jax.random.split(self.rng)
        _, self.state, reward, _, info = self.step_env_jit(
            action_key, self.state, action, self.env_params
        )
        self.episode_return += float(reward)

        tile1, tile2 = self._action_to_tiles(action)
        return {
            "tile1": tile1,
            "tile2": tile2,
            "board": self.state.grid.grid.tolist(),
            "score": self.episode_return,
            "moves_left": int(self.env_params.max_steps_in_episode - self.state.time),
        }

    def swap_ppo(self):
        return self.swap_random()


game_services = [GameService() for _ in range(4)]
