from unittest import TestCase
from jax import numpy as jnp
from tictactoe_ai.gamerules.initialize import initialize_game
from tictactoe_ai.observation import get_observation


class ModelTest(TestCase):
    def test_x_observation(self):
        game = initialize_game()
        obs = get_observation(game, 1)
        expected_obs = jnp.array([
            1, 0, 0, 1, 0, 0, 1, 0, 0,
            1, 0, 0, 1, 0, 0, 1, 0, 0,
            1, 0, 0, 1, 0, 0, 1, 0, 0,
        ], dtype=jnp.float32)

        self.assertTrue(jnp.array_equal(obs, expected_obs).item())

    def test_o_observation(self):
        pass
