import jax
from jax import numpy as jnp, random
from unittest import TestCase
from tictactoe_ai.model.actor_critic import ActorCriticModel
from tictactoe_ai.model.mlp import Mlp


class ModelTest(TestCase):
    def test_model_shapes(self):
        rng_key = random.PRNGKey(43)
        dummy_input = jnp.zeros((9, 3))
        dummy_mask = jnp.full(9, True)

        body = Mlp(features=[64])
        actor = Mlp(features=[64, 9])
        critic = Mlp(features=[64, 1])

        model = ActorCriticModel(body, actor_head=actor, critic_head=critic)
        params = model.init(rng_key, dummy_input, dummy_mask)

        actor_logits, critic_value = model.apply(params, dummy_input, dummy_mask)

        self.assertEqual(actor_logits.shape, (9,))
        self.assertEqual(critic_value.shape, ())
