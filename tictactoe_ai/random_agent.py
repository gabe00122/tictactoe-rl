import jax

from tictactoe_ai.gamerules.types import GameState, VectorizedGameState
from jaxtyping import PRNGKeyArray, Int8, Float32, Key
from jax import numpy as jnp, random, Array
from tictactoe_ai.agent import Agent, S
from tictactoe_ai.metrics import Metrics


def get_random_move(state: GameState, rng_key: PRNGKeyArray):
    board = state.board
    available_moves = board.flatten() == 0

    count = jnp.count_nonzero(available_moves)
    probs = available_moves / count

    return random.choice(rng_key, jnp.arange(9), p=probs)


class RandomAgent(Agent[None]):
    def initialize(self) -> None:
        return

    def act(self, agent_state: None, game_states: VectorizedGameState, rng_keys: Key[Array, "vec"]) -> Int8[Array, "vec"]:
        return jax.vmap(get_random_move, (0, 0))(game_states, rng_keys)

    def learn(
        self,
        params: None,
        game_states: VectorizedGameState,
        actions: Int8[Array, "vec"],
        rewards: Float32[Array, "vec"], next_obs: VectorizedGameState
    ) -> tuple[None, Metrics]:
        return None, Metrics(
            state_value=jnp.float32(0),
            td_error=jnp.float32(0),
            actor_loss=jnp.float32(0),
            critic_loss=jnp.float32(0),
            entropy=jnp.float32(0),
        )


def main():
    agent: Agent = RandomAgent()
    state = agent.initialize()



if __name__ == '__main__':
    main()
