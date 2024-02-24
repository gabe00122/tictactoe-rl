import jax
from jax import random, numpy as jnp
from jaxlib import xla_client
from tictactoe_ai.gamerules.turn import turn
from tictactoe_ai.gamerules.initialize import initialize_game


def main():
    # comp = jax.xla_computation(initial_state, static_argnums=(0,))(6, random.PRNGKey(123))
    # dot_graph = comp.as_hlo_dot_graph()
    rng_key = random.PRNGKey(345)
    state = initialize_game()
    # state = action_phase(state, )

    compiled_text = jax.jit(turn).lower(
        state, jnp.int8(0)
    ).compile().as_text()

    dot_graph = xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(compiled_text))
    with open("out.dot", 'w') as f:
        f.write(dot_graph)


if __name__ == '__main__':
    main()
