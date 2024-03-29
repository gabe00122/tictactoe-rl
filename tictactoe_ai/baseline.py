from tictactoe_ai.agent_train import train_n_steps, StaticState, StepState
from tictactoe_ai.minmax.minmax_loader import get_minmax_agent
from tictactoe_ai.random_agent import RandomAgent
from tictactoe_ai.metrics import metrics_recorder
from tictactoe_ai.gamerules.initialize import initialize_n_games
from jax import random, numpy as jnp


def main():
    rng_key = random.PRNGKey(0)
    env_num = 128
    total_steps = 1000

    minmax_player, minmax_state = get_minmax_agent()
    random_player = RandomAgent()

    static_state = StaticState(
        env_num,
        opponent=random_player,
        agent=minmax_player,
        is_self_play=False,
        is_training=False,
    )

    rng_key, active_agent_keys = random.split(rng_key)
    active_agents = random.choice(
        active_agent_keys, jnp.array([-1, 1], dtype=jnp.int8), (env_num,)
    )

    step_state = StepState(
        rng_key=random.PRNGKey(0),
        opponent_state=None,
        agent_state=minmax_state,
        active_agent=active_agents,
        env_state=initialize_n_games(env_num),
        metrics_state=metrics_recorder.init(total_steps, env_num),
        step=jnp.int32(0),
        total_steps=jnp.int32(total_steps)
    )

    train_n_steps(static_state, total_steps, total_steps, step_state)


if __name__ == '__main__':
    main()
