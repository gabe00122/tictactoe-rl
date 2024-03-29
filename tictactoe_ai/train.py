import argparse
import shutil
from pathlib import Path
from os import PathLike

from jax import random, numpy as jnp


from tictactoe_ai.agent_train import StepState, StaticState
from tictactoe_ai.agent_train import train_n_steps
from tictactoe_ai.gamerules import initialize_n_games
from tictactoe_ai.metrics import metrics_recorder
from tictactoe_ai.metrics.metrics_logger_np import MetricsLoggerNP
from tictactoe_ai.minmax.minmax_loader import get_minmax_agent
from tictactoe_ai.model_agent import ActorCriticAgent
from tictactoe_ai.random_agent import RandomAgent
from tictactoe_ai.train_settings import load_settings, save_settings, TrainSettings


def main():
    parser = argparse.ArgumentParser(
        prog="Train",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("-s", "--settings", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    settings_path = args.settings
    settings = load_settings(settings_path, update_stamp=True)
    train(settings, args.output)


def train(settings: TrainSettings, output: str | PathLike[str]) -> float:
    rng_key = random.PRNGKey(settings["seed"])
    env_num = settings["env_num"]
    total_steps = settings["total_steps"]
    jit_iterations = settings["jit_steps"]

    agent = ActorCriticAgent(settings["agent"], settings["total_steps"])
    random_agent = RandomAgent()
    minmax_agent, minmax_state = get_minmax_agent()

    match settings["opponent"]["type"]:
        case "random":
            opponent = random_agent
        case "minmax":
            opponent = minmax_agent
        case _:
            opponent = agent

    static_state = StaticState(
        settings["env_num"],
        opponent=opponent,
        agent=agent,
        is_self_play=settings["opponent"]["type"] == "self_play",
        is_training=True,
    )

    game_state = initialize_n_games(env_num)

    rng_key, agent_a_key, agent_b_key, active_agent_keys = random.split(rng_key, 4)
    active_agents = random.choice(
        active_agent_keys, jnp.array([-1, 1], dtype=jnp.int8), (env_num,)
    )

    if settings["opponent"]["type"] == "minmax":
        opponent_state = minmax_state
    else:
        opponent_state = static_state.opponent.initialize(agent_a_key, env_num)

    step_state = StepState(
        rng_key=rng_key,
        opponent_state=opponent_state,
        agent_state=static_state.agent.initialize(agent_b_key, env_num),
        active_agent=active_agents,
        env_state=game_state,
        metrics_state=metrics_recorder.init(total_steps, env_num),
        step=jnp.int32(0),
        total_steps=jnp.int32(total_steps)
    )
    step_state = train_n_steps(static_state, total_steps, jit_iterations, step_state)

    if settings["opponent"]["type"] == "minmax":
        minmax_state = step_state.opponent_state
    # ends eval

    metrics_logger = MetricsLoggerNP(total_steps)
    metrics_logger.log(step_state.metrics_state)

    save_path = Path(output)
    create_directory(save_path)

    save_settings(save_path / "settings.json", settings)
    save_metrics(save_path / "metrics.parquet", metrics_logger)

    static_state.agent.save(save_path / "checkpoint", step_state.agent_state)

    # evaluate
    print("Evaluating vs minmax")
    step_state = agent_evaluation(static_state, step_state, minmax_agent, minmax_state)
    minmax_score = score_game_outcomes(step_state)

    print("Evaluating vs random")
    step_state = agent_evaluation(static_state, step_state, random_agent, None)
    random_score = score_game_outcomes(step_state)
    combined_score = (minmax_score + random_score) / 2
    print(f"Minmax Score: {minmax_score}, Random Score: {random_score}, Combined: {combined_score}")

    return combined_score


def agent_evaluation(static_state: StaticState, step_state: StepState, opponent, opponent_state):
    env_num = static_state.env_num
    rng_key = step_state.rng_key
    rng_key, active_agent_keys = random.split(rng_key)
    active_agents = random.choice(
        active_agent_keys, jnp.array([-1, 1], dtype=jnp.int8), (env_num,)
    )
    game_state = initialize_n_games(env_num)

    static_state = static_state._replace(
        opponent=opponent,
        is_self_play=False,
        is_training=False,
    )
    step_state = step_state._replace(
        rng_key=rng_key,
        opponent_state=opponent_state,
        active_agent=active_agents,
        env_state=game_state,
        metrics_state=metrics_recorder.init(1000, env_num),
    )
    return train_n_steps(static_state, 1000, 1000, step_state)


def score_game_outcomes(step_state: StepState) -> float:
    outcomes = step_state.metrics_state.game_outcomes
    total_games = outcomes.sum()
    agent_a_x, agent_a_o, ties, agent_b_x, agent_b_o = (outcomes.sum(0) / total_games).tolist()
    score = (agent_b_x + agent_b_o) - (agent_a_x + agent_a_o)
    return score


def create_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_metrics(path: Path, metrics: MetricsLoggerNP):
    metrics.get_dataframe().to_parquet(path, index=False)


if __name__ == "__main__":
    main()
