from pathlib import Path

import tensorflowjs as tfjs
import tensorflow as tf
from tictactoe_ai.model_agent import ActorCriticAgent
from tictactoe_ai.train_settings import load_settings


def main():
    model_load_path = Path("./models/self_play_resnet")
    settings = load_settings(model_load_path / "settings.json")
    agent = ActorCriticAgent(settings["agent"], 0)
    state = agent.load(model_load_path / "checkpoint")
    model_params = state.training_state.model_params

    tfjs.converters.convert_jax(
        agent.model.model.apply,
        model_params,
        input_signatures=[tf.TensorSpec((29,), tf.float32), tf.TensorSpec((9,), tf.bool)],
        model_dir="./export"
    )
    print(model_params)


if __name__ == '__main__':
    main()
