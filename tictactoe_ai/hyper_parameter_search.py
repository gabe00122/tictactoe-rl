import optuna
from optuna.samplers import TPESampler
from pathlib import Path
import jax
from tictactoe_ai.train import train
from tictactoe_ai.train_settings import TrainSettings
from tictactoe_ai.model.agent_settings import AgentSettings
from tictactoe_ai.util import generate_seed


def sample_settings(trail: optuna.Trial) -> TrainSettings:
    return TrainSettings(
        git_hash=None,
        timestamp=None,
        seed=generate_seed(),
        total_steps=50_000,
        jit_steps=5000,
        env_num=128,
        agent=AgentSettings(
            type="actor_critic",
            discount=trail.suggest_float('discount', 0.8, 1.0),
            root_hidden_layers=[],
            actor_hidden_layers=[32, 32, 32, 32, 32, 32, 32, 32],
            critic_hidden_layers=[32, 32, 32, 32, 32, 32, 32, 32],
            learning_rate=trail.suggest_float('learning_rate', 0.0001, 0.01),
            adam_beta=trail.suggest_float('adam_beta', 0.9, 0.99),
            weight_decay=trail.suggest_float('weight_decay', 0.0, 0.0001),
            actor_coef=trail.suggest_float('actor_coef', 0.05, 0.4),
            entropy_coef=trail.suggest_float('entropy_coef', 0.001, 0.1),
        ),
        opponent={
            "type": "self_play"
        }
    )

base_path = Path("./search").absolute()

def objective(trial: optuna.Trial):
    # trial_num += 1
    settings = sample_settings(trial)
    result = train(settings, base_path / "1")
    jax.clear_caches()
    return result

def main():
    sampler = TPESampler(n_startup_trials=5)

    study = optuna.create_study(
        sampler=sampler,
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name="tictactoe"
    )
    try:
        study.optimize(objective, n_trials=1000)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    main()
