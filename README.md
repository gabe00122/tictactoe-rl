
# Deep Reinforcement TicTacToe

A practice project using deep reinforcement with self play to learn tic-tac-toe.
This project includes a vectorized jax implementation of tictactoe as well as a custom actor critic model similar to a2c.

<img width="400" alt="Screenshot 2024-03-27 at 10 34 12 PM" src="https://github.com/gabe00122/tictactoe-rl/assets/4421994/4ba40616-354b-4f5c-8df6-8948de3c4b6f">


## Installation

Install tictactoe-rl with poetry

```bash
  cd ./tictactoe-rl
  poetry install
```

To install with cuda, first uncomment the cuda related sections from the pyproject.toml. As of when this was written cuda with jax is only supported on linux.

## Play

```bash
  # Play against a minmax agent
  poetry run enjoy --opponent minmax
  
  # Play against a trained model
  poetry run enjoy --model-path ./models/self_play

  # Visualize the models preferences
  poetry run enjoy --model-path ./models/random_play --render-preferences 
```

## Train a model
Training runs use json files to describe the hyper parameter

```bash
  poetry run train --settings ./experiments/self_play.json --output ./runs/model_directory
```

Here's an example experiment .json
```javascript
{
  "seed": "random", // seed can be a integer or the word "random"
  "total_steps": 10000, // how many turns to train for times the env_num
  "jit_steps": 1000, // how many training steps/turns to perform in a single jax operation
  "env_num": 128, // the number of simultaneous games to simulate at once. This also constitutes a batch of data during training.
  "agent": {
    "type": "actor_critic", // only "actor_critic" for now
    "discount": 0.99, // the discounted reward coefficient 
    "root_hidden_layers": [64, 64], // the shared mlp layers between the actor and the critic. More numbers represent depth and the magnitude represents width.
    "actor_hidden_layers": [32, 32, 32, 32], // the mlp network specific to the actor
    "critic_hidden_layers": [32, 32, 32, 32],// the mlp network specific to the critic
    "learning_rate": 0.0001, // the adam learning rate, this is decreased to zero as training pregresses
    "adam_beta": 0.99, // both beta1 and beta2 values for adam
    "weight_decay": 0, // the l2 regularization coefficient 
    "actor_coef": 0.2, // a coefficient for the actor part of the loss. The ciric coefficient is always assumed to be 1.0
    "entropy_coef": 0.01 // a coefficient for the entropy of the action distribution if its added to the loss
  },
  "opponent": {
    "type": "minmax" // determines what the agent is trained against. Can be "minmax", "random" or "self-play".
  }
}
```
