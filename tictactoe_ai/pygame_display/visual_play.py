import random as py_random
from functools import partial
from typing import Any, Literal

import jax
import pygame
from jax import numpy as jnp, random
from jaxtyping import PRNGKeyArray, Float32, Array

from tictactoe_ai.agent import Agent
from tictactoe_ai.gamerules import ONGOING, initialize_game, turn, GameState
from tictactoe_ai.model_agent.observation import get_available_actions
from tictactoe_ai.pygame_display.rendering import render
from tictactoe_ai.pygame_display.constaints import screen_size, cell_size

player_turn = jax.jit(turn)


def opponent_turn(
    agent: Agent,
    agent_state: Any,
    game: GameState,
    rng_key: PRNGKeyArray,
) -> tuple[GameState, PRNGKeyArray, Float32[Array, "9"]]:
    rng_key, action_key = jax.random.split(rng_key)
    action, probs = agent.act(agent_state, game, action_key)

    game = turn(game, action)

    return game, rng_key, probs


def agent_goes_first(
    agent: Agent, agent_state: Any, rng_key: PRNGKeyArray
) -> tuple[GameState, PRNGKeyArray, Float32[Array, "9"]]:
    game = initialize_game()
    rng_key, action_key = random.split(rng_key)
    action, probs = agent.act(agent_state, game, action_key)
    game = turn(game, action)
    return game, rng_key, probs


def player_goes_first(rng_key: PRNGKeyArray) -> tuple[GameState, PRNGKeyArray, Float32[Array, "9"]]:
    return initialize_game(), rng_key, jnp.zeros((9,), dtype=jnp.float32)


def start_game(
    agent: Agent, agent_state: Any, rng_key: PRNGKeyArray, play_as: Literal["x", "o", "random"]
) -> tuple[GameState, PRNGKeyArray, Float32[Array, "9"]]:
    match play_as:
        case "x":
            return player_goes_first(rng_key)
        case "o":
            return agent_goes_first(agent, agent_state, rng_key)
        case "random":
            rng_key, starting_player_key = random.split(rng_key)
            player_first = random.choice(
                starting_player_key, jnp.array([False, True], dtype=jnp.bool)
            )
            return jax.lax.cond(
                player_first,
                lambda: player_goes_first(rng_key),
                lambda: agent_goes_first(agent, agent_state, rng_key),
            )


@partial(jax.jit, static_argnums=(3, 6))
def handle_click(
    click_x: int,
    click_y: int,
    game: GameState,
    agent: Agent,
    agent_state: Any,
    rng_key: PRNGKeyArray,
    play_as: Literal["x", "o", "random"],
) -> tuple[GameState, PRNGKeyArray, Float32[Array, "9"]]:
    def on_ongoing():
        index = jnp.int8(click_y * 3 + click_x)
        mask = get_available_actions(game)
        next_game = turn(game, index)

        return jax.lax.cond(
            mask[index],
            lambda: jax.lax.cond(
                next_game.over_result == ONGOING,
                lambda: opponent_turn(agent, agent_state, next_game, rng_key),
                lambda: (next_game, rng_key, jnp.zeros((9,), dtype=jnp.float32)),
            ),
            lambda: (game, rng_key, jnp.zeros((9,), dtype=jnp.float32)),
        )

    return jax.lax.cond(
        game.over_result != ONGOING,
        lambda: start_game(agent, agent_state, rng_key, play_as),
        lambda: on_ongoing(),
    )


def play(agent: Agent, agent_state: Any, play_as: Literal["x", "o", "random"], display_probs: bool):
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    clock = pygame.time.Clock()
    running = True

    rng_key = random.PRNGKey(py_random.getrandbits(63))

    game_state, rng_key, probs = start_game(agent, agent_state, rng_key, play_as)
    board = game_state.board.flatten().tolist()

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                x //= cell_size
                y //= cell_size
                game_state, rng_key, probs = handle_click(
                    x, y, game_state, agent, agent_state, rng_key, play_as
                )

                probs = probs.tolist()
                board = game_state.board.flatten().tolist()

        screen.fill("white")
        render(screen, board, probs, display_probs)
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60

    pygame.quit()


