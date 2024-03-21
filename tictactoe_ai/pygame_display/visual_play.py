from functools import partial
from typing import Any

import random as py_random
import pygame
import jax
from jax import numpy as jnp, random
from jaxtyping import PRNGKeyArray
from pathlib import Path

from tictactoe_ai.agent import Agent
from tictactoe_ai.gamerules import ONGOING, initialize_game, turn, GameState
from tictactoe_ai.minmax.minmax_player import MinmaxAgent
from tictactoe_ai.model_agent.observation import get_available_actions
from tictactoe_ai.model.run_settings import load_settings
from tictactoe_ai.model.initalize import create_actor_critic
from tictactoe_ai.model_agent import ActorCriticAgent

screen_size = 600
cell_size = screen_size / 3
margin = 20

player_turn = jax.jit(turn)


def opponent_turn(
    agent: Agent,
    agent_state: Any,
    game: GameState,
    rng_key: PRNGKeyArray,
) -> tuple[GameState, PRNGKeyArray]:
    rng_key, action_key = jax.random.split(rng_key)
    action = agent.act(agent_state, game, action_key)

    game = turn(game, action)

    return game, rng_key


def agent_goes_first(
    agent: Agent, agent_state: Any, rng_key: PRNGKeyArray
) -> tuple[GameState, PRNGKeyArray]:
    game = initialize_game()
    rng_key, action_key = random.split(rng_key)
    action = agent.act(agent_state, game, action_key)
    game = turn(game, action)
    return game, rng_key


def player_goes_first(rng_key: PRNGKeyArray) -> tuple[GameState, PRNGKeyArray]:
    return initialize_game(), rng_key


def start_game(
    agent: Agent, agent_state: Any, rng_key: PRNGKeyArray
) -> tuple[GameState, PRNGKeyArray]:
    rng_key, starting_player_key = random.split(rng_key)
    player_first = random.choice(
        starting_player_key, jnp.array([False, True], dtype=jnp.bool)
    )
    return jax.lax.cond(
        player_first,
        lambda: player_goes_first(rng_key),
        lambda: agent_goes_first(agent, agent_state, rng_key),
    )


@partial(jax.jit, static_argnums=(3,))
def handle_click(
    click_x: int,
    click_y: int,
    game: GameState,
    agent: Agent,
    agent_state: Any,
    rng_key: PRNGKeyArray,
) -> tuple[GameState, PRNGKeyArray]:
    def on_ongoing():
        index = jnp.int8(click_y * 3 + click_x)
        mask = get_available_actions(game)
        next_game = turn(game, index)

        return jax.lax.cond(
            mask[index],
            lambda: jax.lax.cond(
                next_game.over_result == ONGOING,
                lambda: opponent_turn(agent, agent_state, next_game, rng_key),
                lambda: (next_game, rng_key)
            ),
            lambda: (game, rng_key)
        )

    return jax.lax.cond(
        game.over_result != ONGOING,
        lambda: start_game(agent, agent_state, rng_key),
        lambda: on_ongoing(),
    )


def play(agent: Agent, agent_state: Any):
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    clock = pygame.time.Clock()
    running = True

    rng_key = random.PRNGKey(py_random.getrandbits(63))

    game_state, rng_key = start_game(agent, agent_state, rng_key)
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
                game_state, rng_key = handle_click(
                    x, y, game_state, agent, agent_state, rng_key
                )

                board = game_state.board.flatten().tolist()

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("white")

        # RENDER YOUR GAME HERE
        render(screen, board)

        # flip() the pygame_display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60

    pygame.quit()


def render(screen: pygame.Surface, board: list[int]):
    render_board(screen)

    for i, cell in enumerate(board):
        x = i % 3
        y = i // 3
        match cell:
            case 1:
                render_x(screen, (x, y))
            case -1:
                render_o(screen, (x, y))


def render_board(screen: pygame.Surface):
    color = "grey"
    line_width = 2

    pygame.draw.line(
        screen, color, (cell_size, 0), (cell_size, screen_size), line_width
    )
    pygame.draw.line(
        screen, color, (cell_size * 2, 0), (cell_size * 2, screen_size), line_width
    )
    pygame.draw.line(
        screen, color, (0, cell_size), (screen_size, cell_size), line_width
    )
    pygame.draw.line(
        screen, color, (0, cell_size * 2), (screen_size, cell_size * 2), line_width
    )


def render_x(screen: pygame.Surface, pos: tuple[int, int]):
    color = "black"
    line_width = 20
    x, y = pos
    x *= cell_size
    y *= cell_size

    pygame.draw.line(
        screen,
        color,
        (x + margin, y + margin),
        (x + cell_size - margin, y + cell_size - margin),
        line_width,
    )
    pygame.draw.line(
        screen,
        color,
        (x + cell_size - margin, y + margin),
        (x + margin, y + cell_size - margin),
        line_width,
    )


def render_o(screen: pygame.Surface, pos: tuple[int, int]):
    color = "black"
    line_width = 15
    x, y = pos

    x *= cell_size
    y *= cell_size

    pygame.draw.circle(
        screen,
        color,
        (x + cell_size / 2, y + cell_size / 2),
        (cell_size / 2) - margin,
        line_width,
    )


def main():
    model = MinmaxAgent()
    params = model.load(Path("./optimal_play.npy"))

    # settings = load_settings(Path("./run/settings.json"))
    #
    # model = ActorCriticAgent(create_actor_critic(settings))
    # params = model.load(Path("./run-selfplay/checkpoint"))

    play(model, params)


if __name__ == "__main__":
    main()
