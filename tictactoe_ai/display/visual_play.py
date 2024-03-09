from functools import partial
from pathlib import Path

import jax.random
import orbax.checkpoint as ocp
import pygame

from jax import numpy as jnp, random
from jaxtyping import PRNGKeyArray

from tictactoe_ai.gamerules.initialize import initialize_game
from tictactoe_ai.gamerules.turn import turn, reset_if_done
from tictactoe_ai.gamerules.types import GameState
from tictactoe_ai.model.actor_critic import ModelParams
from tictactoe_ai.model.actor_critic_model import ActorCriticModel
from tictactoe_ai.model.initalize import create_actor_critic
from tictactoe_ai.model.run_settings import load_settings
from tictactoe_ai.observation import get_observation, get_available_actions

screen_size = 600
cell_size = screen_size / 3
margin = 20


def load_params(path: Path, actor_critic: ActorCriticModel) -> ModelParams:
    observation_dummy = jnp.zeros((9 * 3), jnp.float32)
    mask_dummy = jnp.full((9,), True)
    random_params: ModelParams = actor_critic.init(
        random.PRNGKey(0), observation_dummy, mask_dummy
    )

    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = ocp.checkpoint_utils.construct_restore_args(random_params)
    return checkpointer.restore(
        path.absolute(), item=random_params, restore_args=restore_args
    )


@partial(jax.jit, static_argnums=(1,))
def play_round(
    player_action,
    actor_critic: ActorCriticModel,
    params: ModelParams,
    game: GameState,
    rng_key: PRNGKeyArray,
) -> tuple[GameState, PRNGKeyArray]:
    game = reset_if_done(game)
    game = turn(game, player_action)

    rng_key, action_key = jax.random.split(rng_key)

    obs = get_observation(game, -1)
    mask = get_available_actions(game)
    logits, value = actor_critic.apply(params, obs, mask)
    jax.debug.print("{}\n{}", logits, value)

    action = random.categorical(action_key, logits)

    # action = get_random_move(game, action_key)
    game = turn(game, action)
    return game, rng_key


def play(actor_critic: ActorCriticModel, params: ModelParams):
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    clock = pygame.time.Clock()
    running = True

    game_state = initialize_game()
    board = game_state["board"].flatten().tolist()

    rng_key = random.PRNGKey(0)

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
                index = y * 3 + x
                # game_state = turn(game_state, jnp.int8(index))
                game_state, rng_key = play_round(
                    jnp.int8(index), actor_critic, params, game_state, rng_key
                )

                board = game_state["board"].flatten().tolist()
                print(game_state["over_result"].game_state)

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("white")

        # RENDER YOUR GAME HERE
        render(screen, board)

        # flip() the display to put your work on screen
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
    path = Path("./run-selfplay")
    settings = load_settings(path / "settings.json")
    actor_critic = create_actor_critic(settings)
    model = actor_critic.model
    params = load_params(path / "model", model)

    play(model, params)


if __name__ == "__main__":
    main()
