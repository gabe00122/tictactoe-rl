import pygame
from tictactoe_ai.pygame_display.constaints import screen_size, cell_size, margin


def render(
    screen: pygame.Surface, board: list[int], probs: list[float], display_probs: bool
):
    if display_probs:
        for i, prob in enumerate(probs):
            color = pygame.Color(255 - int(prob * 255), 255, 255, 255)
            x = (i % 3) * cell_size
            y = (i // 3) * cell_size
            pygame.draw.rect(screen, color, (x, y, cell_size, cell_size))

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
