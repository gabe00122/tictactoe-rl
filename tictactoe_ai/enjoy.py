import argparse
from pathlib import Path
from tictactoe_ai.pygame_display.visual_play import play
from tictactoe_ai.train_settings import load_settings
from tictactoe_ai.model_agent import ActorCriticAgent
from tictactoe_ai.random_agent import RandomAgent
from tictactoe_ai.minmax.minmax_loader import get_minmax_agent


def main():
    parser = argparse.ArgumentParser(
        prog="play_tictactoe",
        description="Play against a AI in a visual game of tictactoe",
    )

    parser.add_argument(
        "--opponent",
        choices=["random", "minmax", "a2c"],
        default="minmax",
        help="what agent your up against, for a2c a model path is required",
    )
    parser.add_argument(
        "--model-path", help="the directory for the saved training results"
    )
    parser.add_argument(
        "--render-preferences",
        action="store_true",
        help="if true then show the action preferences of the ai agent after it takes a turn",
    )
    parser.add_argument(
        "--play-as",
        choices=["x", "o", "random"],
        default="random",
        help="always start the player as the given side",
    )

    args = parser.parse_args()

    model = RandomAgent()
    params = None

    opponent = args.opponent
    if args.model_path is not None:
        opponent = "a2c"

    match opponent:
        case "minmax":
            model, params = get_minmax_agent()
        case "a2c":
            model_path = Path(args.model_path)
            settings = load_settings(model_path / "settings.json")
            model = ActorCriticAgent(settings["agent"], settings["total_steps"])
            params = model.load(model_path / "checkpoint")

    play(model, params, args.play_as, args.render_preferences)


if __name__ == "__main__":
    main()
