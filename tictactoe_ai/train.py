import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="Train",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("--settings")
    parser.add_argument("--output-dir")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
