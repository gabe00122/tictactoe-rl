import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="Train",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    # parser.add_argument()
    args = parser.parse_args()


if __name__ == "__main__":
    main()
