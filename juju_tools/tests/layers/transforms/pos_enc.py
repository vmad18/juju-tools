from utils.layers.transforms.pos_embed import PositionalEncoding


def main() -> None:
    pe = PositionalEncoding(512, 2048)


if __name__ == "__main__":
    main()

