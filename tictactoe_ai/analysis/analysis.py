import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class Analysis:
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @classmethod
    def load(cls, path: Path) -> 'Analysis':
        return cls(pd.read_parquet(path))

    def plot(self):
        df = self.df.rolling(100).mean()
        #loss = df["critic_loss"]

        df.plot.line()
        plt.show()


def main():
    analysis = Analysis.load(Path("../../run/metrics.parquet"))
    analysis.plot()


if __name__ == '__main__':
    main()
