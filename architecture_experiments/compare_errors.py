import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


TECHNIQUES = ['implicit', 'memorization_masked_mlp']
FONTS = ['times_new_roman', 'arial', 'tahoma']


def main():
    for font in FONTS:
        dfs = [pd.read_csv(f'/home/ubuntu/data/results/{technique}/{font}/all_errors.csv')
               for technique in TECHNIQUES]

        dfs = [df[df.columns[1:]] for df in dfs]

        # for df, technique in zip(dfs, TECHNIQUES):
            # sns.kdeplot(df.values.ravel(), label=technique)

        plt.hist([df.values.ravel() for df in dfs], label=TECHNIQUES, bins=30)

        plt.title(font.replace('_', ' ').capitalize())
        plt.legend()

        plt.savefig(f'/home/ubuntu/data/results/{font}_errors.png')
        plt.close()


if __name__ == '__main__':
    main()
