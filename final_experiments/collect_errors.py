import pandas as pd
from functools import reduce


def main():
    # for technique in ['memorization_masked_mlp', 'implicit']:
    #     for FONT in ['times_new_roman', 'arial', 'tahoma']:
    #         dfs = []
    #         for glyph_idx in range(ord('a'), ord('z') + 1):
    #             GLYPH = chr(glyph_idx)
    #             EXP_NAME = f'lowercase_{GLYPH}'

    #             dfs.append(pd.read_csv(f'/home/ubuntu/data/results/{technique}/{FONT}/{EXP_NAME}/results/errors.csv'))
    #             dfs[-1].columns = ['size', dfs[-1].columns[1]]

    #         res_df = reduce(lambda df1, df2: df1.merge(df2, on='size'), dfs)
    #         res_df.to_csv(f'/home/ubuntu/data/results/{technique}/{FONT}/all_errors.csv', index=False)

    for technique in ['implicit_no_freq_encoding', 'implicit_no_residual', 'implicit_no_memorization']:
        for FONT in ['times_new_roman']:
            dfs = []
            for glyph_idx in range(ord('a'), ord('z') + 1):
                GLYPH = chr(glyph_idx)
                EXP_NAME = f'lowercase_{GLYPH}'

                dfs.append(pd.read_csv(f'/home/ubuntu/data/results/{technique}/{FONT}/{EXP_NAME}/results/errors.csv'))
                dfs[-1].columns = ['size', dfs[-1].columns[1]]

            res_df = reduce(lambda df1, df2: df1.merge(df2, on='size'), dfs)
            res_df.to_csv(f'/home/ubuntu/data/results/{technique}/{FONT}/all_errors.csv', index=False)


if __name__ == '__main__':
    main()
