import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('files', type=str, nargs='+')
parser.add_argument('--aggregation', '-a', help='Type of aggregation\
                     that should be used to merge the submissions',
                    default='mean', choices=['mean'])
args = parser.parse_args()

dfs = [pd.read_csv(fn) for fn in args.files]
grouped_df = pd.concat(dfs).groupby('ID')

# check type of aggregation
if args.aggregation == 'mean':
    merged_df = grouped_df.mean()


# write out
merged_df.to_csv('merged.csv')
