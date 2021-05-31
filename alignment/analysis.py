# import plotly.express as px
import plotly.express as px
import argparse
import sys
import numpy as np
import pandas as pd
import csv

END = 0.5


def gen_rows(stream, max_length=None):
    rows = csv.reader(stream)
    if max_length is None:
        rows = list(rows)
        max_length = max(len(row) for row in rows)
    for row in rows:
        yield row + [None] * (max_length - len(row))


def analyze(files, target, names):
    if target == 'onsets':
        start = 0
        step = 2
    elif target == 'offsets':
        start = 1
        step = 2
    elif target == 'both':
        start = 0
        step = 1
    else:
        raise Exception(
            "Please, provide a valid target type for the analysis: 'onsets', 'offsets' or 'both'"
        )

    target = []
    for file in files:
        file.seek(0)
        df = pd.DataFrame.from_records(list(gen_rows(file)))
        target.append(df.to_numpy(dtype=np.float)[start::step])

    for i in range(len(target)):
        target[i] = np.abs(target[i][~np.isnan(target[i])])

    thresholds = np.arange(0, END, END / 100)
    values = pd.DataFrame()
    for i, file in enumerate(target):
        values_th = []
        for th in thresholds:
            if target == 'both':
                ons_idx = np.argwhere(file[::2] <= th)
                offs_idx = np.argwhere(file[1::2] <= th)
                values_th.append(2 * np.count_nonzero(ons_idx == offs_idx) /
                                 len(file))
            else:
                values_th.append(len(file[file <= th]) / len(file))
        values[names[i]] = values_th
    values.index = thresholds

    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze csv files and plot precision of alignments over threshold')
    parser.add_argument('infile',
                        nargs='*',
                        type=argparse.FileType('rt'),
                        default=sys.stdin,
                        help="Files that will be analyzed")
    if len(sys.argv) < 2:
        parser.print_help()
    else:
        args = parser.parse_args()

        files = args.infile
        names = [file.name for file in files]

        print("Analyzing files...")
        values_ons = analyze(files, 'onsets', names)
        values_offs = analyze(files, 'offsets', names)
        values_both = analyze(files, 'both', names)

        print("Plotting...")
        df = pd.concat(
            {
                # 'ons': values_ons,
                # 'offs': values_offs,
                'both': values_both
            },
            axis=1)
        th = df.index
        th.values[-1] = END
        df = df.melt()
        df = df.rename(columns={
            'variable_0': 'Type',
            'variable_1': 'Method',
            'value': '% matches'
        })

        df['Method'] = df['Method'] + ' ' + df['Type']
        df['Time (s)'] = np.hstack([th.values for i in range(len(files))])

        fig = px.line(df,
                      x='Time (s)',
                      y='% matches',
                      # facet_col='Type',
                      color='Method')
        # fig.update_layout(yaxis_tickformat='%', yaxis={'dtick':0.05}, xaxis={'dtick':0.5})
        fig.update_yaxes(dtick=0.05, tickformat='%')
        fig.update_xaxes(dtick=END / 10)
        fig.show()
        fig.write_image("results/alignment.svg")
