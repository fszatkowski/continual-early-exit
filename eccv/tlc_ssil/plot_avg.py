from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    root = Path(__file__).parent
    data_files = sorted(list(root.rglob('data.csv')))
    output_dir = root / 'aggregated'
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_data = {}
    for data_file in data_files:
        run_name = data_file.parent.parent.parent.name
        dataset = data_file.parent.parent.parent.parent.name
        if (dataset, run_name) not in grouped_data:
            grouped_data[(dataset, run_name)] = []
        grouped_data[(dataset, run_name)].append(pd.read_csv(data_file).sort_values(by=['method', 'cost']))

    for (dataset, run_name), data in grouped_data.items():
        costs = [d['cost'].values for d in data]
        cost = 0
        cnt = 0
        for c in costs:
            cost += c
            cnt += 1
        cost /= cnt

        for i in range(len(data)):
            data[i]['cost'] = cost

    for (dataset, run_name), data in grouped_data.items():
        plt.figure()
        plt.cla()
        plt.clf()
        output_path = output_dir / f'{dataset}_{run_name}.png'
        cat_data = pd.concat(data)
        sns.lineplot(x='cost', y='acc', hue='method', data=cat_data).get_figure().savefig(str(output_path))
