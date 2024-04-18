from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

sns.set_style('whitegrid')


def best_th_acc(costs, accs, cost_th):
    cost_acc = [(c, a) for c, a in zip(costs, accs)]
    cost_acc_below = [(c, a) for c, a in cost_acc if c <= cost_th]
    best_acc = max(cost_acc_below, key=lambda x: x[1])[1]
    return best_acc


def plot_compare(data, output_path):
    th_records = []
    for run_name, d in data.items():
        th_cost = d['th_costs']
        th_acc = d['th_accs']
        for c, a in zip(th_cost.tolist(), th_acc.tolist()):
            th_records.append({'name': run_name, 'cost': c, 'acc': a})
    ic_records = []
    for run_name, d in data.items():
        for c, a in zip(d['ic_costs'].tolist(), d['ic_accs'].values()):
            ic_records.append({'name': run_name, 'cost': c, 'acc': a})

    plt.cla()
    plt.clf()
    plt.figure()
    plot = sns.lineplot(x='cost', y='acc', hue='name', data=pd.DataFrame(th_records))
    plot = sns.scatterplot(x='cost', y='acc', hue='name', data=pd.DataFrame(ic_records), legend=False)
    plot.get_figure().savefig(str(output_path))


if __name__ == "__main__":
    root_name = 'results_athena_v2'
    output_dir = Path('parsed_results') / 'athena_v2'
    output_dir.mkdir(parents=True, exist_ok=True)

    root_dir = Path(root_name)
    result_dirs = list(root_dir.rglob('*/results'))
    assert len(result_dirs) > 0, f"Didn't find any results in {root_dir}"

    plot_data = {}
    output_data = []

    for r in tqdm(result_dirs):
        results = list(r.glob("*"))
        if len(results) == 0:
            print(f"Didn't find any results in {r}")
            continue

        ee_filenames = list(r.glob("ee_eval.npy"))
        final_results_filenames = list(r.glob("avg_accs_tag*"))
        if not (len(ee_filenames) == 1 or len(final_results_filenames) == 1):
            print(f"Didn't find any results for {r}")
            continue

        assert len(ee_filenames) == 1 or len(final_results_filenames) == 1

        run_name = str(r.parent.parent).replace(root_name, '')
        run_name = "_".join(run_name.strip('/').split('/')[1:])
        if len(ee_filenames) == 1:
            ee_filename = ee_filenames[0]
            ee_data = np.load(ee_filename, allow_pickle=True).item()['avg']
            exit_costs = ee_data['exit_costs']
            baseline_cost = ee_data['baseline_cost']
            per_ic_acc = ee_data['per_ic_acc']['tag'] * 100
            per_th_acc = ee_data['per_th_acc']['tag'] * 100
            per_th_exit_cnt = ee_data['per_th_exit_cnt']['tag']
            per_th_cost = (exit_costs[None, :] * per_th_exit_cnt).sum(1) / baseline_cost
            ic_accs = {f'ic_{i}': acc for i, acc in enumerate(per_ic_acc.tolist())}
            th_accs = {
                'th_025': best_th_acc(per_th_cost, per_th_acc, 0.25),
                'th_050': best_th_acc(per_th_cost, per_th_acc, 0.5),
                'th_075': best_th_acc(per_th_cost, per_th_acc, 0.75),
                'th_100': best_th_acc(per_th_cost, per_th_acc, 1.),
                'th_none': max(per_ic_acc),
            }
            output = {
                'run_name': run_name
            }
            output.update(ic_accs)
            output.update(th_accs)
            output_data.append(output)

            plot_data[run_name] = {
                'ic_accs': ic_accs,
                'ic_costs': exit_costs / baseline_cost,
                'th_costs': per_th_cost,
                'th_accs': per_th_acc
            }
        else:
            final_results_filename = final_results_filenames[0]
            with final_results_filename.open('r') as f:
                data = f.read().strip()
            final_val = float(data.split('\t')[-1])
            output = {
                'run_name': run_name,
                'no_ee_acc': 100 * final_val
            }
            output_data.append(output)

    data = pd.DataFrame(output_data)
    data = data.sort_values(by=['run_name'])
    # Data to csv formatting floats to 3 points
    data.to_csv(output_dir / 'data.csv', index=False, float_format='%.3f')

    plot_names = list(plot_data.keys())
    plot_prefixes = list(set([n.split("_")[0] for n in plot_names]))
    for p in tqdm(plot_prefixes, 'Plotting early exit scores...'):
        method_data = {method_name: method_data for method_name, method_data in plot_data.items() if
                       method_name.startswith(p)}
        sorted_method_data = {k: v for k, v in sorted(method_data.items(), key=lambda item: item[0])}
        plot_compare(sorted_method_data, output_dir / (p + ".png"))
