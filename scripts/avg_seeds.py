import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path

sns.set_style('whitegrid')

if __name__ == '__main__':
    # files = [
    #     'results/CIFAR100/lwf_ee/10splits/seed0/cifar100_icarl_lwf_ee_test/results/ee_eval.json.npy',
    #     'results/CIFAR100/lwf_ee/10splits/seed1/cifar100_icarl_lwf_ee_test/results/ee_eval.json.npy',
    #     'results/CIFAR100/lwf_ee/10splits/seed2/cifar100_icarl_lwf_ee_test/results/ee_eval.json.npy'
    # ]
    # title = 'lwf'

    files = [
        'results/CIFAR100/finetuning_ex2000_ee/10splits/seed0/cifar100_icarl_finetuning_ee_test/results/ee_eval.json.npy',
        'results/CIFAR100/finetuning_ex2000_ee/10splits/seed1/cifar100_icarl_finetuning_ee_test/results/ee_eval.json.npy',
        'results/CIFAR100/finetuning_ex2000_ee/10splits/seed2/cifar100_icarl_finetuning_ee_test/results/ee_eval.json.npy'
    ]
    title = 'ft_e'

    th_costs = []
    th_accs = []
    ic_accs = []
    ic_costs = []
    for i, file in enumerate(files):
        data = np.load(file, allow_pickle=True).item()
        baseline_cost = data['avg']['baseline_cost']
        per_exit_cost = data['avg']['exit_costs'] / baseline_cost
        per_exit_acc = data['avg']['per_ic_acc']['tag']
        exit_cnts = data['avg']['per_th_exit_cnt']['tag']

        th_acc = data['avg']['per_th_acc']['tag']
        th_cost = (exit_cnts * per_exit_cost[None, :]).sum(axis=1)
        th_costs.append(th_cost)
        th_accs.append(th_acc)

        ic_accs.append(per_exit_acc)
        ic_costs.append(per_exit_cost)

    line_data = []
    th_costs = np.stack(th_costs, axis=0).mean(axis=0)
    for th_acc in th_accs:
        for i, cost in enumerate(th_costs):
            line_data.append({'cost': cost, 'acc': th_acc[i]})

    ic_data = []
    ic_costs = np.stack(ic_costs, axis=0).mean(axis=0)
    ic_accs = np.stack(ic_accs, axis=0).mean(axis=0)
    for acc, cost in zip(ic_accs, ic_costs):
        ic_data.append({'cost': cost, 'acc': acc})

    sns.lineplot(x='cost', y='acc', data=pd.DataFrame(line_data))
    plot = sns.scatterplot(x='cost', y='acc', data=pd.DataFrame(ic_data))
    output_dir = Path('figs')
    output_dir.mkdir(exist_ok=True, parents=True)
    plot.get_figure().savefig(str(output_dir / f'{title}.png'))
