from pathlib import Path

import PIL.Image
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    root = Path(__file__).parent
    data_paths = sorted(list((root / 'cka').rglob('*.pt')))
    output_dir = root / 'heatmaps'
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(data_paths):
        exp_name = path.parent.parent.parent.name
        dataset = path.parent.parent.parent.parent.name
        data = torch.load(path, map_location='cpu')
        for feat_idx in range(7):
            heatmap_data = data[feat_idx].cpu().numpy()
            output_path = output_dir / f'{dataset}_{exp_name}_ic{feat_idx}.png'
            plt.figure()
            plt.clf()
            plt.cla()
            heatmap = sns.heatmap(heatmap_data, vmin=0, vmax=1, annot=True, cbar=False, fmt='.2f')
            heatmap.set_title(f'{dataset}_{exp_name}_ic{feat_idx}')
            plt.savefig(output_path)

        # merge per-ic figures vertically
        output_path = output_dir / f'merged_{dataset}_{exp_name}.png'
        plt.figure()
        plt.clf()
        plt.cla()
        imgs = [PIL.Image.open(output_dir / f'{dataset}_{exp_name}_ic{feat_idx}.png') for feat_idx in range(7)]
        widths, heights = zip(*(i.size for i in imgs))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = PIL.Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in imgs:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        new_im.save(output_path)

        non_zero_entries = (data != 0).float().sum(dim=1)
        non_zero_entries[:, 0] = 1
        means = data.sum(dim=1) / non_zero_entries
        # plot means over ICs
        output_path = output_dir / f'mean_old_{dataset}_{exp_name}.png'
        plt.figure()
        plt.clf()
        plt.cla()
        heatmap = sns.heatmap(means, vmin=0, vmax=1, annot=True, cbar=False, fmt='.2f')
        heatmap.set_title(f'{dataset}_{exp_name} mean CKA across data seen so far')
        heatmap.set_xlabel('task idx')
        heatmap.set_ylabel('IC idx')
        plt.savefig(output_path)
