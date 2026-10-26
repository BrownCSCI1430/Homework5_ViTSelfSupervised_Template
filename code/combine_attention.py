"""
Combine Task 4 attention maps into side-by-side comparison figures (provided).

Task 4 saves one PNG per (image, style, model) using your visualize_attention():
    results/compare_<image>_<style>_<model>.png
This script stacks the four models for each image and style into one figure:
    results/compare_grid_<image>_<style>.png
with rows Random -> Rotation -> DINO -> DINOv3.

It runs automatically at the end of Task 4. To re-run it by itself:
    uv run python combine_attention.py
"""

import os
import re
import argparse

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = [('random', 'Random'), ('rotation', 'Rotation'),
          ('dino', 'DINO (ours)'), ('dinov3', 'DINOv3')]
STYLES = ['gray', 'fade']
_PATTERN = re.compile(
    r'^compare_(.+)_(' + '|'.join(STYLES) + r')_(' +
    '|'.join(m for m, _ in MODELS) + r')\.png$')


def combine_attention_maps(results_dir='results'):
    """Stack the per-model compare_*.png files into one figure per image/style."""
    # Find the per-model files: {(image, style): {model: path}}
    found = {}
    for f in sorted(os.listdir(results_dir)):
        match = _PATTERN.match(f)
        if match:
            image, style, model = match.groups()
            found.setdefault((image, style), {})[model] = os.path.join(results_dir, f)

    if not found:
        print(f"No compare_*.png files found in {results_dir}. Run Task 4 first.")
        return []

    saved = []
    for (image, style), paths in sorted(found.items()):
        rows = [(label, paths[m]) for m, label in MODELS if m in paths]
        missing = [label for m, label in MODELS if m not in paths]
        if missing:
            print(f"  {image} ({style}): missing {', '.join(missing)}; skipping those rows.")

        # Load each strip and scale to a common height
        strips = [np.array(Image.open(p).convert('RGB')) for _, p in rows]
        height = strips[0].shape[0]
        for i, s in enumerate(strips):
            if s.shape[0] != height:
                width = round(s.shape[1] * height / s.shape[0])
                strips[i] = np.array(Image.fromarray(s).resize((width, height)))

        # Pad narrower strips on the right (white) so the inputs line up,
        # then stack vertically with a small gap between rows
        gap = 20
        width = max(s.shape[1] for s in strips)
        canvas = []
        for s in strips:
            pad = np.full((height, width - s.shape[1], 3), 255, dtype=np.uint8)
            canvas.append(np.concatenate([s, pad], axis=1))
            canvas.append(np.full((gap, width, 3), 255, dtype=np.uint8))
        canvas = np.concatenate(canvas[:-1], axis=0)

        # Draw with a label column on the left
        dpi = 100
        label_px = 180
        title_px = 60
        fig_w = (canvas.shape[1] + label_px) / dpi
        fig_h = (canvas.shape[0] + title_px) / dpi
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor='white')
        ax = fig.add_axes([label_px / (fig_w * dpi), 0,
                           canvas.shape[1] / (fig_w * dpi),
                           canvas.shape[0] / (fig_h * dpi)])
        ax.imshow(canvas)
        ax.axis('off')
        for i, (label, _) in enumerate(rows):
            y = i * (height + gap) + height / 2
            ax.text(-20, y, label, ha='right', va='center',
                    fontsize=16, fontweight='bold', clip_on=False)
        style_name = 'Grayscale' if style == 'gray' else 'Fade to black'
        fig.suptitle(f'Attention comparison: {image} ({style_name})',
                     fontsize=18, fontweight='bold',
                     y=1 - (title_px / 2) / (fig_h * dpi))

        save_path = os.path.join(results_dir, f'compare_grid_{image}_{style}.png')
        fig.savefig(save_path, dpi=dpi, facecolor='white')
        plt.close(fig)
        saved.append(save_path)
        print(f"Attention comparison saved to {save_path}")

    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine Task 4 attention maps into side-by-side figures.')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
                        help='Directory containing the compare_*.png files.')
    args = parser.parse_args()
    combine_attention_maps(args.results_dir)
