"""
CSCI 1430 — Homework 5: Vision Transformers and Self-Supervised Learning

Main entry point. Run tasks with:
    uv run python main.py --task t0_attention
    uv run python main.py --task t1_endtoend
    uv run python main.py --task t2_rotation
    uv run python main.py --task t3_dino
    uv run python main.py --task t4_transfer
"""

import os
import argparse
from collections import namedtuple

import torch

import student
import hyperparameters as hp
from tasks import t0_attention, t1_endtoend, t2_rotation, t4_transfer

# ============================================================================
# Output file tracking
# ============================================================================

Approach = namedtuple('Approach', ['label', 'weights', 'curve_train', 'curve_val'])

os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)

APPROACHES = {
    'endtoend':       Approach('End-to-end ViT (from scratch)',   'results/endtoend_classifier.pt',  'results/train_endtoend.npy',       'results/val_endtoend.npy'),
    'rotation':       Approach('Rotation-pretrained encoder',     'results/rotation_encoder.pt',     'results/train_rotation.npy',       'results/val_rotation_probe.npy'),
    'dino':           Approach('DINO-pretrained encoder',         'results/dino_encoder.pt',         'results/train_dino_loss.npy',      None),
    'frozen_random':  Approach('Frozen random ViT probe',         'results/frozen_random.pt',        'results/train_frozen_random.npy',  'results/val_frozen_random.npy'),
    'frozen_rotation':Approach('Frozen rotation probe',           'results/frozen_rotation.pt',      'results/train_frozen_rotation.npy','results/val_frozen_rotation.npy'),
    'frozen_dino':    Approach('Frozen DINO probe',               'results/frozen_dino.pt',          'results/train_frozen_dino.npy',    'results/val_frozen_dino.npy'),
    'finetune':       Approach('Finetune DINO',                   'results/finetune.pt',             'results/train_finetune.npy',       'results/val_finetune.npy'),
    'dinov3_probe':   Approach('Frozen DINOv3 probe',             'results/dinov3_probe.pt',         'results/train_dinov3_probe.npy',   'results/val_dinov3_probe.npy'),
}


# ============================================================================
# Task dispatchers
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='HW5: Self-Supervised Learning')
    parser.add_argument('--task', type=str, required=True,
                        choices=['t0_attention', 't1_endtoend',
                                 't2_rotation', 't3_dino', 't4_transfer'],
                        help='Which task to run.')
    parser.add_argument('--data', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data'),
                        help='Path to data directory.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Task: {args.task}")
    print(f"BANNER_ID: {student.BANNER_ID}")
    torch.manual_seed(student.BANNER_ID)

    if args.task == 't0_attention':
        t0_attention(device, args.data)

    elif args.task == 't1_endtoend':
        classify_data = student.SceneDataset(
            os.path.join(args.data, '15-scenes-csci1430'),
            image_size=hp.ENDTOEND_IMAGE_SIZE,
            batch_size=hp.ENDTOEND_BATCH_SIZE,
        )
        t1_endtoend(classify_data, device, APPROACHES)

    elif args.task == 't2_rotation':
        rotation_data = student.CropRotationDataset(
            device,
            os.path.join(args.data, 'single-images', 'train'),
            crop_size=hp.ROTATION_CROP_SIZE,
        )
        classify_data = student.SceneDataset(
            os.path.join(args.data, '15-scenes-csci1430'),
            image_size=hp.TRANSFER_IMAGE_SIZE,
            batch_size=hp.TRANSFER_BATCH_SIZE,
        )
        t2_rotation(rotation_data, classify_data, device, APPROACHES)

    elif args.task == 't3_dino':
        dino_data = student.DINOMultiCropDataset(
            device,
            os.path.join(args.data, 'single-images'),
        )
        student.t3_dino_pretrain(dino_data, device, APPROACHES)

    elif args.task == 't4_transfer':
        classify_data = student.SceneDataset(
            os.path.join(args.data, '15-scenes-csci1430'),
            image_size=hp.TRANSFER_IMAGE_SIZE,
            batch_size=hp.TRANSFER_BATCH_SIZE,
        )
        t4_transfer(classify_data, device, APPROACHES, args.data)

    print(f"\nTask {args.task} complete.")


if __name__ == '__main__':
    main()
