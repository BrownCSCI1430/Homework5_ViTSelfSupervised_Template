"""
Provided task drivers for HW5 (do not modify).

Task 0: Attention visualization — loads models, calls student.visualize_attention().
Task 4: Transfer evaluation — runs 5 probe experiments, generates attention comparison.
"""

import os
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import student
import hyperparameters as hp
from helpers import create_vit_tiny, load_dinov3_encoder


# ============================================================================
# Task 0: Attention visualization
# ============================================================================

def t0_attention(device, data_dir):
    """Visualize attention from pretrained and random ViTs."""
    print("Task 0: Attention Map Visualization")
    pretrained_model, _ = create_vit_tiny(pretrained=True)
    pretrained_model.eval()
    random_model, _ = create_vit_tiny()
    random_model.eval()

    to_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    sample_paths = []
    for root, dirs, files in os.walk(os.path.join(data_dir, 'single-images', 'train')):
        for f in sorted(files):
            if f.endswith(('.jpg', '.png')):
                sample_paths.append(os.path.join(root, f))

    for img_path in sample_paths[:2]:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        tensor = to_tensor(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)

        for label, model in [('pretrained', pretrained_model), ('random', random_model)]:
            student.visualize_attention(
                model, tensor,
                os.path.join('results', f'attn_fade_{label}_{img_name}.png'),
                style='fade', device=device)
            student.visualize_attention(
                model, tensor,
                os.path.join('results', f'attn_gray_{label}_{img_name}.png'),
                style='gray', device=device)

    print("  Done. Compare pretrained vs random in your reslog.")


# ============================================================================
# Task 1: End-to-end ViT classification
# ============================================================================

def t1_endtoend(classify_data, device, approaches):
    """Train ViTEncoder from scratch on 15-scenes."""
    torch.manual_seed(student.BANNER_ID)
    classifier = student.ViTEncoder(nn.Linear(192, classify_data.num_classes)).to(device)
    print(f"ViTEncoder: {sum(p.numel() for p in classifier.parameters()):,} parameters")

    optimizer = torch.optim.Adam(classifier.parameters(), lr=hp.ENDTOEND_LR)
    train_accs, val_accs = student.train_loop(
        classifier, classify_data.train_loader, optimizer,
        nn.CrossEntropyLoss(), hp.ENDTOEND_EPOCHS, device,
        val_loader=classify_data.val_loader, tasklabel="End-to-end")

    torch.save(classifier.state_dict(), approaches['endtoend'].weights)
    np.save(approaches['endtoend'].curve_val, val_accs)
    np.save(approaches['endtoend'].curve_train, train_accs)
    print(f"Saved classifier -> {approaches['endtoend'].weights}")


# ============================================================================
# Task 2: Rotation prediction for ViT
# ============================================================================

def t2_rotation(rotation_data, classify_data, device, approaches):
    """Train ViT with rotation prediction, then probe on 15-scenes."""
    from torch.utils.data import DataLoader

    torch.manual_seed(student.BANNER_ID)

    # Phase 1: Rotation prediction
    model = student.ViTEncoder(nn.Linear(192, 4)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.ROTATION_LR)

    train_accs, _ = student.train_loop(
        model, rotation_data.train_loader, optimizer,
        nn.CrossEntropyLoss(), hp.ROTATION_EPOCHS, device,
        tasklabel="Rotation")

    torch.save(model.encoder.state_dict(), approaches['rotation'].weights)
    np.save(approaches['rotation'].curve_train, train_accs)
    print(f"Saved encoder -> {approaches['rotation'].weights}")

    # Phase 2: Frozen linear probe on 15-scenes
    probe = student.ViTEncoder(nn.Linear(192, classify_data.num_classes)).to(device)
    probe.encoder.load_state_dict(
        torch.load(approaches['rotation'].weights, weights_only=True, map_location='cpu'))
    for p in probe.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(probe.head.parameters(), lr=hp.TRANSFER_HEAD_LR)

    _, val_accs = student.train_loop(
        probe, classify_data.train_loader, optimizer,
        nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
        val_loader=classify_data.val_loader, tasklabel="Rotation-Probe")

    np.save(approaches['rotation'].curve_val, val_accs)
    print(f"Best rotation probe val: {max(val_accs):.3f}")


# ============================================================================
# Task 4: Transfer evaluation
# ============================================================================

def t4_transfer(classify_data, device, approaches, data_dir):
    """Run transfer probes and attention map comparison."""
    num_classes = classify_data.num_classes

    # --- 1. Frozen random probe (control) ---
    print("=== Frozen random probe (control) ===")
    model = student.ViTEncoder(nn.Linear(192, num_classes)).to(device)
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=hp.TRANSFER_HEAD_LR)
    train_accs, val_accs = student.train_loop(
        model, classify_data.train_loader, optimizer,
        nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
        val_loader=classify_data.val_loader, tasklabel="Frozen-Random")
    np.save(approaches['frozen_random'].curve_train, train_accs)
    np.save(approaches['frozen_random'].curve_val, val_accs)
    torch.save(model.state_dict(), approaches['frozen_random'].weights)

    # --- 2. Frozen rotation-pretrained probe ---
    print("\n=== Frozen rotation-pretrained probe ===")
    model = student.ViTEncoder(nn.Linear(192, num_classes)).to(device)
    model.encoder.load_state_dict(
        torch.load(approaches['rotation'].weights, weights_only=True, map_location='cpu'))
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=hp.TRANSFER_HEAD_LR)
    train_accs, val_accs = student.train_loop(
        model, classify_data.train_loader, optimizer,
        nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
        val_loader=classify_data.val_loader, tasklabel="Frozen-Rotation")
    np.save(approaches['frozen_rotation'].curve_train, train_accs)
    np.save(approaches['frozen_rotation'].curve_val, val_accs)
    torch.save(model.state_dict(), approaches['frozen_rotation'].weights)

    # --- 3. Frozen DINO-pretrained probe ---
    print("\n=== Frozen DINO-pretrained probe ===")
    model = student.ViTEncoder(nn.Linear(192, num_classes)).to(device)
    model.encoder.load_state_dict(
        torch.load(approaches['dino'].weights, weights_only=True, map_location='cpu'))
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=hp.TRANSFER_HEAD_LR)
    train_accs, val_accs = student.train_loop(
        model, classify_data.train_loader, optimizer,
        nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
        val_loader=classify_data.val_loader, tasklabel="Frozen-DINO")
    np.save(approaches['frozen_dino'].curve_train, train_accs)
    np.save(approaches['frozen_dino'].curve_val, val_accs)
    torch.save(model.state_dict(), approaches['frozen_dino'].weights)

    # --- 4. Finetune DINO ---
    print("\n=== Finetune DINO ===")
    model = student.ViTEncoder(nn.Linear(192, num_classes)).to(device)
    model.encoder.load_state_dict(
        torch.load(approaches['dino'].weights, weights_only=True, map_location='cpu'))
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': hp.TRANSFER_ENCODER_LR},
        {'params': model.head.parameters(), 'lr': hp.TRANSFER_HEAD_LR},
    ])
    train_accs, val_accs = student.train_loop(
        model, classify_data.train_loader, optimizer,
        nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
        val_loader=classify_data.val_loader, tasklabel="Finetune-DINO")
    np.save(approaches['finetune'].curve_train, train_accs)
    np.save(approaches['finetune'].curve_val, val_accs)
    torch.save(model.state_dict(), approaches['finetune'].weights)

    # --- 5. Frozen DINOv3 probe ---
    print("\n=== Frozen DINOv3 probe ===")
    dinov3_encoder, dinov3_dim = load_dinov3_encoder(device=device)
    model = student.ViTEncoder(nn.Linear(dinov3_dim, num_classes),
                               encoder=dinov3_encoder).to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=hp.TRANSFER_HEAD_LR)
    train_accs, val_accs = student.train_loop(
        model, classify_data.train_loader, optimizer,
        nn.CrossEntropyLoss(), hp.TRANSFER_EPOCHS, device,
        val_loader=classify_data.val_loader, tasklabel="Frozen-DINOv3")
    np.save(approaches['dinov3_probe'].curve_train, train_accs)
    np.save(approaches['dinov3_probe'].curve_val, val_accs)
    torch.save(model.head.state_dict(), approaches['dinov3_probe'].weights)

    # --- Attention map comparison on high-res test images ---
    print("\n=== Attention map comparison ===")
    to_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img_dir = os.path.join(data_dir, 'highres-images')
    sample_paths = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.endswith(('.jpg', '.png'))
    ]) if os.path.isdir(img_dir) else []

    # Load all encoders for comparison
    random_enc, _ = create_vit_tiny()
    random_enc.eval()
    rotation_enc, _ = create_vit_tiny()
    rotation_enc.load_state_dict(
        torch.load(approaches['rotation'].weights, weights_only=True, map_location='cpu'))
    rotation_enc.eval()
    dino_enc, _ = create_vit_tiny()
    dino_enc.load_state_dict(
        torch.load(approaches['dino'].weights, weights_only=True, map_location='cpu'))
    dino_enc.eval()

    for img_path in sample_paths[:3]:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        tensor = to_tensor(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)

        for label, enc in [('random', random_enc), ('rotation', rotation_enc),
                           ('dino', dino_enc), ('dinov3', dinov3_encoder)]:
            student.visualize_attention(
                enc, tensor, os.path.join('results', f'compare_{img_name}_fade_{label}.png'),
                style='fade', device=device)
            student.visualize_attention(
                enc, tensor, os.path.join('results', f'compare_{img_name}_gray_{label}.png'),
                style='gray', device=device)
