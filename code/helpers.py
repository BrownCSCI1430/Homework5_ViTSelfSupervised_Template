# Helper functions for HW5: Self-Supervised Learning with Transformers.
# You don't need to change anything here, but reading the code will help
# you understand how the ViT backbone and attention extraction work.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# ViT-Tiny Backbone
# ---------------------------------------------------------------------------
# We use timm (PyTorch Image Models) to create a ViT-Tiny backbone.
# ViT-Tiny: 192-dim embeddings, 3 attention heads, 12 transformer layers,
# 16x16 patches, ~5.5M parameters. Small enough to train on Oscar.

def create_vit_tiny(image_size=224, patch_size=16, pretrained=False):
    """Create a ViT-Tiny backbone using timm.

    Returns a model whose .forward_features(x) produces a tensor of shape
    (batch_size, num_tokens, 192), where token 0 is the [CLS] token.

    The model has no classification head — you attach your own.

    Parameters
    ----------
    image_size : int
        Input image resolution (must be divisible by patch_size).
    patch_size : int
        Patch size for tokenization.
    pretrained : bool
        If True, load ImageNet-pretrained weights.

    Returns
    -------
    model : nn.Module
        ViT-Tiny backbone. Call model.forward_features(x) to get token embeddings.
    embed_dim : int
        The embedding dimension (192 for ViT-Tiny).
    """
    import timm

    model = timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=pretrained,
        num_classes=0,           # No classification head
        img_size=image_size,
        dynamic_img_size=True,   # Accept any resolution (needed for DINO local crops)
    )
    embed_dim = model.embed_dim  # 192 for ViT-Tiny
    return model, embed_dim


# ---------------------------------------------------------------------------
# Attention Map Extraction
# ---------------------------------------------------------------------------

def get_attention_weights(model, image_tensor, device='cpu'):
    """Run a forward pass and capture the raw attention weight matrix
    from the last transformer layer (given, do not modify).

    How it works:
    A PyTorch 'forward hook' is a callback that runs every time a module's
    forward() is called. We register one on the last attention layer
    (model.blocks[-1].attn) to intercept its computation.

    Inside the hook, we recompute the attention weights from scratch:
      1. The attention module's .qkv layer projects input tokens into
         queries (Q), keys (K), and values (V) for each head.
      2. Attention weights = softmax(Q @ K^T / sqrt(head_dim)).
      3. We save these weights and detach them from the computation graph.

    We must recompute because timm's attention module does not store the
    raw weights — it applies them to V and returns the result directly.

    Parameters
    ----------
    model : nn.Module
        A timm ViT model (e.g., from create_vit_tiny()).
    image_tensor : torch.Tensor
        A single image tensor of shape (1, 3, H, W).
    device : str or torch.device

    Returns
    -------
    attention : torch.Tensor
        Shape (num_heads, num_tokens, num_tokens). The full attention
        matrix from the last transformer layer. Token 0 is the [class]
        token; the remaining tokens correspond to image patches in
        row-major order.
    """
    model = model.to(device).eval()

    # Storage for the hook to write into
    attn_storage = {}

    def hook(module, input, output):
        # input[0] has shape (B, num_tokens, embed_dim)
        B, N, C = input[0].shape

        # head_dim: standard timm ViT has module.head_dim;
        # EVA/DINOv3 models don't, so infer from num_heads.
        num_heads = module.num_heads
        head_dim = getattr(module, 'head_dim', C // num_heads)
        scale = getattr(module, 'scale', head_dim ** -0.5)

        # Project to queries, keys, values — all heads at once
        # qkv shape after reshape: (3, B, num_heads, N, head_dim)
        qkv = module.qkv(input[0]).reshape(
            B, N, 3, num_heads, head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention: softmax(Q K^T / sqrt(d_k))
        # Result shape: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1) * scale).softmax(dim=-1)
        attn_storage['attn'] = attn.detach()

    # Register hook on the last transformer block's attention module
    handle = model.blocks[-1].attn.register_forward_hook(hook)

    # Run forward pass — the hook fires and captures attention weights
    with torch.no_grad():
        model.forward_features(image_tensor.to(device))

    # Remove the hook (clean up)
    handle.remove()

    return attn_storage['attn'][0].cpu()  # (num_heads, N, N)


# NOTE: Attention visualization is implemented by students in student.py.
# See visualize_attention_fade() and visualize_attention_grayscale().


# ---------------------------------------------------------------------------
# DINOv3 Pretrained Features (adapted from HW2)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DINO Training Dashboard
# ---------------------------------------------------------------------------

class DINODashboard:
    """Live training dashboard for DINO. Call update() each epoch.

    Produces a multi-panel PNG showing training dynamics:
      - Loss curve
      - Teacher/student output entropy (collapse detection)
      - Center vector norm (centering health)
      - EMA momentum schedule
      - Per-head [CLS] attention maps on a fixed sample image
      - Student and teacher output distributions P(k) for a few images

    A numbered copy is saved every epoch to <save_dir>/dashboard/
    (dino_dashboard_ep000.png, ep001.png, ...) so you can flip through
    the progression; <save_dir>/dino_dashboard.png is always the latest.

    Usage in your training loop:
        dashboard = DINODashboard(save_dir='results', sample_image=some_tensor)
        for epoch in range(epochs):
            ... training ...
            dashboard.update(
                epoch=epoch,
                loss=avg_loss,
                student_out=student_outputs[0],   # (B, K) from this epoch
                teacher_out=teacher_outputs[0],    # (B, K) from this epoch
                center=center,                     # (K,) current center
                encoder=student_encoder,           # for attention maps
                ema_momentum=current_momentum,     # current lambda
            )
    """

    def __init__(self, save_dir='results', sample_image=None, device='cpu'):
        self.save_dir = save_dir
        self.sample_image = sample_image  # (1, 3, H, W) tensor for attention maps
        self.device = device
        self.epoch_dir = os.path.join(save_dir, 'dashboard')
        os.makedirs(self.epoch_dir, exist_ok=True)

        # History
        self.losses = []
        self.student_entropies = []
        self.teacher_entropies = []
        self.center_norms = []
        self.ema_momentums = []
        self.attn_snapshots = []  # list of (epoch, attention_maps)

        # Latest output distributions, (n, K) each
        self.student_probs = None
        self.teacher_probs = None

    def _entropy(self, logits, temp):
        """Compute mean entropy of softmax distribution."""
        probs = torch.softmax(logits / temp, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()
        return entropy

    def update(self, epoch, loss, student_out, teacher_out, center,
               encoder=None, ema_momentum=None,
               student_temp=None, teacher_temp=None,
               update_every=1, num_dist_samples=3):
        """Record metrics and regenerate dashboard.

        Parameters
        ----------
        epoch : int
        loss : float
        student_out : torch.Tensor, shape (B, K) — raw logits before softmax
        teacher_out : torch.Tensor, shape (B, K) — raw logits before softmax
        center : torch.Tensor, shape (K,)
        encoder : nn.Module or None — student encoder for attention maps
        ema_momentum : float or None — current EMA lambda
        student_temp : float or None — defaults to hp.DINO_STUDENT_TEMP
        teacher_temp : float or None — defaults to hp.DINO_TEACHER_TEMP
        update_every : int — save a dashboard PNG every N epochs
        num_dist_samples : int — number of images shown in the P(k) panels
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import hyperparameters as hp

        if student_temp is None:
            student_temp = hp.DINO_STUDENT_TEMP
        if teacher_temp is None:
            teacher_temp = hp.DINO_TEACHER_TEMP
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        with torch.no_grad():
            # Work on CPU copies so center/outputs can come from any device
            student_out = student_out.detach().float().cpu()
            teacher_out = teacher_out.detach().float().cpu()
            center = center.detach().float().cpu()

            self.losses.append(loss)
            self.student_entropies.append(self._entropy(student_out, student_temp))
            self.teacher_entropies.append(
                self._entropy(teacher_out - center.unsqueeze(0), teacher_temp))
            self.center_norms.append(center.norm().item())
            if ema_momentum is not None:
                self.ema_momentums.append(ema_momentum)

            # Output distributions P(k) for the first few images in the batch
            # (same softmax + temperature + centering as the DINO loss)
            n = min(num_dist_samples, student_out.shape[0])
            self.student_probs = torch.softmax(
                student_out[:n] / student_temp, dim=-1).numpy()
            self.teacher_probs = torch.softmax(
                (teacher_out[:n] - center.unsqueeze(0)) / teacher_temp,
                dim=-1).numpy()

            # Attention map snapshot (normalize for ViT forward pass)
            if encoder is not None and self.sample_image is not None:
                _norm = transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                _img_norm = _norm(self.sample_image[0]).unsqueeze(0)
                raw = get_attention_weights(encoder, _img_norm, self.device)
                num_prefix = getattr(encoder, 'num_prefix_tokens', 1)
                cls_attn = raw[:, 0, num_prefix:]
                h = w = int(cls_attn.shape[1] ** 0.5)
                attn = cls_attn.reshape(-1, h, w)
                self.attn_snapshots.append((epoch, attn))

        if epoch % update_every != 0 and epoch != 0:
            return

        # --- Build the dashboard ---
        #   Row 0: loss | entropy | ||center|| | EMA lambda
        #   Row 1: input image | attention head 0 | head 1 | ...  (if available)
        #   Row 2: student P(k) | teacher P(k)
        has_attn = self.sample_image is not None and len(self.attn_snapshots) > 0
        num_heads = self.attn_snapshots[-1][1].shape[0] if has_attn else 0
        ncols = max(4, num_heads + 1)
        nrows = 3 if has_attn else 2
        fig = plt.figure(figsize=(3.2 * ncols, 3.3 * nrows))
        gs = fig.add_gridspec(nrows, ncols)
        axes_flat = [fig.add_subplot(gs[0, c]) for c in range(4)]
        fig.patch.set_facecolor('white')
        epochs = list(range(len(self.losses)))

        # Panel 1: Loss
        ax = axes_flat[0]
        ax.plot(epochs, self.losses, color='#C62828', linewidth=1.5)
        ax.set_title('DINO Loss', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Panel 2: Entropy (collapse detection)
        ax = axes_flat[1]
        ax.plot(epochs, self.student_entropies, color='#1E88E5',
                linewidth=1.5, label='Student')
        ax.plot(epochs, self.teacher_entropies, color='#E65C00',
                linewidth=1.5, label='Teacher')
        ax.set_title('Output Entropy', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Collapse warning
        if len(self.teacher_entropies) > 3:
            recent_t = self.teacher_entropies[-1]
            if recent_t < 0.1:
                ax.set_facecolor('#FFF3E0')
                ax.text(0.5, 0.5, 'COLLAPSE?', transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='red',
                        alpha=0.4, fontweight='bold')

        # Panel 3: Center norm
        ax = axes_flat[2]
        ax.plot(epochs, self.center_norms, color='#7B1FA2', linewidth=1.5)
        ax.set_title('||center||', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Panel 4: EMA momentum
        ax = axes_flat[3]
        if self.ema_momentums:
            ax.plot(epochs[:len(self.ema_momentums)], self.ema_momentums,
                    color='#F57F17', linewidth=1.5)
        ax.set_title('EMA \u03bb', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylim(0.99, 1.001)
        ax.grid(True, alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Row 1: Input image + per-head attention (last snapshot)
        #   Each head is rescaled to [0, 1] for display; the title shows the
        #   raw range (max - min) so nearly-uniform heads can be spotted.
        #   For reference, uniform attention over 196 patches is ~0.005.
        if has_attn:
            ep, attn = self.attn_snapshots[-1]
            ax = fig.add_subplot(gs[1, 0])
            img = self.sample_image[0].permute(1, 2, 0).cpu().numpy()
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title('Input', fontsize=11, fontweight='bold')
            ax.axis('off')
            for i in range(num_heads):
                a = attn[i].numpy()
                a_range = a.max() - a.min()
                a = (a - a.min()) / (a_range + 1e-8)
                ax = fig.add_subplot(gs[1, i + 1])
                ax.imshow(a, cmap='gray', vmin=0, vmax=1,
                          interpolation='nearest')
                ax.set_title(f'Head {i} (range {a_range:.3f})',
                             fontsize=10, fontweight='bold')
                ax.axis('off')

        # Last row: output distributions P(k), student vs teacher
        #   Same images (first few of the last batch, first global crop) in
        #   both panels. The batch is shuffled, so the images change per epoch.
        if self.student_probs is not None:
            r = nrows - 1
            half = ncols // 2
            K = self.student_probs.shape[1]
            x = np.arange(K)
            ymax = max(self.student_probs.max(), self.teacher_probs.max())
            colors = ['#E53935', '#1E88E5', '#43A047', '#8E24AA', '#FB8C00']
            panels = [
                (gs[r, :half], self.student_probs,
                 f'Student P(k) = softmax(s / {self.student_temp:g})'),
                (gs[r, half:], self.teacher_probs,
                 f'Teacher P(k) = softmax((t - c) / {self.teacher_temp:g})'),
            ]
            for cell, probs, title in panels:
                ax = fig.add_subplot(cell)
                for j in range(probs.shape[0]):
                    ax.bar(x, probs[j], width=1.0, alpha=0.5,
                           color=colors[j % len(colors)], label=f'Image {j}')
                ax.axhline(1.0 / K, color='gray', linestyle='--',
                           linewidth=1, label='Uniform (1/K)')
                ax.set_xlim(-0.5, K - 0.5)
                ax.set_ylim(0, ymax * 1.1 + 1e-8)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xlabel('Output dimension k')
                ax.grid(True, alpha=0.15)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            ax.legend(fontsize=8, loc='upper right')

        fig.suptitle(f'DINO Training Dashboard — Epoch {epoch}',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(self.epoch_dir,
                                 f'dino_dashboard_ep{epoch:03d}.png'),
                    dpi=100, bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(self.save_dir, 'dino_dashboard.png'),
                    dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def save_attention_evolution(self, filename='attention_evolution.png'):
        """Save a grid of per-head attention maps at different training epochs.

        Rows are attention heads, columns are epochs.
        """
        if not self.attn_snapshots:
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n = len(self.attn_snapshots)
        n_show = min(n, 8)
        indices = np.linspace(0, n - 1, n_show, dtype=int)
        num_heads = self.attn_snapshots[0][1].shape[0]

        fig, axes = plt.subplots(num_heads, n_show,
                                 figsize=(2.2 * n_show, 2.2 * num_heads),
                                 squeeze=False)

        for col, idx in enumerate(indices):
            ep, attn = self.attn_snapshots[idx]
            for head in range(num_heads):
                a = attn[head].numpy()
                a = (a - a.min()) / (a.max() - a.min() + 1e-8)
                ax = axes[head, col]
                ax.imshow(a, cmap='gray', vmin=0, vmax=1,
                          interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                if head == 0:
                    ax.set_title(f'Epoch {ep}', fontsize=10)
                if col == 0:
                    ax.set_ylabel(f'Head {head}', fontsize=10)

        fig.suptitle('[CLS] Attention Evolution (per head)',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, filename),
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)


_DINOV3_MODEL_CACHE = None


def load_dinov3_encoder(device='cpu'):
    """Load pretrained DINOv3 ViT-Small encoder via timm.

    This is the same model used in HW2 for feature matching.
    ViT-Small/16 with 384-dim embeddings, trained on LVD-1689M.

    Returns
    -------
    model : nn.Module
        Frozen DINOv3 encoder. Use model.forward_features(x) to get
        token embeddings, then take token 0 ([CLS]) as the image embedding.
    embed_dim : int
        The embedding dimension (384 for ViT-Small).
    """
    import timm

    global _DINOV3_MODEL_CACHE
    if _DINOV3_MODEL_CACHE is None:
        print("Downloading DINOv3 model (first time only, ~80 MB)...")
        model = timm.create_model(
            'vit_small_patch16_dinov3_qkvb.lvd1689m',
            pretrained=True,
            num_classes=0,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _DINOV3_MODEL_CACHE = model

    model = _DINOV3_MODEL_CACHE.to(device)
    embed_dim = model.embed_dim  # 384
    return model, embed_dim
