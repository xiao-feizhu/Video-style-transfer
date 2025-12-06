# train.py
import sys
sys.path.append("core")

import os
import argparse
from typing import Tuple
import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import VGGEncoder, Decoder, AdaINStyleTransfer, calc_mean_std, adain
from data import ImageFolderDataset, VimeoPairDataset
from utility import normalize_for_vgg, warp_with_flow

import torchvision.models.optical_flow as OF

def compute_content_loss(g_feats_4, t_feats_4):
    return nn.functional.mse_loss(g_feats_4, t_feats_4)

def compute_style_loss(g_feats_list, s_feats_list):
    style_loss = 0.0
    for g, s in zip(g_feats_list, s_feats_list):
        gm, gs = calc_mean_std(g)
        sm, ss = calc_mean_std(s)
        style_loss += nn.functional.mse_loss(gm, sm) + nn.functional.mse_loss(gs, ss)
    return style_loss

def compute_temporal_loss(g_t, g_tp1, c_t, c_tp1, flownet):
    """
    g_t, g_tp1: stylized frames at t and t+1, [B,3,H,W]
    c_t, c_tp1: original content frames at t and t+1, [B,3,H,W]
    """
    with torch.no_grad():
        flow = flownet(c_t, c_tp1)[-1]  # flow from t -> t+1

    g_t_warp = warp_with_flow(g_t, flow)  # warp stylized t into t+1 coords
    temp_loss = torch.mean((g_t_warp - g_tp1) ** 2)
    return temp_loss

class WarmupCosineDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        # warmup
        if step < self.warmup_steps:
            return [
                base_lr * (step + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]

        # cosine
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine
            for base_lr in self.base_lrs
        ]

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    content_dataset = ImageFolderDataset(args.content, size=args.image_size*2)
    style_dataset = ImageFolderDataset(args.style, size=args.image_size*2)

    if args.flow:
        content_loader = DataLoader(
            VimeoPairDataset(args.content, list_file="data/video/tri_testlist.txt", resize=args.image_size*2),
            batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, drop_last=True
        ) 
    else:
        content_loader = DataLoader(
            content_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, drop_last=True
        )

    style_loader = DataLoader(
        style_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, drop_last=True
    )

    content_iter = iter(content_loader)
    style_iter = iter(style_loader)

    # Model
    encoder = VGGEncoder().to(device).eval()
    decoder = Decoder().to(device)
    # adain_net = AdaINStyleTransfer(encoder, decoder).to(device)

    # Only train decoder
    for p in encoder.parameters():
        p.requires_grad = False

    raft = OF.raft_small(pretrained=True).cuda().eval()

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = WarmupCosineDecay(optimizer, 
                                  warmup_steps=1*args.max_iter_per_epoch, 
                                  total_steps=args.max_steps)

    content_weight = args.content_weight
    style_weight = args.style_weight
    temporal_weight = args.temporal_weight

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "samples"), exist_ok=True)


    global_step = 0
    losses = []
    for epoch in range(5, args.epochs):
        if args.flow:
            pre_g = None

        for step in range(args.max_iter_per_epoch):
            try:
                if args.flow:
                    pre_c, content = next(content_iter)
                else:
                    content = next(content_iter)

            except StopIteration:
                content_iter = iter(content_loader)
                content = next(content_iter)

            try:
                style = next(style_iter)
            except StopIteration:
                style_iter = iter(style_loader)
                style = next(style_iter)

            content = content.to(device)
            style = style.to(device)

            # Normalize for VGG
            c_norm = normalize_for_vgg(content)
            s_norm = normalize_for_vgg(style)

            # Forward
            with torch.no_grad():
                c_feats = encoder(c_norm)
                s_feats = encoder(s_norm)
                c4 = c_feats[-1]
                s4 = s_feats[-1]
            #     t_feats_4 = adain_net.encoder  # dummy to satisfy type checker

            # # Actually use AdaIN here (without grad on encoder)
            # with torch.no_grad():
            #     t_feats_4 = adain_net.encoder  # no-op; keep for clarity
            #     t_feats_4 = adain_net.encoder  # we only need c4, s4
            # Compute AdaIN t directly:
            t_feats_4 = adain(c4, s4)

            # Style strength mixing
            t_feats_4 = args.alpha * t_feats_4 + (1.0 - args.alpha) * c4

            # Decode
            g = decoder(t_feats_4)
            g_norm = normalize_for_vgg(g)

            # Re-encode for losses
            g_feats = encoder(g_norm)
            # s_feats_list = s_feats  # already computed
            # print("g_feats:", g_feats[-1].shape)
            # print("t_feats_4:", t_feats_4.shape)

            # Content loss uses relu4_1
            content_loss = compute_content_loss(g_feats[-1], t_feats_4)

            # Style loss uses multiple layers
            style_loss = compute_style_loss(g_feats, s_feats)
            
            # Temporal loss
            if args.flow:
                if pre_g is None:
                    temp_loss = torch.tensor(0.0, device=device)
                else:
                    temp_loss = compute_temporal_loss(pre_g, g, pre_c, content, raft)
            
            # Total loss
            if args.flow:
                loss = content_weight * content_loss + style_weight * style_loss + temporal_weight * temp_loss
            else:
                loss = content_weight * content_loss + style_weight * style_loss

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()    

            pre_g = g.detach()

            global_step += 1

            if global_step % args.log_interval == 0:
                print(
                    f"Epoch {epoch+1}/{args.epochs} "
                    f"Step {step+1}/{args.max_iter_per_epoch} "
                    f"GlobalStep {global_step} "
                    f"Loss: {loss.item():.4f} "
                    f"Content: {content_weight * content_loss.item():.4f} "
                    f"Style: {style_weight * style_loss.item():.4f} "
                    f"Temporal: {temporal_weight * temp_loss.item():.4f} "
                    f"Total_loss: {np.mean(losses):.4f}"
                )

            if global_step % args.sample_interval == 0:
                # save a small grid: content / style / output
                with torch.no_grad():
                    grid = torch.cat([content, style, g], dim=0)
                    save_path = os.path.join(
                        args.save_dir, "samples",
                        f"step_{global_step:07d}.jpg"
                    )
                    save_image(grid, save_path, nrow=content.size(0))

            if global_step >= args.max_steps:
                break

        if global_step >= args.max_steps:
            break

        # Save checkpoint after each epoch
        ckpt_path = os.path.join(args.save_dir, f"decoder_epoch_{epoch+1}.pth")
        torch.save(decoder.state_dict(), ckpt_path)
        print(f"Saved decoder checkpoint to {ckpt_path}")

    # Save final decoder
    final_path = os.path.join(args.save_dir, "decoder_final.pth")
    torch.save(decoder.state_dict(), final_path)
    print(f"Training done. Final decoder saved to {final_path}")


def calc_adain_direct(c4, s4):
    # helper to keep code structure clean
    return adain(c4, s4)

import time 

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--content", type=str, required=True,
                        help="Path to content image folder")
    parser.add_argument("--style", type=str, required=True,
                        help="Path to style image folder")
    parser.add_argument("--save_dir", type=str, default="models/flow")

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max_steps", type=int, default=400000)
    parser.add_argument("--max_iter_per_epoch", type=int, default=10000)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--content_weight", type=float, default=1.0)
    parser.add_argument("--style_weight", type=float, default=10.0)
    parser.add_argument("--temporal_weight", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--flow", type=bool, default=False)

    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--sample_interval", type=int, default=1000)

    args = parser.parse_args()
    train(args)
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/3600:.2f} hours.")
