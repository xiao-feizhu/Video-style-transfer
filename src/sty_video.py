# stylize_video.py
import os
import argparse

import cv2
import torch
from torchvision import transforms
import torchvision.models.optical_flow as OF

from model import VGGEncoder, Decoder, adain
from utility import *

def load_style(style_path: str, size: int, device):
    from PIL import Image
    tfm = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    img = Image.open(style_path).convert("RGB")
    img = tfm(img).unsqueeze(0).to(device)  # (1,3,H,W)
    return img


def stylize_video(input_video: str,
                  style_path: str,
                  decoder_path: str,
                  output_video: str,
                  alpha: float = 1.0,
                  gamma: float = 1.0,
                  smooth: bool = False,
                  smooth_factor: float = 0.1,
                  max_size: int = 512,
                  device="cuda"):
    device = torch.device(device)

    encoder = VGGEncoder().to(device).eval()
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval()

    if smooth:
        raft = OF.raft_small(pretrained=True).cuda().eval()

    # Load style image once
    style = load_style(style_path, size=max_size, device=device)
    s_norm = normalize_for_vgg(style)
    with torch.no_grad():
        s_feats = encoder(s_norm)
        s4 = s_feats[-1]

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine resize ratio to fit into max_size
    scale = min(max_size / max(width, height), 1.0)
    out_w = int(width * scale)
    out_h = int(height * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
    out_writer = cv2.VideoWriter(output_video, fourcc, fps, (out_w, out_h))

    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((out_h, out_w))

    if smooth:
        pre_c = None
        pre_g = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # frame: HxWxC, BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = resize(
            transforms.functional.to_pil_image(frame_rgb)
        )
        frame_tensor = to_tensor(frame_pil).unsqueeze(0).to(device)  # [0,1]

        c_norm = normalize_for_vgg(frame_tensor)

        with torch.no_grad():
            c_feats = encoder(c_norm)
            c4 = c_feats[-1]
            t = adain(c4, s4)
            # local_alpha = compute_local_alpha(c4)  # (B,1,H,W)
            # final_alpha = combine_alpha(local_alpha, alpha, gamma)
            
            t = alpha * t + (1.0 - alpha) * c4
            # t = alpha * t + (1.0 - alpha) * c4
            g = decoder(t)
            g = torch.clamp(g, 0.0, 1.0)

        if smooth and pre_c is not None and pre_g is not None:
            with torch.no_grad():
                flow_fw = raft(pre_c, frame_tensor)[-1]
                flow_bw = raft(frame_tensor, pre_c)[-1]
                mask = occlusion_mask(flow_fw, flow_bw)

                # local_alpha = compute_local_alpha(frame_tensor) 
                flow_warped = warp_with_flow(pre_g, flow_fw)
                
                # W = smooth_factor
                W = mask * smooth_factor
                # W = local_alpha * mask * smooth_factor 
                
                g = (1.0 - W) * g + W * flow_warped
        
        # postprocess
        # g = postprocess_image(g)

        pre_c = frame_tensor.clone()
        pre_g = g.clone()

        # print(g.shape)
        out_frame = g.squeeze(0).cpu()
        out_frame = (out_frame.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        out_frame_bgr = cv2.resize(out_frame_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        # print(out_frame_bgr.shape)
        out_writer.write(out_frame_bgr)

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames")

    cap.release()
    out_writer.release()
    print(f"Saved stylized video to {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--decoder", type=str, required=True,
                        default="models/flow/decoder_epoch_8.pth")
    parser.add_argument("--output", type=str, default="result/stylized_videos/flow/stylized.mp4")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_size", type=int, default=512)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stylize_video(
        input_video=args.input,
        style_path=args.style,
        decoder_path=args.decoder,
        output_video=args.output,
        alpha=args.alpha,
        max_size=args.max_size,
        smooth=True,
        smooth_factor=0.1,
        device=device,
    )
