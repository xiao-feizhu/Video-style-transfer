# stylize.py
import os
import argparse
from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import save_image

from model import VGGEncoder, Decoder, adain
from utility import normalize_for_vgg, denorm_from_vgg, style_interpolation, match_color, compute_local_alpha, combine_alpha

def load_image(path: str, size: int = 512, device="cpu"):
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    img = tfm(img).unsqueeze(0).to(device)  # (1,3,H,W)
    return img


def stylize(content_path: str, style_path: str,
            decoder_path: str, output_path: str,
            style_weights=None,preserve_color=False,
            alpha: float = 1.0, size: int = 512, device="cpu"):

    encoder = VGGEncoder().to(device).eval()
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval()

    # content = load_image(content_path, size=size, device=device)
    # style = load_image(style_path, size=size, device=device)

    # c_norm = normalize_for_vgg(content)
    # s_norm = normalize_for_vgg(style)

    # with torch.no_grad():
    #     c_feats = encoder(c_norm)
    #     s_feats = encoder(s_norm)
    #     c4 = c_feats[-1]
    #     s4 = s_feats[-1]

    #     t = adain(c4, s4)
    #     t = alpha * t + (1.0 - alpha) * c4
    #     g = decoder(t)
    #     # g is already in [0,1] range because we trained like this
    #     out = torch.clamp(g, 0.0, 1.0)
        # Load images
    content_img = load_image(content_path, size, device)
    content_feat = encoder(normalize_for_vgg(content_img))

    # Load style images
    style_imgs = [load_image(p, size, device) for p in style_path]

    # if isinstance(style_path, str):
    #     style_imgs = [load_image(style_path, size, device)]
    # else:
    #     style_imgs = [load_image(p, size, device) for p in style_path]

    # If style interpolation
    if len(style_imgs) > 1:
        assert style_weights is not None
        # t = style_interpolation(encoder, content_feat, style_imgs, style_weights)
        t = style_interpolation(content_feat[-1],
                                [encoder(normalize_for_vgg(img))[-1] for img in style_imgs],
                                style_weights)
    else:
        style_feat = encoder(normalize_for_vgg(style_imgs[0]))
        t = adain(content_feat[-1], style_feat[-1])

    # Content-style tradeoff (α)
    local_alpha = compute_loacl_alpha(c4)  # (B,1,H,W)
    final_alpha = combine_alpha(local_alpha, args.global_alpha, args.gamma)
    t_feats_4 = final_alpha * t_feats_4 + (1.0 - final_alpha) * c4

    # Decode
    with torch.no_grad():
        g = decoder(t)
        g = torch.clamp(g, 0, 1)

    # Transfer style but keep color
    if preserve_color:
        g = match_color(content_img, g)


    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_image(g, output_path)
    print(f"Saved stylized image to {output_path}")

    return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style", type=list, required=True)
    parser.add_argument("--decoder", type=str, required=True,
                        default="models/adain/decoder_epoch_14.pth",
                        help="Path to trained decoder .pth")
    parser.add_argument("--output", type=str, default="result/output.jpg")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--global_alpha", type=float, default=1.0,
                        help="Global alpha for combining with local alpha")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma for global alpha adjustment")
    parser.add_argument("--preserve_color", action="store_true", default=False)
    parser.add_argument("--style_weights", nargs="+", type=float, default=None)
    parser.add_argument("--size", type=int, default=512)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stylize(args.content, args.style, args.decoder,
            args.output, alpha=args.alpha,
            style_weights=args.style_weights, preserve_color=args.preserve_color,
            size=args.size, device=device)
