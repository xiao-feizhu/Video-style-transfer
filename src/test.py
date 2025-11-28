import os
import argparse
from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import save_image

from sty_img import stylize
from sty_video import stylize_video
from utility import interpolate_weights

content_list = ['chicago', 'stata']
style_list = ['rain_princess', 'the_scream']
# content_list = ['avril']
# style_list = ['picasso_self_portrait', 'impronte_d_artista', 'trial', 'antimonocromatismo','rain_princess']

device = "cuda" if torch.cuda.is_available() else "cpu"

# AdaIN
# decoder = 'models/adain/decoder_epoch_14.pth'
# output_dir = 'result/stylized_images/adain'

# Optical flow
# decoder = f'models/adain/decoder_epoch_14.pth'
# output_dir = f'result/stylized_images/test/adain_epoch_14'  

# for i in range(1, 12):
#     decoder = f'models/adain/decoder_epoch_{i}.pth'
#     output_dir = f'result/stylized_images/test/adain_epoch_{i}'    

#     os.makedirs(output_dir, exist_ok=True)


#     #Test sty_img.py with various settings

#     for content_img in content_list:
#         for style_img in style_list:
#             content_path = os.path.join('data/test/content', content_img + '.jpg')
#             style_path = [os.path.join('data/test/style', style_img + '.jpg')]
#             output_path = os.path.join(
#                 output_dir,
#                 f"{style_img}_{content_img}.jpg"
#             )
#             stylize(content_path, style_path, decoder,
#                     output_path, alpha=1.0,
#                     preserve_color=False,
#                     size=512, device=device)

# Alpha

# content_path = os.path.join('data/test/content', 'chicago.jpg')
# style_path = [os.path.join('data/test/style', 'rain_princess.jpg')]

# for alpha in [1.25, 1.5]:
#     output_path = os.path.join(
#         output_dir,
#         f"alpha/alpha_{alpha}_rain_princess_chicago.jpg"
#     )
#     stylize(content_path, style_path, decoder,
#             output_path, alpha=alpha,
#             preserve_color=False,
#             size=512, device=device)
    
# Preserve color

# output_path = os.path.join(
#     output_dir,
#     f"preserve_color/style_only_rain_princess_chicago.jpg"
# )
# stylize(content_path, style_path, decoder,
#         output_path, alpha=1.0,
#         preserve_color=True,
#         size=512, device=device)

# Style interpolation

# content_path = os.path.join('data/test/content', 'avril.jpg')
# style_path = [
#     os.path.join('data/test/style', 'picasso_self_portrait.jpg'),
#     os.path.join('data/test/style', 'impronte_d_artista.jpg'),
#     os.path.join('data/test/style', 'trial.jpg'),
#     os.path.join('data/test/style', 'antimonocromatismo.jpg')
# ]
# W = interpolate_weights(n=5, device=device)  # (n*n, 4)
# results = []
# idx = 0

# for r in range(5):
#     row_imgs = []

#     for c in range(5):
#         w = W[r*5 + c]  # 4-style 权重
#         output_path = os.path.join(
#             output_dir, 
#             f"style_interpolation/weights_{w.tolist()}_avril.jpg"
#         )
#         output = stylize(
#             content_path, style_path,
#             decoder, output_path,
#             style_weights=w.tolist(),
#             alpha=1.0,
#             preserve_color=False,
#             size=512,
#             device=device
#         )  # (3,H,W)·
#         row_imgs.append(output)

#     # cat row horizontally
#     row_tensor = torch.cat(row_imgs, dim=0)
#     results.append(row_tensor)

#     # cat 5 行 vertically
#     final = torch.cat(results, dim=0)

# output_path = os.path.join(
#     output_dir,
#     f"style_interpolation/4interpolation_avril.jpg"
# )
# save_image(final, output_path, nrow=5)

# Test sty_video.py
factor = 0.2
model_name = 'flow'

input_video = 'data/test/fox.mp4'
output_dir = f'result/stylized_videos/{model_name}'
os.makedirs(output_dir, exist_ok=True)
decoder = f'models/{model_name}/decoder_epoch_8.pth'


for style_img in style_list:
    style_path = os.path.join('data/test/style', style_img + '.jpg')
    stylize_video(
        input_video=input_video,
        style_path=style_path,
        decoder_path=decoder,
        output_video=os.path.join(
            output_dir,
            f"{style_img}_fox_smooth{factor}.mp4"
        ),
        alpha=1.0,
        max_size=512,
        smooth=True,
        smooth_factor=factor,
        device=device
    )
    