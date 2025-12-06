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
# decoder = 'models/adain/large/decoder_epoch_16.pth'
# output_dir = 'result/stylized_images/adain/large_alpha'

# Optical flow
# decoder = f'models/adain/decoder_epoch_14.pth'
# output_dir = f'result/stylized_images/test/adain_epoch_14'  

# for i in range(1, 12):
#     decoder = f'models/adain/decoder_epoch_{i}.pth'
#     output_dir = f'result/stylized_images/test/adain_epoch_{i}'    



#### Start here
# def gen_img(content_list, style_list, output_dir, decoder, device, preserve_color=False, local_alpha=False):
#     for content_img in content_list:
#         for style_img in style_list:
#             content_path = os.path.join('data/test/content', content_img + '.jpg')
#             style_path = [os.path.join('data/test/style', style_img + '.jpg')]
#             output_path = os.path.join(
#                 output_dir,
#                 f"{style_img}_{content_img}.jpg"
#             )
#             stylize(content_path, style_path, decoder,
#                     output_path, global_alpha=1.0, local_alpha=local_alpha,
#                     preserve_color=preserve_color,
#                     size=512, device=device)
            
# decoder = 'models/adain/large/decoder_epoch_16.pth'
# output_dir = 'result/stylized_images/adain/large/postprocess'

# content_list = ['newyork', 'chicago']
# style_list = ['asheville', 'woman_with_hat_matisse','impronte_d_artista','rain_princess']
# gen_img(content_list, style_list, output_dir, decoder, device, preserve_color=False, local_alpha=False)  




# decoder = 'models/adain/large/decoder_epoch_16.pth'
# output_dir = 'result/stylized_images/adain/large/mix'
# os.makedirs(output_dir, exist_ok=True)
# gen_img(content_list, style_list, output_dir, decoder, device, preserve_color=False, local_alpha=False)

# output_dir = 'result/stylized_images/adain/large_alpha/mix'
# gen_img(content_list, style_list, output_dir, decoder, device, preserve_color=False, local_alpha=True)

# Test sty_img.py with various settings
# for content_img in content_list:
#     for style_img in style_list:
#         content_path = os.path.join('data/test/content', content_img + '.jpg')
#         style_path = [os.path.join('data/test/style', style_img + '.jpg')]
#         output_path = os.path.join(
#             output_dir,
#             f"test/{style_img}_{content_img}.png"
#         )
#         stylize(content_path, style_path, decoder,
#                 output_path, global_alpha=1.0,
#                 preserve_color=False,
#                 size=512, device=device)

# Alpha

# content_path = os.path.join('data/test/content', 'chicago.jpg')
# style_path = [os.path.join('data/test/style', 'rain_princess.jpg')]

# for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
#     output_path = os.path.join(
#         output_dir,
#         f"alpha/alpha_{alpha}_rain_princess_chicago.png"
#     )
#     stylize(content_path, style_path, decoder,
#             output_path, global_alpha=alpha,
#             preserve_color=False,
#             size=512, device=device)
    
# Preserve color
# for content_img in content_list:
#     for style_img in style_list:
#         content_path = os.path.join('data/test/content', content_img + '.jpg')
#         style_path = [os.path.join('data/test/style', style_img + '.jpg')]
#         output_path = os.path.join(
#             output_dir,
#             f"preserve_color/{style_img}_{content_img}.png"
#         )
#         stylize(content_path, style_path, decoder,
#                 output_path, global_alpha=1.0,
#                 preserve_color=True,
#                 size=512, device=device)

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
#             f"style_interpolation/weights_{w.tolist()}_avril.png"
#         )
#         output = stylize(
#             content_path, style_path,
#             decoder, output_path,
#             style_weights=w.tolist(),
#             global_alpha=1.0,
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
#     f"style_interpolation/4interpolation_avril.png"
# )
# save_image(final, output_path, nrow=5)

# Test sty_video.py
import time
# style_list = ['mondrian.jpg', 'minotaur.jpg']
# style_list = ['wave.jpg', 'la_muse.jpg', 'rain_princess.jpg']
style_list = ['mondrian.jpg', 'rain_princess.jpg']
start_time = time.time()
for factor in [0.2]:
    models = ['flow/ft_4']
    model_paths = ['models/flow/ft_4/decoder_final.pth']
    video = 'giraff'
    input_video = f'data/test/{video}.mp4'
    for model, decoder in zip(models, model_paths):
        output_dir = f'result/stylized_videos/{model}/mask'
        os.makedirs(output_dir, exist_ok=True)
        for style_img in style_list:
            style_path = os.path.join('data/test/style', style_img)
            stylize_video(
                input_video=input_video,
                style_path=style_path,
                decoder_path=decoder,
                output_video=os.path.join(
                    output_dir,
                    f"{style_img.split('.')[0]}_{video}_1024_smooth{factor}.mp4"
                ),
                alpha=0.8,
                max_size=1024,
                smooth=True,
                smooth_factor=factor,
                device=device
            )
print(time.time() - start_time)
# model = 'flow/ft_2'
# input_video = 'data/test/fox.mp4'
# output_dir = f'result/stylized_videos/{model}/test'
# factor = 0.2
# decoder = 'models/flow/ft_2/decoder_final.pth'
# style_list = ['rain_princess', 'la_muse']

# os.makedirs(output_dir, exist_ok=True)
# for style_img in style_list:
#     style_path = os.path.join('data/test/style', style_img + '.jpg')
#     stylize_video(
#         input_video=input_video,
#         style_path=style_path,
#         decoder_path=decoder,
#         output_video=os.path.join(
#             output_dir,
#             f"{style_img}_fox_smooth{factor}.mp4"
#         ),
#         alpha=1.0,
#         max_size=512,
#         smooth=True,
#         smooth_factor=factor,
#         device=device
#     )
