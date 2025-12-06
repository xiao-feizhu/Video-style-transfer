Environment Dependencies:
```
Numpy<=1.26
PyTorch, torchvision (Up-to-cuda)
matplotlib
opencv-python
```

Download COCO dataset as content data
```
wget http://images.cocodataset.org/zips/train2014.zip (Large)
wget http://images.cocodataset.org/zips/val2017.zip (Small)
```

Download style data from Wikiart
```
wget https://www.kaggle.com/api/v1/datasets/download/sivarazadi/wikiart-art-movementsstyles
```

Download video data from Vimeo90K(Only Using Testing Set)
```
wget http://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip
```