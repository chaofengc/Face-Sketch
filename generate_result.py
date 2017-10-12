import os
from glob import glob

test_path = './test/'
no_region_list = ['r1.png', 'r2.png', 'r3.png', 'r4.png', 'p1.png', 'p2.png']
for img_path in glob(os.path.join(test_path, '*.png')):
    photo_name = img_path.split('/')[-1]
    content_name = 'content_' + photo_name
    sketch_name = 'sketch_' + photo_name
    if photo_name in no_region_list:
        region_weight = 0.
    else:
        region_weight = 0.1
    os.system('THEANO_FLAGS=device=gpu3 python sketch_generate.py ./test/%s ./result/content/%s ./result/sketch/%s 1. 0.001 %f' % (photo_name, content_name, sketch_name, region_weight) )
