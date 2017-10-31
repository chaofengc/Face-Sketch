import os
from glob import glob

os.environ['KERAS_BACKEND'] = 'theano'

#  test_path = './Data/AR_test/photos'
#  save_path = './result_CUHK'

test_path = './Data/CUHK_student_test/photos'
save_path = './result_CUHK'

no_region_list = ['r1.png', 'r2.png', 'r3.png', 'r4.png', 'p1.png', 'p2.png']
for img_path in glob(os.path.join(test_path, '*.png')):
    photo_name = img_path.split('/')[-1]
    content_name = photo_name
    sketch_name = photo_name
    if photo_name in no_region_list:
        region_weight = 0.
    else:
        region_weight = 0.1
    os.system('THEANO_FLAGS=device=gpu0 python sketch_generate.py %s/%s %s/content/%s %s/sketch/%s 1. 0.001 %f' % (test_path, photo_name, save_path, content_name, save_path, sketch_name, region_weight) )
