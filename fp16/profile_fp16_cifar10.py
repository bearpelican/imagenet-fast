from time import time
from fastai.conv_learner import *
from fp16utils import *
from fastai.models.cifar10.resnext import resnext29_8_64

def run_model(fp16=True):
    PATH = "data/cifar10/"
    print(f'Testing of fp16: {fp16}')

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))

    def get_data(sz,bs):
        tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)
        return ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)

    bs=128


    # Measure half precision for 8x8
    m = resnext29_8_64()
    if fp16:
        m = network_to_half(m)
    bm = BasicModel(m.cuda(), name='cifar10_resnet50')

    data = get_data(8,bs*4*4)

    learn = ConvLearner(data, bm)
    learn.unfreeze()

    lr=4e-2; wd=5e-4

    t1 = time.time()
    learn.fit(lr, 1, cycle_len=1, use_clr=(20,8))
    t2 = time.time()
    elapsed = t2 - t1
    print(f'Time elapsed: {elapsed}')

    # Measure half precision on 32x32
    m = resnext29_8_64()
    if fp16:
        m = network_to_half(m)
    bm = BasicModel(m.cuda(), name='cifar10_resnet50')

    data = get_data(32,bs*4)

    t1_32 = time.time()
    learn.fit(lr, 1, cycle_len=1, use_clr=(20,8))
    t2_32 = time.time()
    elapsed_32 = t2_32 - t1_32
    print(f'Time elapsed: {elapsed_32}')

import fire

if __name__ == '__main__':
  fire.Fire(run_model)
