from fastai.dataloader import *
from fastai.dataset import *
from fastai.transforms import *
from fastai.models import *
from fastai.conv_learner import *

from fp16utils import *

DIR = Path('data/imagenet/')
TRAIN_CSV='train.csv'

from pathlib import Path

arch = resnet34
tfms = tfms_from_model(arch, 256, aug_tfms=transforms_side_on)
bs = 128

data = ImageClassifierData.from_csv(DIR, 'train1', DIR/TRAIN_CSV, tfms=tfms, bs=bs)

# m = ConvnetBuilder(resnet34, data.c, data.is_multi, data.is_reg)
# # models = ConvnetBuilder(resnet34, data.c, data.is_multi, data.is_reg)
# # m.model = network_to_half(m.model)
# learner = ConvLearner(data, m)

# learner.fit(0.5,1,cycle_len=1)

m16 = ConvnetBuilder(resnet34, data.c, data.is_multi, data.is_reg)
m16.model = network_to_half(m16.model)
learner16 = ConvLearner(data, m16)

learner16.fit(0.5,1,cycle_len=1)