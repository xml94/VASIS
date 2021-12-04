"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import torch.nn as nn
from collections import OrderedDict
from models.networks.sync_batchnorm import DataParallelWithCallback

import data
from options.test_options import TestOptions
from options.train_options import TrainOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from trainers.pix2pix_trainer import Pix2PixTrainer

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

trainer = Pix2PixTrainer(opt)
trainer.pix2pix_model.eval()

# model = Pix2PixModel(opt)
# if len(opt.gpu_ids) > 1:
    # model = nn.DataParallel(model, device_ids=opt.gpu_ids)
# model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       'test_%s' % (opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = test, Epoch = %s' %
                    (opt.name, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    # generated = model(data_i, mode='inference')
    generated = trainer.pix2pix_model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
