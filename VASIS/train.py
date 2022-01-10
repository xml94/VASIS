"""
Copyright (C) University of Science and Technology of China.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from data.base_dataset import repair_data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
from options.test_options import TestOptions
import tqdm
from util import html
from util.util import tensor2im, tensor2label
import torch
import copy


# CUDA_LAUNCH_BLOCKING=1
# parse options
opt = TrainOptions().parse()

if opt.env == 'horovod':
    import horovod.torch as hvd
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    print(f'Checking size of hvd: {hvd.size()}')

# print options to help debugging
print(' '.join(sys.argv))
command_path = os.path.join(opt.results_dir, opt.name)
os.makedirs(command_path, exist_ok=True)
with open(command_path + '/command.txt', 'w+') as file:
    file.writelines(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

val_opt = copy.deepcopy(opt)
if opt.train_eval:
    val_opt.no_flip = True
    val_opt.phase = 'test'
    val_opt.serial_batches = True
    # val_opt.batchSize = 4  # should be n * number of gpus
    assert val_opt.batchSize % len(val_opt.gpu_ids) == 0
    val_opt.isTrain = False
    dataloader_val = data.create_dataloader(val_opt)
    val_visualizer = Visualizer(val_opt)
    # # create a webpage that summarizes the all results
    web_dir = os.path.join(val_opt.results_dir, val_opt.name,
                           'test_%s' % (val_opt.which_epoch))
    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = test, Epoch = %s' %
                        (val_opt.name, val_opt.which_epoch))

    # process for calculate FID scores
    from inception import InceptionV3
    from fid_score import *
    import pathlib
    # define the inceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[opt.eval_dims]
    eval_model = InceptionV3([block_idx]).cuda()
    # load real images distributions on the training set
    mu_np_root = os.path.join('datasets/train_mu_si', val_opt.dataset_mode,'m.npy')
    st_np_root = os.path.join('datasets/train_mu_si', val_opt.dataset_mode,'s.npy')
    m0, s0 = np.load(mu_np_root), np.load(st_np_root)
    # load previous best FID
    if val_opt.continue_train:
        fid_record_dir = os.path.join(val_opt.checkpoints_dir, val_opt.name, 'fid.txt')
        FID_score, _ = np.loadtxt(fid_record_dir, delimiter=',', dtype=float)
    else:
        FID_score = 1000
else:
    FID_score = 1000

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)
        # train discriminator
        trainer.run_discriminator_one_step(data_i)
        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            if opt.train_eval:
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter, FID_score)
            else:
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter(FID_score)

    if epoch % val_opt.eval_epoch_freq == 0 and val_opt.train_eval and epoch > 50:
        trainer.pix2pix_model.eval()
        # generate fake image
        print('Checking: start evaluation .... ')
        if val_opt.use_vae:
            flag = True
            val_opt.use_vae = False
        else:
            flag = False
        for i, data_i in enumerate(dataloader_val):
            if data_i['label'].size()[0] != val_opt.batchSize:
                if val_opt.batchSize > 2*data_i['label'].size()[0]:
                    print('batch size is too large')
                    break
                data_i = repair_data(data_i, val_opt.batchSize)
            generated = trainer.pix2pix_model(data_i, mode='inference')
            img_path = data_i['path']
            for b in range(generated.shape[0]):
                # tmp = tensor2im(generated[b])
                visuals = OrderedDict([('input_label', data_i['label'][b]),
                                       ('synthesized_image', generated[b])])
                val_visualizer.save_images(webpage, visuals, img_path[b:b + 1])
        webpage.save()

        if flag:
            val_opt.use_vae = True
        # cal fid score
        fake_path = pathlib.Path(os.path.join(web_dir, 'images/synthesized_image/'))
        files = list(fake_path.glob('*.jpg')) + list(fake_path.glob('*.png'))
        m1, s1 = calculate_activation_statistics(files, eval_model, 1, val_opt.eval_dims, True, images=None)
        fid_value = calculate_frechet_distance(m0, s0, m1, s1)
        visualizer.print_eval_fids(epoch, fid_value, FID_score)

        # save the fid
        file_name = os.path.join(val_opt.checkpoints_dir, val_opt.name, 'fid_history.txt')
        with open(file_name, 'a') as file:
            file.writelines(str(fid_value) + ',\n')

        # save the best model if necessary
        if fid_value < FID_score:
            FID_score = fid_value
            trainer.save('best')

    if epoch % val_opt.save_epoch_freq == 0 or \
            epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()
    trainer.pix2pix_model.train()


print('Training was successfully finished.')
