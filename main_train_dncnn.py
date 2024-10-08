import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# training code for DnCNN
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
#         https://github.com/cszn/DnCNN
#
# Reference:
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_dncnn.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter

    loss_type = opt['train']['G_lossfn_type'] # Tomer - for graphs
    print(f"Using loss function: {loss_type}")
    border = 0
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    dataset_type = opt['datasets']['train']['dataset_type']
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    mse = torch.nn.MSELoss().type(torch.cuda.FloatTensor)
    real_mses = []
    sure = []
    epochs = []
    bsd68_psnr = []
    bsd68_epochs = []
    for epoch in range(10000):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1

            if dataset_type == 'dnpatch' and current_step % 20000 == 0:  # for 'train400'
                train_loader.dataset.update_data()

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # merge bnorm
            # -------------------------------
            if opt['merge_bn'] and opt['merge_bn_startpoint'] == current_step:
                logger.info('^_^ -----merging bnorm----- ^_^')
                model.merge_bnorm_train()
                model.print_network()
            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                results = model.current_results()
                curr_mse = mse(results['E'], train_data['GT']).detach().cpu().numpy()
                real_mses.append(curr_mse)
                logs['MSE'] = curr_mse
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                    if k=="G_loss":
                        sure.append(v)
                logger.info(message)

            # real_mses.append(mse(visuals["E"], train_data["H"]))
            #     print(visuals['E'].min(), visuals['E'].max())
            #     print(train_data['H'].min(), train_data['H'].max())
            #     print(f"MSE: {mse(visuals['E'], train_data['H'])}")
            #     print("IN REAL MSE")
            #     print(results["E"].shape, train_data['H'].shape)
            #     print(mse(results['E'], train_data['H']).detach().cpu().numpy())
                epochs.append(epoch)
            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step, epoch)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:
                bsd68_epochs.append(epoch)
                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])
                    L_img = util.tensor2uint(visuals['L'])
                    GT_img = util.tensor2uint(visuals['GT'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    E_save_img_path = os.path.join(img_dir, 'E_{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, E_save_img_path)
                    H_save_img_path = os.path.join(img_dir, 'H_{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(H_img, H_save_img_path)
                    L_save_img_path = os.path.join(img_dir, 'L_{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(L_img, L_save_img_path)
                    GT_save_img_path = os.path.join(img_dir, 'GT_{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(GT_img, GT_save_img_path)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, GT_img, border=border)

                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx
                bsd68_psnr.append(avg_psnr)
                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))
    plt.figure()
    # Plot the first set of data
    # x = [i for i in range(len(real_mses))]
    # print(real_mses)
    # print(sure)
    plt.plot(epochs, real_mses, label='MSE')

    # Plot the second set of data
    plt.plot(epochs, sure, label=loss_type)
    # plt.plot(bsd68_epochs, bsd68_psnr, label='PSNR (BSD68)')

    # Add title and labels
    plt.title(f'{loss_type}/MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')

    # Add a legend
    plt.legend()
    # plt.ylim(0, 1)
    # Show the plot
    plt.savefig("/opt/KAIR/denoising/dncnn25/images/sure-mse.png")
    plt.close()

    # Plot PSNR
    plt.figure()
    plt.plot(bsd68_epochs, bsd68_psnr)
    plt.title('BSD68 PSNR Noise level 25')
    plt.xlabel('Epoch')
    plt.ylabel('dB')
    plt.savefig("/opt/KAIR/denoising/dncnn25/images/PSNR.png")
    plt.close()

    best_psnr = max(bsd68_psnr)
    best_psnr_idx = np.argmax(bsd68_psnr)
    best_psnr_epoch = bsd68_epochs[best_psnr_idx]
    logger.info(f"Best BSD68 PSNR: {best_psnr} at epoch: {best_psnr_epoch}")
    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
