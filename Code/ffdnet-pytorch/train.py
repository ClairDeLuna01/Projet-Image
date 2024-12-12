"""
Trains a FFDNet model

By default, the training starts with a learning rate equal to 1e-3 (--lr).
After the number of epochs surpasses the first milestone (--milestone), the
lr gets divided by 100. Up until this point, the orthogonalization technique
described in the FFDNet paper is performed (--no_orthog to set it off).

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils
from tensorboardX import SummaryWriter
from models import FFDNet
from dataset import Dataset, estimate_noise_with_ground_truth_batch
from utils import weights_init_kaiming, batch_psnr, init_logger, \
    svd_orthogonalization
from tqdm import tqdm
# from rich.progress import track

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    r"""Performs the main training loop
    """
    # Load dataset
    print('> Loading dataset ...')
    dataset_train = Dataset(train=True, gray_mode=args.gray, shuffle=True)
    dataset_val = Dataset(train=False, gray_mode=args.gray, shuffle=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0,
                              batch_size=args.batch_size, shuffle=True)
    print("\t# of training samples: %d\n" % int(len(dataset_train)))

    # Init loggers
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(args.log_dir)
    logger = init_logger(args)

    # Create model
    if not args.gray:
        in_ch = 3
    else:
        in_ch = 1
    net = FFDNet(num_input_channels=in_ch)
    # Initialize model with He init
    net.apply(weights_init_kaiming)
    # Define loss
    criterion = nn.MSELoss(reduction='mean')

    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Resume training or start anew
    if args.resume_training:
        resumef = os.path.join(args.log_dir, 'ckpt.pth')
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            print("> Resuming previous training")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_epoch = args.epochs
            new_milestone = args.milestone
            current_lr = args.lr
            args = checkpoint['args']
            training_params = checkpoint['training_params']
            start_epoch = training_params['start_epoch']
            args.epochs = new_epoch
            args.milestone = new_milestone
            args.lr = current_lr
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resumef, start_epoch))
            print("=> loaded parameters :")
            print("==> checkpoint['optimizer']['param_groups']")
            print("\t{}".format(checkpoint['optimizer']['param_groups']))
            print("==> checkpoint['training_params']")
            for k in checkpoint['training_params']:
                print("\t{}, {}".format(k, checkpoint['training_params'][k]))
            argpri = vars(checkpoint['args'])
            print("==> checkpoint['args']")
            for k in argpri:
                print("\t{}, {}".format(k, argpri[k]))

            args.resume_training = False
        else:
            raise Exception("Couldn't resume training with checkpoint {}".
                            format(resumef))
    else:
        start_epoch = 0
        training_params = {}
        training_params['step'] = 0
        training_params['current_lr'] = 0
        training_params['no_orthog'] = args.no_orthog

    # Training
    for epoch in range(start_epoch, args.epochs):
        # Learning rate value scheduling according to args.milestone
        if epoch > args.milestone[1]:
            current_lr = args.lr / 1000.
            training_params['no_orthog'] = True
        elif epoch > args.milestone[0]:
            current_lr = args.lr / 10.
        else:
            current_lr = args.lr

        # set learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # train
        for i, (data, orig) in enumerate(loader_train, 0):
            # Pre-training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # inputs: noise and noisy image

            # noise = torch.zeros(img_train.size())
            # stdn = np.random.uniform(args.noiseIntL[0], args.noiseIntL[1],
            #                          size=noise.size()[0])
            # for nx in range(noise.size()[0]):
            #     sizen = noise[0, :, :, :].size()
            #     noise[nx, :, :, :] = torch.FloatTensor(sizen).\
            #         normal_(mean=0, std=stdn[nx])
            # imgn_train = img_train + noise

            # Create input Variables
            data = Variable(data.cuda())
            orig = Variable(orig.cuda())

            noise, noiseSTD = estimate_noise_with_ground_truth_batch(
                data, orig
            )

            noiseSTD = Variable(noiseSTD.cuda())
            noise = Variable(noise.cuda())
            noiseNormalized = (noise + 1.0) / 2.0
            # noise = img_train - img_train_orig
            # stdn_var = Variable(torch.cuda.FloatTensor(stdn))

            # Evaluate model and optimize it
            # print("img_train.size(): {}".format(img_train.size()))
            # print("noise.size(): {}".format(noise.size()))
            out_train = model(data, noiseSTD)
            out_train_normalized = (out_train + 1.0) / 2.0
            loss = criterion(out_train_normalized, noiseNormalized)

            # debug print the max and min of out_train and of noise
            # print("out_train: max: {}, min: {}".format(
            #     out_train.max(), out_train.min()))
            # print("noise: max: {}, min: {}".format(noise.max(), noise.min()))
            # print("loss: {}".format(loss))

            # quit()

            loss.backward()
            optimizer.step()

            # print("loss: {}".format(loss))
            # print("loss.data: {}".format(loss.data))

            # PyTorch v0.4.0: loss.data[0] --> loss.item()

            # debug save patches to patches/
            # for i in range(img_train.size()[0]):
            #     utils.save_image(
            #         img_train[i], 'patches/{}_noisy.png'.format(i))
            #     utils.save_image(
            #         img_train_orig[i], 'patches/{}_orig.png'.format(i))
            #     utils.save_image(
            #         denoisedImage[i], 'patches/{}_denoised.png'.format(i))
            #     utils.save_image(
            #         (out_train[i] + 1.0) / 2.0, 'patches/{}_out.png'.format(i))
            #     utils.save_image(
            #         (noise[i] + 1.0) / 2.0, 'patches/{}_noise_std_{}.png'.format(i, noiseSTD[i]))

            # quit()

            if training_params['step'] % args.save_every == 0:
                loss_val = loss.item()

                # Results
                model.eval()
                denoisedImage = torch.clamp(
                    data-model(data, noiseSTD), 0., 1.)
                psnr_train, psnr_train_best, psnr_train_worst = batch_psnr(
                    denoisedImage, orig, 1.)
                psnr_orig, psnr_orig_best, psnr_orig_worst = batch_psnr(
                    data, orig, 1.)
                psnr_improv = ((psnr_train / psnr_orig) - 1) * 100.0

                # Apply regularization by orthogonalizing filters
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)

                # Log the scalar values
                writer.add_scalar(
                    'loss', loss_val, training_params['step'])
                writer.add_scalar('PSNR on training data', psnr_train,
                                  training_params['step'])
                writer.add_scalar('PSNR improvement on training data', psnr_improv,
                                  training_params['step'])
                writer.add_scalar('Best PSNR on training data', psnr_train_best if psnr_train_best != np.inf else 100,
                                  training_params['step'])
                writer.add_scalar('Worst PSNR on training data', psnr_train_worst,
                                  training_params['step'])

                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f PSNR_orig: %.4f PSNR_improv: % 5.2f%% train_best: %.4f train_worst: %.4f orig_best: %.4f orig_worst: %.4f"
                      %
                      (epoch+1, i+1, len(loader_train), loss.data, psnr_train, psnr_orig, psnr_improv, psnr_train_best, psnr_train_worst, psnr_orig_best, psnr_orig_worst))
            training_params['step'] += 1

        # The end of each epoch
        model.eval()

        # Validation
        psnr_val = 0
        psnr_orig = 0
        for valimg, valimg_orig in dataset_val:
            img_val = torch.unsqueeze(valimg, 0)
            valimg_orig = torch.unsqueeze(valimg_orig, 0)

            imgn_val = img_val
            valimg_orig, imgn_val = Variable(
                valimg_orig.cuda()), Variable(imgn_val.cuda())

            noise, noiseSTD = estimate_noise_with_ground_truth_batch(
                imgn_val, valimg_orig
            )

            noiseSTD = Variable(noiseSTD.cuda())
            noise = Variable(noise.cuda())
            # noise = valimg - valimg_orig

            out_val = torch.clamp(
                imgn_val-model(imgn_val, noiseSTD), 0., 1.)
            psnr_val += batch_psnr(out_val, valimg_orig, 1.)[0]
            psnr_orig += batch_psnr(imgn_val, valimg_orig, 1.)[0]

        psnr_val /= len(dataset_val)
        psnr_orig /= len(dataset_val)
        psnr_improv = ((psnr_val / psnr_orig) - 1) * 100.0
        print("\n[epoch %d] PSNR_val: %.4f PSNR_improv: %.2f%%" %
              (epoch+1, psnr_val, psnr_improv))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('Learning rate', current_lr, epoch)

        # Log val images
        try:
            if epoch == 0:
                # Log graph of the model
                writer.add_graph(model, (imgn_val,), )
                # Log validation images
                for idx in range(2):
                    imclean = utils.make_grid(valimg_orig.data[idx].clamp(0., 1.),
                                              nrow=2, normalize=False, scale_each=False)
                    imnsy = utils.make_grid(imgn_val.data[idx].clamp(0., 1.),
                                            nrow=2, normalize=False, scale_each=False)
                    writer.add_image(
                        'Clean validation image {}'.format(idx), imclean, epoch)
                    writer.add_image(
                        'Noisy validation image {}'.format(idx), imnsy, epoch)
            for idx in range(2):
                imrecons = utils.make_grid(out_val.data[idx].clamp(0., 1.),
                                           nrow=2, normalize=False, scale_each=False)
                writer.add_image('Reconstructed validation image {}'.format(idx),
                                 imrecons, epoch)
            # Log training images
            imclean = utils.make_grid(data.data, nrow=8, normalize=True,
                                      scale_each=True)
            writer.add_image('Training patches', imclean, epoch)

        except Exception as e:
            logger.error("Couldn't log results: {}".format(e))

        # save model and checkpoint
        training_params['start_epoch'] = epoch + 1
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'net.pth'))
        save_dict = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'training_params': training_params,
            'args': args
        }
        torch.save(save_dict, os.path.join(args.log_dir, 'ckpt.pth'))
        if epoch % args.save_every_epochs == 0:
            torch.save(save_dict, os.path.join(args.log_dir,
                                               'ckpt_e{}.pth'.format(epoch+1)))
        del save_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FFDNet")
    parser.add_argument("--gray", action='store_true',
                        help='train grayscale image denoising instead of RGB')

    parser.add_argument("--log_dir", type=str, default="logs",
                        help='path of log files')
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--epochs", "--e", type=int, default=80,
                        help="Number of total training epochs")
    parser.add_argument("--resume_training", "--r", action='store_true',
                        help="resume training from a previous checkpoint")
    parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60],
                        help="When to decay learning rate; should be lower than 'epochs'")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--no_orthog", action='store_true',
                        help="Don't perform orthogonalization as regularization")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Number of training steps to log psnr and perform \
						orthogonalization")
    parser.add_argument("--save_every_epochs", type=int, default=5,
                        help="Number of training epochs to save state")

    argspar = parser.parse_args()

    print("\n### Training FFDNet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(argspar)
