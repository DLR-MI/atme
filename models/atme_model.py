import torch
from .base_model import BaseModel
from . import networks
from util.image_pool import DiscPool
from itertools import chain

class AtmeModel(BaseModel):
    """ This class implements the ATME model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet_256_attn' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    atme paper: https://arxiv.org/pdf/x.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For atme, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with instance norm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='instance', netG='unet_256_ddm', netD='basic', dataset_mode='aligned')
        parser.add_argument('--mask_size', type=int, default=256)
        parser.add_argument('--dim', type=int, default=64, help='dim for the ddm UNet')
        parser.add_argument('--dim_mults', type=tuple, default=(1,2,4,8), help='dim_mults for the ddm UNet')
        parser.add_argument('--groups', type=int, default=8, help='number of groups for GroupNorm within ResnetBlocks')
        parser.add_argument('--init_dim', type=int, default=64, help='output channels after initial conv2d of x_t')
        parser.add_argument('--learned_sinusoidal_cond', type=bool, default=False, help='learn fourier features for positional embedding?')
        parser.add_argument('--random_fourier_features', type=bool, default=False, help='random fourier features for positional embedding?')
        parser.add_argument('--learned_sinusoidal_dim', type=int, default=16, help='twice the number of fourier frequencies to learn')
        parser.add_argument('--time_dim_mult', type=int, default=4, help='dim * time_dim_mult amounts to output channels after time-MLP')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['Disc_B', 'real_A', 'noisy_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'W']
        else:  # during test time, only load G
            self.model_names = ['G', 'W'] 
        # define networks (both generator and discriminator)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 
                                      **{'dim': opt.dim, 
                                         'dim_mults': opt.dim_mults, 
                                         'init_dim': opt.init_dim, 
                                         'resnet_block_groups': opt.groups,
                                         'learned_sinusoidal_cond': opt.learned_sinusoidal_cond,
                                         'learned_sinusoidal_dim': opt.learned_sinusoidal_dim,
                                         'random_fourier_features': opt.random_fourier_features, 
                                         'time_dim_mult': opt.time_dim_mult})

        self.netW = networks.define_W(opt.init_type, opt.init_gain, self.gpu_ids)
        self.disc_pool = DiscPool(opt, self.gpu_ids[0], isTrain=self.isTrain)
    
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(chain(self.netW.parameters(), self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.batch_indices = input['batch_indices']
        self.disc_B = self.disc_pool.query(self.batch_indices)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.Disc_B = self.netW(self.disc_B)
        self.noisy_A = self.real_A * (1 + self.Disc_B)
        self.fake_B = self.netG(self.noisy_A, self.Disc_B)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        self.disc_B = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(self.disc_B, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients 
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        # Save discriminator output
        self.disc_pool.insert(self.disc_B.detach(), self.batch_indices)