import torch
from .base_model import BaseModel
from . import networks

"""
EPEW model means EPE model add weighted loss on different layer

desing -> |gan-opc| -> mask -> |lithogan| -> wafer
(l2 loss)
desing <- |gan-opc| <- mask <- |lithogan| <- wafer

to get smaller epenum in mask
"""

class GAN2GANWModel(BaseModel):
    """ This class implements the EPE model, for learning a mapping from input images to output images given paired data.
    Learning the mapping from design and imagine a middle output, than make the middle output into the lithogan before.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG dcupp' U++ by dcupp generator
    a '--netD naive6' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.add_argument('--lambda_tanh_scale', type=float, default=1.0, help='scale factor for tanh scale')
        parser.add_argument('--lambda_uppscale', type=int, default=2, help='scale factor for uppscale')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=200.0, help='weight for L1 loss')
            parser.add_argument('--lambda_EPE_L1', type=float, default=30.0, help='weight for EPE L1 loss')
            # parser.add_argument('--lambda_OPC_L1', type=float, default=200, help='weight for opc l1 loss')
            parser.add_argument('--lambda_R', type=float, default=100.0, help='weight for opc red layer l1loss')
            parser.add_argument('--lambda_G', type=float, default=10.0, help='weight for opc green layer l1loss')
            parser.add_argument('--lambda_B', type=float, default=100.0, help='weight for opc blue layer l1loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G0_GAN', 'G_L1', 'OPC_L1', 'EPE_L1', 'D_real', 'D_fake', 'D0_real', 'D0_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'opc_A', 'real_opc','fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G0', 'G', 'D0', 'D']
        else:  # during test time, only load G
            self.model_names = ['G0', 'G']

        # define networks of stage1, only the generator.
        self.netG0 = networks.define_G0(opt.input_nc, opt.output_nc, opt.ngf, opt.netG0, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.lambda_tanh_scale,
                                      lambda_uppscale=opt.lambda_uppscale)
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      lambda_uppscale=opt.lambda_uppscale)
        if opt.netG_pretrained_path is not '':
            self.netG.module.load_state_dict(torch.load(opt.netG_pretrained_path))

        if opt.netG0_pretrained_path is not '':
            self.netG0.module.load_state_dict(torch.load(opt.netG0_pretrained_path))
        self.set_requires_grad(self.netG, False)  # G requires no gradients when optimizing G0
        self.netG.eval()

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD0 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G0 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG0.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D0 = torch.optim.Adam(self.netD0.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G0)
            # self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D0)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_opc = input['C'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.opc_A = self.netG0(self.real_A)
        # print("opc_A的最小值：", torch.min(self.opc_A))
        # print("opc_A的最大值：", torch.max(self.opc_A))
        # self.opc_A[self.opc_A < 0] = -1
        # self.opc_A[self.opc_A > 0] = 1
        self.fake_B = self.netG(self.opc_A)  # G(A)
        # print(self.fake_B.size())

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_opc, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_opc, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_D0(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.opc_A), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD0(fake_AB.detach())
        self.loss_D0_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_opc), 1)
        pred_real = self.netD(real_AB)
        self.loss_D0_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D0 = (self.loss_D0_fake + self.loss_D0_real) * 0.5
        self.loss_D0.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()


    def backward_G0(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_opc, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        fake_AB = torch.cat((self.real_A, self.opc_A), 1)
        pred_fake = self.netD0(fake_AB)
        self.loss_G0_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # weight_t = torch.tensor([[[[1., 1., 100.]]]])
        red_layer = 0
        green_layer = 1
        blue_layer = 2
        # lambda_L1 = self.opt.lambda_L1
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # green layer l1loss need to be small
        self.loss_OPC_L1 = self.criterionL1(self.opc_A[:, green_layer], self.real_opc[:, green_layer]) * self.opt.lambda_G
        self.loss_OPC_L1 += self.criterionL1(self.opc_A[:, red_layer], self.real_opc[:, red_layer]) * self.opt.lambda_R
        self.loss_OPC_L1 += self.criterionL1(self.opc_A[:, blue_layer], self.real_opc[:, blue_layer]) * self.opt.lambda_B
        # self.loss_OPC_L1 = self.criterionL1(self.opc_A, self.real_opc) * self.opt.lambda_OPC_L1
        # self.loss_G_L1 += self.criterionL1(self.fake_B[:, 1]*100, self.real_B[:, 1]*100) * self.opt.lambda_L1
        # self.loss_G_L1 += self.criterionL1(self.fake_B[:, 0]*10, self.real_B[:, 0]*10) * self.opt.lambda_L1
        # combine loss and calculate gradients
        # self.loss_EPE_L2 = self.criterionL2(self.fake_B[:, green_layer], self.real_A[:, red_layer]) * self.opt.lambda_L2
        # self.loss_EPE_L2 += self.criterionL2(self.fake_B[:, red_layer], self.real_A[:, red_layer]) * self.opt.lambda_L2
        # self.loss_EPE_L2 += self.criterionL2(self.fake_B[:, blue_layer], self.real_A[:, red_layer]) * self.opt.lambda_L2

        self.loss_EPE_L1 = self.criterionL1(self.fake_B[:, green_layer], self.real_A[:, red_layer]) * self.opt.lambda_EPE_L1
        self.loss_EPE_L1 += self.criterionL1(self.fake_B[:, red_layer], self.real_A[:, red_layer]) * self.opt.lambda_EPE_L1
        self.loss_EPE_L1 += self.criterionL1(self.fake_B[:, blue_layer], self.real_A[:, red_layer]) * self.opt.lambda_EPE_L1

        self.loss_G = self.loss_G_GAN + self.loss_G0_GAN + self.loss_G_L1+ self.loss_OPC_L1 + self.loss_EPE_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        self.set_requires_grad(self.netD0, True)  # enable backprop for D
        self.optimizer_D0.zero_grad()     # set D's gradients to zero
        self.backward_D0()                # calculate gradients for D
        self.optimizer_D0.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD0, False)
        self.set_requires_grad(self.netG, False)  # G requires no gradients when optimizing G0
        self.netG.eval()
        self.optimizer_G0.zero_grad()        # set G's gradients to zero
        self.backward_G0()                   # calculate graidents for G
        self.optimizer_G0.step()             # udpate G's weights
