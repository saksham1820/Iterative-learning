# this file is to test the cross correlation loss, maybe in 2d.
import math

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


class CCLoss:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def __call__(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt

    ref_image = Image.open(
        "/home/jizong/UserSpace/Iterative-learning/.data/ACDC_contrast_with_prediction/train/img/patient002_00_07.png").convert(
        "L")
    ref_image = torch.from_numpy(np.asarray(ref_image) / 255.0).float()
    ref_image = ref_image[None, None, ...]
    ref_image = ref_image.to("cuda")


    def run(win=3):
        input_image = torch.rand_like(ref_image, requires_grad=True, device=ref_image.device)
        optimizer = torch.optim.Adam((input_image,), lr=1e-1)
        criterion = CCLoss(win=(win, win))
        with tqdm(range(1000)) as indicator:
            for i in indicator:
                optimizer.zero_grad()
                loss = criterion(input_image, ref_image)
                loss.backward()
                optimizer.step()
                indicator.set_postfix_str(f"loss:{loss.item()}")
        return input_image.detach()


    wins = (3, 5, 7)
    images = [run(x) for x in wins]
    plt.figure(0)
    plt.imshow(ref_image.squeeze().cpu())

    for e, i in enumerate(wins):
        plt.figure(i)
        plt.imshow(images[e].cpu().squeeze())
    plt.show()
