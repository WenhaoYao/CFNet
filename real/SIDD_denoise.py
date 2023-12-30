import os
import numpy as np
from skimage import img_as_ubyte
import argparse

from torch.backends import cudnn

# from Less_channles import Network
from model.New_TestNet6_1111stage import Network
from tqdm import tqdm
from scipy.io import loadmat, savemat
import torch
cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def denoise(model, noisy_image):
    with torch.autograd.set_grad_enabled(False):
        torch.cuda.synchronize()

        _,phi_Z = model(noisy_image)
        torch.cuda.synchronize()
        im_denoise = phi_Z.cpu().numpy()

    im_denoise = np.transpose(im_denoise.squeeze(), (1, 2, 0))
    im_denoise = img_as_ubyte(im_denoise.clip(0, 1))

    return im_denoise


def test(args):
    use_gpu = True
    # load the pretrained model
    print('Loading the Model')
    # args = parse_benchmark_processing_arguments()
    checkpoint = torch.load(args.model)
    net = Network().cuda(device=device)
    if use_gpu:
        # net = net()
        net = Network().cuda(device=device)
        # net.load_state_dict(checkpoint)
        net.load_state_dict(checkpoint['state_dict'])
        net = torch.nn.DataParallel(net).cuda(device=device)

    net.eval()

    # load SIDD benchmark dataset and information
    noisy_data_mat_file = os.path.join(args.data_folder, 'BenchmarkNoisyBlocksSrgb.mat')
    noisy_data_mat_name = os.path.basename(noisy_data_mat_file).replace('.mat', '')
    noisy_data_mat = loadmat(noisy_data_mat_file)[noisy_data_mat_name]

    npose = (noisy_data_mat.shape[0])
    nsmile = noisy_data_mat.shape[1]
    poseSmile_cell = np.empty((npose, nsmile), dtype=object)

    # for image_index in tqdm(range(noisy_data_mat.shape[0])):
    #     for block_index in range(noisy_data_mat.shape[1]):
    for i in range(40):
        for j in range(32):
            noisy_image = noisy_data_mat[i, j, :, :, :]
            noisy_image = np.float32(noisy_image / 255.)
            noisy_image = torch.from_numpy(noisy_image.transpose((2, 0, 1))[np.newaxis,])
            print(noisy_image.shape)
            poseSmile_cell[i,j] = denoise(net, noisy_image)

    submit_data = {
            'DenoisedBlocksSrgb': poseSmile_cell
        }

    savemat(
            os.path.join(os.path.dirname(noisy_data_mat_file), 'SubmitSrgb.mat'),
            submit_data
        )