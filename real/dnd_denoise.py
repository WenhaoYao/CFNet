import PIL.Image
import numpy as np
import scipy.io as sio
import os
import h5py
import torch
import tqdm
import cv2


def load_nlf(info, img_id):
    nlf = {}
    nlf_h5 = info[info["nlf"][0][img_id]]
    nlf["a"] = nlf_h5["a"][0][0]
    nlf["b"] = nlf_h5["b"][0][0]
    return nlf


def load_sigma_raw(info, img_id, bb, yy, xx):
    nlf_h5 = info[info["sigma_raw"][0][img_id]]
    sigma = nlf_h5[xx, yy, bb]
    return sigma


def load_sigma_srgb(info, img_id, bb):
    nlf_h5 = info[info["sigma_srgb"][0][img_id]]
    sigma = nlf_h5[0, bb]
    return sigma


def denoise_raw(denoiser, data_folder, out_folder):
    """
    Utility function for denoising all bounding boxes in all raw images of
    the Benchmark_test dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch
                  and nlf is a dictionary containing the parameters of the noise level
                  function (nlf["a"], nlf["b"]) and a mean noise strength (nlf["sigma"])
    data_folder   Folder where the Benchmark_test dataset resides
    out_folder    Folder where denoised output should be written to
    """

    try:
        os.makedirs(out_folder)
    except:
        pass

    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in range(50):
        filename = os.path.join(data_folder, 'images_raw', '%04d.mat' % (i + 1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['Inoisy']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1], idx[2]:idx[3]].copy()
            Idenoised_crop = Inoisy_crop.copy()
            H = Inoisy_crop.shape[0]
            W = Inoisy_crop.shape[1]
            nlf = load_nlf(info, i)
            for yy in range(2):
                for xx in range(2):
                    nlf["sigma"] = load_sigma_raw(info, i, k, yy, xx)
                    Inoisy_crop_c = Inoisy_crop[yy:H:2, xx:W:2].copy()
                    Idenoised_crop_c = denoiser(Inoisy_crop_c, nlf)
                    Idenoised_crop[yy:H:2, xx:W:2] = Idenoised_crop_c
            # save denoised data
            Idenoised_crop = np.float32(Idenoised_crop)
            save_file = os.path.join(out_folder, '%04d_%02d.mat' % (i + 1, k + 1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            print('%s crop %d/%d' % (filename, k + 1, 20))
        print('[%d/%d] %s done\n' % (i + 1, 50, filename))


def tensor2img_Real(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        # n_img = len(tensor)
        # img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = tensor.numpy()
        # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def denoise_srgb(denoiser, data_folder, out_folder):
    """
    Utility function for denoising all bounding boxes in all sRGB images of
    the Benchmark_test dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch
                  and nlf is a dictionary containing the  mean noise strength (nlf["sigma"])
    data_folder   Folder where the Benchmark_test dataset resides
    out_folder    Folder where denoised output should be written to
    """
    try:
        os.makedirs(out_folder)
    except:
        pass

    print('model loaded\n')
    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data

    for i in range(50):
        filename = os.path.join(data_folder, 'images_srgb', '%04d.mat' % (i + 1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1], idx[2]:idx[3], :].copy()
            nlf = load_nlf(info, i)
            nlf["sigma"] = load_sigma_srgb(info, i, k)
            Inoisy_crop = torch.from_numpy(Inoisy_crop.transpose((2, 0, 1))[np.newaxis,])
            # print(Inoisy_crop.shape) # b ,c ,h,w = [1,3,512,512]
            with torch.autograd.set_grad_enabled(False):
                torch.cuda.synchronize()
                Idenoised_crop = denoiser(Inoisy_crop)  # b ,c ,h,w = [1,3,512,512]
                Idenoised_crop = torch.clamp(Idenoised_crop[1], 0., 1.)
                Idenoised_crop = Idenoised_crop.cpu().numpy()

                torch.cuda.synchronize()
            # Idenoised_crop = tensor2img_Real(Idenoised_crop, np.float32)
            Idenoised_crop = np.transpose(Idenoised_crop.squeeze(), (1, 2, 0))
            # print(Idenoised_crop.shape) # h,w,c
            # save denoised data
            Idenoised_crop = np.float32(Idenoised_crop)

            save_file = os.path.join(out_folder, '%04d_%02d.mat' % (i + 1, k + 1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            save_file = os.path.join(out_folder, 'Images', '%04d_%02d.PNG' % (i + 1, k + 1))
            cv2.imwrite(save_file, cv2.cvtColor(Idenoised_crop * 255, cv2.COLOR_RGB2BGR))
            print('%s crop %d/%d' % (filename, k + 1, 20))
        print('[%d/%d] %s done\n' % (i + 1, 50, filename))

    # for i in range(50):
    #     filename = os.path.join(data_folder, 'images_srgb', '%04d.mat' % (i + 1))
    #     img = h5py.File(filename, 'r')
    #     Inoisy = np.float32(np.array(img['InoisySRGB']).T)
    #     # bounding box
    #     ref = bb[0][i]
    #     boxes = np.array(info[ref]).T
    #     for k in range(20):
    #         idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]
    #         Inoisy_crop = Inoisy[idx[0]:idx[1], idx[2]:idx[3], :].copy()
    #         Inoisy_crop = torch.from_numpy(Inoisy_crop.transpose((2, 0, 1))[np.newaxis,])
    #         data = Inoisy_crop.unsqueeze(dim=0)
    #         with torch.autograd.set_grad_enabled(False):
    #             torch.cuda.synchronize()
    #             _,Idenoised_crop = denoiser(data)
    #             img = torch.clamp(Idenoised_crop, 0., 1.)
    #             img = img.cpu().numpy()
    #
    #             torch.cuda.synchronize()
    #         Idenoised_crop = tensor2img_Real(img, np.float32)
    #         Idenoised_crop = np.transpose(Idenoised_crop, (1, 2, 0))
    #         # save denoised data
    #         save_file = os.path.join(out_folder, 'Submit', '%04d_%02d.mat' % (i + 1, k + 1))
    #         sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
    #         save_file = os.path.join(out_folder, 'Images', '%04d_%02d.PNG' % (i + 1, k + 1))
    #         cv2.imwrite(save_file, cv2.cvtColor(Idenoised_crop * 255, cv2.COLOR_RGB2BGR))
    #         print('%s crop %d/%d' % (filename, k + 1, 20))
    #
    #     print('[%d/%d] %s done\n' % (i + 1, 50, filename))
