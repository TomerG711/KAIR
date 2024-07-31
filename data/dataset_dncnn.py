import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util

#TODO: Fix original to use only the first 3k images of npy file. perhaps create new npy file?
class OriginalDatasetDnCNN(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(OriginalDatasetDnCNN, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        # self.paths_H = util.get_image_paths(opt['dataroot_H'])
        if self.opt['phase'] == 'train':
            self.data = np.load(opt['npy_path'])
        else:
            self.data = np.load(opt['npy_path'], allow_pickle=True)

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        # H_path = self.paths_H[index]
        # img_H = util.imread_uint(H_path, self.n_channels)

        # L_path = H_path

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            img_H = self.data[index][0]
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()

            # --------------------------------
            # add noise
            # --------------------------------
            noise = torch.randn(img_L.size()).mul_(self.sigma / 255.0)
            img_L.add_(noise)

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            # img_H = util.uint2single(img_H)
            # img_L = np.copy(img_H)
            #
            # # --------------------------------
            # # add noise
            # # --------------------------------
            # np.random.seed(seed=0)
            # img_L += np.random.normal(0, self.sigma_test / 255.0, img_L.shape)
            #
            # # --------------------------------
            # # HWC to CHW, numpy to tensor
            # # --------------------------------
            # img_L = util.single2tensor3(img_L)
            # img_H = util.single2tensor3(img_H)
            img_H = self.data[index][0]
            img_L = self.data[index][1]
            img_H = util.uint2tensor3(img_H)
            img_L = util.uint2tensor3(img_L)

        return {'L': img_L, 'H': img_H, 'GT': img_H, 'H_path': f'H_{index}', 'L_path': f'L_{index}'}

    def __len__(self):
        # return len(self.paths_H)
        if self.opt['phase'] == 'train':
            # print(self.data.shape[0])
            return self.data.shape[0]
        else:
            return self.data.shape[0]


class SUREDatasetDnCNN(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(SUREDatasetDnCNN, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        # self.sigma = opt['sigma'] if opt['sigma'] else 25
        # self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        # Tomer - Use NPY
        if self.opt['phase'] == 'train':
            self.data = np.load(opt['npy_path'])
        else:
            self.data = np.load(opt['npy_path'], allow_pickle=True)
            # self.paths_H = util.get_image_paths(opt['dataroot_H'])
        # Tomer - for constant noise per image
        # self.train_noise = torch.randn((self.__len__(), 180, 180, 1)).mul_(self.sigma / 255.0).detach().cpu().numpy()
        # self.test_noise = torch.randn((68, 1, self.patch_size, self.patch_size)).mul_(self.sigma_test / 255.0).detach().cpu().numpy()

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        # H_path = self.paths_H[index]
        # img_H = util.imread_uint(H_path, self.n_channels)
        # L_path = H_path

        if self.opt['phase'] == 'train':
            img_H = self.data[index][0]
            img_L = self.data[index][1]
            # print(img_H.min(), img_H.max())
            # print(img_L.min(), img_L.max())
            # img_H = np.expand_dims(img_H, axis=2)  # HxWx1

            # Tomer - add constant noise. There is no need for real H (we assume there is no GT)
            # img_H = util.uint2single(img_H)
            # img_H += self.train_noise[index]

            # if 'shift' in H_path:
            #     np.random.seed(index)
            #     img_H = cyclic_shift(img_H, 0.25)

            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            # print(H,W)
            # print(rnd_h, rnd_w)
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            # noise_for_patch = self.train_noise[index][:,rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            # print(self.train_noise[index].shape)
            # print(noise_for_patch.shape)
            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)
            patch_L = util.augment_img(patch_L, mode=mode)
            # noise_for_patch = util.augment_img(noise_for_patch, mode=mode)
            # print(noise_for_patch)
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)
            # img_H = util.single2tensor3(patch_H)
            # print("img_H: ")
            # print(img_H.shape)
            # print("nosie_for_patch: ")
            # print(noise_for_patch.shape)
            # img_H += noise_for_patch

            # img_L = img_H.clone()

            # --------------------------------
            # add noise
            # --------------------------------
            # TOMER - Noise already added to img_H
            # noise = torch.randn(img_L.size()).mul_(self.sigma/255.0)
            # img_L.add_(noise)
            return {'L': img_L, 'H': img_L, 'GT':img_H, 'H_path': f'H_{index}', 'L_path': f'L_{index}'}

        else:
            # H_path = self.paths_H[index]
            # img_H = util.imread_uint(H_path, self.n_channels)
            img_H = self.data[index][0]
            # L_path = H_path
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """

            # img_H = util.uint2single(img_H)
            # img_L = np.copy(img_H)
            # img_H = util.uint2tensor3(img_H)

            img_L = self.data[index][1]
            # img_L = np.expand_dims(img_L, axis=2)  # HxWx1

            img_H = util.uint2tensor3(img_H)
            img_L = util.uint2tensor3(img_L)

            # --------------------------------
            # add noise
            # --------------------------------
            # np.random.seed(seed=0)
            # img_L += np.random.normal(0, self.sigma_test / 255.0, img_L.shape)

            # Tomer - use pre-defined noise
            # print("img_L:")
            # print(img_L.shape)
            # print("nosie:")
            # print(self.test_noise[index].shape)
            # img_L += self.test_noise[index]

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            # img_L = util.single2tensor3(img_L)
            # img_H = util.single2tensor3(img_H)

        # return {'L': img_L, 'H': img_H, 'H_path': f"H_{index}", 'L_path': L_path}
            return {'L': img_L, 'H': img_H, 'GT': img_H, 'H_path': f'H_{index}', 'L_path': f'L_{index}'}

    def __len__(self):
        # if self.opt['phase'] == 'train':
        return self.data.shape[0]
        # else:
        #     return len(self.paths_H)


class N2NDatasetDnCNN(data.Dataset):
    # TODO: Train N2N with predefined pairs
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------

    N2V - Assume NPY of triplets:
        0 - clean
        1 - first noisy
        2 - aligned noisy
    """

    def __init__(self, opt):
        super(N2NDatasetDnCNN, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        # self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        # Tomer - Use NPY
        if self.opt['phase'] == 'train':
            self.data = np.load(opt['npy_path'])
        else:
            self.data = np.load(opt['npy_path'], allow_pickle=True)
            # self.paths_H = util.get_image_paths(opt['dataroot_H'])
        # Tomer - for constant noise per image
        self.train_noise1 = torch.randn((self.__len__(), 180, 180, 1)).mul_(self.sigma).detach().cpu().numpy()
        self.train_noise2 = torch.randn((self.__len__(), 180, 180, 1)).mul_(self.sigma).detach().cpu().numpy()
        # self.test_noise = torch.randn((68, 1, self.patch_size, self.patch_size)).mul_(self.sigma_test / 255.0).detach().cpu().numpy()

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        # H_path = self.paths_H[index]
        # img_H = util.imread_uint(H_path, self.n_channels)
        # L_path = H_path

        if self.opt['phase'] == 'train':
            img_H = self.data[index][0]
            img_L = img_H + self.train_noise1[index]
            img_L2 = img_H + self.train_noise2[index]
            # img_L = self.data[index][1]
            # img_L2 = self.data[index][2]

            # print(img_H.min(), img_H.max())
            # print(img_L.min(), img_L.max())
            # img_H = np.expand_dims(img_H, axis=2)  # HxWx1

            # Tomer - add constant noise. There is no need for real H (we assume there is no GT)
            # img_H = util.uint2single(img_H)
            # img_H += self.train_noise[index]

            # if 'shift' in H_path:
            #     np.random.seed(index)
            #     img_H = cyclic_shift(img_H, 0.25)

            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            # print(H,W)
            # print(rnd_h, rnd_w)
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L2 = img_L2[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # patch_L = patch_H + torch.randn(patch_H.shape).mul_(self.sigma)
            # patch_L = patch_H + np.random.normal(size=patch_H.shape) * self.sigma
            # patch_L2 = patch_H + np.random.normal(size=patch_H.shape) * self.sigma
            # noise_for_patch = self.train_noise[index][:,rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            # print(self.train_noise[index].shape)
            # print(noise_for_patch.shape)
            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)
            patch_L = util.augment_img(patch_L, mode=mode)
            patch_L2 = util.augment_img(patch_L2, mode=mode)
            # noise_for_patch = util.augment_img(noise_for_patch, mode=mode)
            # print(noise_for_patch)
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)
            img_L2 = util.uint2tensor3(patch_L2)
            # img_H = util.single2tensor3(patch_H)
            # print("img_H: ")
            # print(img_H.shape)
            # print("nosie_for_patch: ")
            # print(noise_for_patch.shape)
            # img_H += noise_for_patch

            # img_L = img_H.clone()

            # --------------------------------
            # add noise
            # --------------------------------
            # TOMER - Noise already added to img_H
            # noise = torch.randn(img_L.size()).mul_(self.sigma/255.0)
            # img_L.add_(noise)
            return {'L': img_L, 'H': img_L2, 'GT': img_H, 'H_path': f'H_{index}', 'L_path': f'L_{index}'}

        else:
            # H_path = self.paths_H[index]
            # img_H = util.imread_uint(H_path, self.n_channels)
            # L_path = H_path
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """

            # img_H = util.uint2single(img_H)
            # img_L = np.copy(img_H)
            # img_H = util.uint2tensor3(img_H)
            img_H = self.data[index][0]
            img_L = self.data[index][1]
            # img_L = np.expand_dims(img_L, axis=2)  # HxWx1

            img_H = util.uint2tensor3(img_H)
            img_L = util.uint2tensor3(img_L)

            # --------------------------------
            # add noise
            # --------------------------------
            # np.random.seed(seed=0)
            # img_L += np.random.normal(0, self.sigma_test / 255.0, img_L.shape)

            # Tomer - use pre-defined noise
            # print("img_L:")
            # print(img_L.shape)
            # print("nosie:")
            # print(self.test_noise[index].shape)
            # img_L += self.test_noise[index]

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            # img_L = util.single2tensor3(img_L)
            # img_H = util.single2tensor3(img_H)

        # return {'L': img_L, 'H': img_H, 'H_path': f"H_{index}", 'L_path': L_path}
            return {'L': img_L, 'H': img_H, 'GT': img_H, 'H_path': f'H_{index}', 'L_path': f'L_{index}'}

    def __len__(self):
        # if self.opt['phase'] == 'train':
        return self.data.shape[0]
        # else:
        #     return len(self.paths_H)
