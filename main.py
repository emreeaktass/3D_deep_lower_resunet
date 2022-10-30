import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from math import atan2, tan


class FingerprintSeperation:

    def __init__(self, overlap_image_path, mask1_path, mask2_path):
        self.overlap_image_path = overlap_image_path
        self.mask1_path = mask1_path
        self.mask2_path = mask2_path
        self.I = None
        self.I1 = None
        self.I2 = None
        self.IO = None
        self.R1 = None
        self.R2 = None
        self.RO = None
        self.RN1 = None
        self.RN2 = None

    def read_files(self):
        self.I = cv2.imread(self.overlap_image_path, 0)
        self.R1 = cv2.imread(self.mask1_path, 0)
        self.R2 = cv2.imread(self.mask2_path, 0)
        self.RO = cv2.bitwise_and(self.R1, self.R2)

        self.RN1 = self.R1 - self.RO
        self.RN2 = self.R2 - self.RO
        # print(np.nonzero(self.RO[128:144, 208:224]))
        # print(self.RN1[128:144, 208:224].sum(), self.RN2[128:144, 208:224].sum(), self.RO[128:144, 208:224].sum())
        self.I1 = self.I.copy()
        self.I2 = self.I.copy()
        self.IO = self.I.copy()
        self.I1[self.RN1 != 255] = 0
        self.I2[self.RN2 != 255] = 0
        self.IO[self.RO != 255] = 0
        # print("emre")
        # print(self.I1[128:144, 208:224].sum(), self.I2[128:144, 208:224].sum(), self.IO[128:144, 208:224].sum())
        # self.I1 = cv2.bitwise_and(self.I, self.I, mask=self.RN1)
        # self.I2 = cv2.bitwise_and(self.I, self.I, mask=self.RN2)
        # self.IO = cv2.bitwise_and(self.I, self.I, mask=self.RO)
        return self.I, self.I1, self.I2, self.IO, self.RO, self.R1, self.R2, self.RN1, self.RN2

    def initial_pad(self, block_size):
        width = self.IO.shape[0]
        height = self.IO.shape[1]

        new_width = width + block_size - (width % block_size)
        new_height = height + block_size - (height % block_size)

        new_IO = np.zeros((new_width, new_height))
        new_I1 = np.zeros((new_width, new_height))
        new_I2 = np.zeros((new_width, new_height))

        new_IO[:width, :height] = self.IO
        new_I1[:width, :height] = self.I1
        new_I2[:width, :height] = self.I2

        return new_IO, new_I1, new_I2

    # Taken from https://github.com/rahulsingh786/overlapped-fingerprint-seperator/blob/master/model.py
    def gaussian_window(self):
        biv_normal = scipy.stats.multivariate_normal(mean=[31.5, 31.5],
                                                     cov=[[16, 0], [0, 16]])
        biv_normal_pdf = np.zeros((64, 64))
        for ii in np.arange(64):
            for jj in np.arange(64):
                biv_normal_pdf[ii, jj] = biv_normal.pdf((ii, jj))
        return biv_normal_pdf

    def sort_blocks(self, IO_dict, I1_dict, I2_dict, row_block_size, column_block_size):

        sorted_dict = dict()
        block_number = len(I1_dict)
        print(block_number, row_block_size, column_block_size, row_block_size * column_block_size)

        iteration = 1
        while len(IO_dict) > 0:
            for i, j in list(zip(IO_dict.keys(), IO_dict.values())):
                if j.emthy == 0:
                    # if I1_dict[i].orientation != I2_dict[i].orientation:
                    # print("emre", I1_dict[i].orientation, I2_dict[i].orientation, I2_dict[i].left_up_coordinate, I1_dict[i].left_up_coordinate)
                    # print(I1_dict[i].orientation, I2_dict[i].orientation)
                    if (i - iteration) % column_block_size != column_block_size - 1 or (
                            i + iteration) % column_block_size != 0:
                        if I1_dict[i - iteration].emthy == 0 or I2_dict[i + iteration].emthy == 0:
                            sorted_dict[i] = j
                            IO_dict.pop(i)

                    elif (i + iteration * row_block_size) < row_block_size:
                        if I1_dict[i + iteration * row_block_size].emthy == 0 or I2_dict[
                            i + iteration * row_block_size].emthy == 0:
                            sorted_dict[i] = j
                            IO_dict.pop(i)
                    elif i - iteration * row_block_size > 0:
                        if I1_dict[i - iteration * row_block_size].emthy == 0 or I2_dict[
                            i + iteration * row_block_size].emthy == 0:
                            sorted_dict[i] = j
                            IO_dict.pop(i)

                else:
                    IO_dict.pop(i)
            iteration = iteration + 1

        return sorted_dict


class Block:
    def __init__(self, type_, left_up_coordinate, img, gaussian_window, label_prob, label, empthy=1, block_size=16,
                 window_size=64):
        self.type_ = type_
        self.left_up_coordinate = left_up_coordinate
        self.orientation = None
        self.img = img
        self.block_size = block_size
        self.window_size = window_size
        self.gaussian_window = gaussian_window
        self.emthy = empthy
        self.label_prob = label_prob
        self.label = label

    def find_orientation(self):
        x = self.left_up_coordinate[0]
        y = self.left_up_coordinate[1]
        block = self.img[x:x + 16, y:y + 16]

        if block.sum() > 0:
            if x == 128 and y == 208:
                print("test", block.sum())
            if x - 24 < 0:
                # left
                self.img = np.pad(self.img, [(0, 0), (24 - x, 0)], mode="constant")
                x = x + (24 - x)
            if x + 40 > self.img.shape[1]:
                # right
                self.img = np.pad(self.img, [(0, 0), (0, np.abs((self.img.shape[1] - (x + 16) - 24)))], mode="constant")
                x = x + (np.abs((self.img.shape[1] - (x + 16) - 24)))
            if y - 24 < 0:
                # up
                self.img = np.pad(self.img, [(24 - y, 0), (0, 0)], mode="constant")
                y = y + (24 - y)
            if y + 40 > self.img.shape[0]:
                # down
                self.img = np.pad(self.img, [(0, np.abs((self.img.shape[0] - (y + 16) - 24))), (0, 0)], mode="constant")
                y = y + np.abs((self.img.shape[0] - (y + 16) - 24))

            sub_image = self.img[x - 24:x + 40, y - 24:y + 40]
            sub_image = sub_image * self.gaussian_window

            dft = cv2.dft(np.float32(sub_image), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            dft_shift[self.window_size // 2 - 1: self.window_size // 2 + 2,
            self.window_size // 2 - 1: self.window_size // 2 + 2, :] = 0
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
            maximum_index = np.unravel_index(magnitude_spectrum.argmax(), magnitude_spectrum.shape)
            ori_list = []
            orientation = atan2(maximum_index[1], maximum_index[0]) * 180 / np.pi
            ori_list.append(orientation)

            # comp = complex(dft_shift[maximum_index][0], dft_shift[maximum_index][1])
            # theta = np.angle([comp])

            if self.type_ == 2:
                magnitude_spectrum[maximum_index] = 0
                maximum_index = np.unravel_index(magnitude_spectrum.argmax(), magnitude_spectrum.shape)
                orientation = atan2(maximum_index[1], maximum_index[0])
                ori_list.append(orientation)

            self.orientation = ori_list
            self.emthy = 0
        else:
            self.emthy = 1

    def calculate_R(self):
        pass

    def calculate_W(self):
        pass

    def calculate_Q(self):
        pass

    def labeling(self, sorted_overlap_blocks):

        k = 0
        n = len(sorted_overlap_blocks)
        while k < 50:
            for i in range(n):
                pass


if __name__ == '__main__':
    BLOCK_SIZE = 16
    overlap_mask_path = "latentOverlap/overlap/00+00_3.bmp"
    mask1_path = "latentOverlap/mask/00+00_3_1.bmp"
    mask2_path = "latentOverlap/mask/00+00_3_2.bmp"
    sep = FingerprintSeperation(overlap_mask_path, mask1_path, mask2_path)
    gaussian_window = sep.gaussian_window()

    I, a, b, c, RO, R1, R2, RN1, RN2 = sep.read_files()
    # print(a[128:144, 208:224].sum(), b[128:144, 208:224].sum(), c[128:144, 208:224].sum())
    IO, I1, I2 = sep.initial_pad(BLOCK_SIZE)

    # print(I1[128:144, 208:224].sum(), I2[128:144, 208:224].sum(), IO[128:144, 208:224].sum())

    left_up_coordinate = [(i, j) for i in np.arange(0, IO.shape[0], BLOCK_SIZE) for j in
                          np.arange(0, IO.shape[1], BLOCK_SIZE)]
    block_list_1 = []
    block_list_2 = []
    block_list_3 = []
    for i in left_up_coordinate:
        block_1 = Block(1, i, I1, gaussian_window, 1, 1)
        block_1.find_orientation()
        block_list_1.append(block_1)
        block_2 = Block(2, i, IO, gaussian_window, [0.5, 0.5], 0)
        block_2.find_orientation()
        block_list_2.append(block_2)
        block_3 = Block(1, i, I2, gaussian_window, 1, 2)
        block_3.find_orientation()
        block_list_3.append(block_3)
    block_ids = np.arange(0, len(left_up_coordinate))

    I1_dict = dict(zip(block_ids, block_list_1))
    IO_dict = dict(zip(block_ids, block_list_2))
    I2_dict = dict(zip(block_ids, block_list_3))

    # for j in IO_dict.values():
    #     if j.orientation is not None:
    #         print(j.orientation, j.left_up_coordinate)
    # for j in I1_dict.values():
    #     if j.orientation is not None:
    #         print(j.orientation, j.left_up_coordinate)
    # for j in I2_dict.values():
    #     if j.orientation is not None:
    #         print(j.orientation, j.left_up_coordinate)

    sorted_dict = sep.sort_blocks(IO_dict, I1_dict, I2_dict, IO.shape[1] // 16, IO.shape[0] // 16)
    for i, j in zip(sorted_dict.keys(), sorted_dict.values()):
        print(i, j.left_up_coordinate)
    # for i, j in zip(I1_dict.keys(), I1_dict.values()):
    #     print(i, j.left_up_coordinate)

    plt.figure(1)
    plt.subplot(331)
    plt.imshow(I, cmap='gray')
    plt.title("I"), plt.xticks([]), plt.yticks([])
    plt.subplot(332)
    plt.imshow(I1, cmap='gray')
    plt.title("I1"), plt.xticks([]), plt.yticks([])
    plt.subplot(333)
    plt.imshow(IO, cmap='gray')
    plt.title("IO"), plt.xticks([]), plt.yticks([])
    plt.subplot(334)
    plt.imshow(I2, cmap='gray')
    plt.title("I2"), plt.xticks([]), plt.yticks([])
    plt.subplot(335)
    plt.imshow(RO, cmap='gray')
    plt.title("RO"), plt.xticks([]), plt.yticks([])
    plt.subplot(336)
    plt.imshow(R1, cmap='gray')
    plt.title("R1"), plt.xticks([]), plt.yticks([])
    plt.subplot(337)
    plt.imshow(R2, cmap='gray')
    plt.title("R2"), plt.xticks([]), plt.yticks([])
    plt.subplot(338)
    plt.imshow(RN1, cmap='gray')
    plt.title("RN1"), plt.xticks([]), plt.yticks([])
    plt.subplot(339)
    plt.imshow(RN2, cmap='gray')
    plt.title("RN2"), plt.xticks([]), plt.yticks([])
    plt.show()
