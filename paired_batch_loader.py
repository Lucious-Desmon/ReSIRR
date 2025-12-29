import os
import numpy as np
import cv2


class PairedMiniBatchLoader(object):
    def __init__(self, input_path, label_path, image_dir_path, crop_size):
        self.input_path_infos = self.read_paths(input_path, image_dir_path)
        self.label_path_infos = self.read_paths(label_path, image_dir_path)
        if len(self.input_path_infos) != len(self.label_path_infos):
            raise RuntimeError("input/label list length mismatch")
        self.crop_size = crop_size

    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path

    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c

    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in PairedMiniBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs

    def load_training_data(self, indices):
        return self.load_data(self.input_path_infos, self.label_path_infos, indices, augment=True)

    def load_testing_data(self, indices):
        return self.load_data(self.input_path_infos, self.label_path_infos, indices, augment=False)

    def load_data(self, input_infos, label_infos, indices, augment=False):
        mini_batch_size = len(indices)
        in_channels = 3

        if augment:
            xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
            ys = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
            for i, index in enumerate(indices):
                in_path = input_infos[index]
                label_path = label_infos[index]

                img = cv2.imread(in_path)
                label = cv2.imread(label_path)
                if img is None or label is None:
                    raise RuntimeError("invalid image: {i} {l}".format(i=in_path, l=label_path))
                if img.shape != label.shape:
                    raise RuntimeError("input/label size mismatch: {i} {l}".format(i=in_path, l=label_path))

                h, w, _ = img.shape

                if np.random.rand() > 0.5:
                    img = np.fliplr(img)
                    label = np.fliplr(label)

                if np.random.rand() > 0.5:
                    angle = 10 * np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                    img = cv2.warpAffine(img, M, (w, h))
                    label = cv2.warpAffine(label, M, (w, h))

                rand_range_h = h - self.crop_size
                rand_range_w = w - self.crop_size
                x_offset = np.random.randint(rand_range_w)
                y_offset = np.random.randint(rand_range_h)

                img = img[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size]
                label = label[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size]

                xs[i, :, :, :] = np.transpose((img / 255).astype(np.float32), (2, 0, 1))
                ys[i, :, :, :] = np.transpose((label / 255).astype(np.float32), (2, 0, 1))
        elif mini_batch_size == 1:
            for i, index in enumerate(indices):
                in_path = input_infos[index]
                label_path = label_infos[index]
                img = cv2.imread(in_path)
                label = cv2.imread(label_path)
                if img is None or label is None:
                    raise RuntimeError("invalid image: {i} {l}".format(i=in_path, l=label_path))
                if img.shape != label.shape:
                    raise RuntimeError("input/label size mismatch: {i} {l}".format(i=in_path, l=label_path))

            h, w, _ = img.shape
            xs = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            ys = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            xs[0, :, :, :] = np.transpose((img / 255).astype(np.float32), (2, 0, 1))
            ys[0, :, :, :] = np.transpose((label / 255).astype(np.float32), (2, 0, 1))
        else:
            raise RuntimeError("mini batch size must be 1 when testing")

        return xs, ys
