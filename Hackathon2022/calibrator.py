import tensorrt as trt
import os

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from sys import getsizeof

# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
# import helpers.tokenization as tokenization
# import helpers.data_processing as dp

class EncoderCalibrator(trt.IInt8LegacyCalibrator):
    def __init__(self, calibration_data_file, cache_file, batch_size):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8LegacyCalibrator.__init__(self)

        self.cache_file = cache_file

        # self.feat_list = feat_list
        # self.feat_len_list = feat_len_list
        self.batch_size = batch_size
        self.current_index = 0

        print("start read " + calibration_data_file)
        # feat_name_list = []
        self.feat_list = []
        self.feat_len_list = []
        data = np.load(calibration_data_file)
        for i in data.files:
            if "speech-" in i:
                self.feat_list.append(data[i])
                print(i)
                print(data[i].shape)
            if "speech_lengths" in i:
                self.feat_len_list.append(data[i])
                print(i)
                print(data[i].shape)

        if len(self.feat_list) != len(self.feat_len_list):
            print("len(feat_list) != len(feat_len_list)")
            assert(0)

        self.num_inputs = len(self.feat_list)
        # self.num_inputs = 1

        self.d_feat = None
        self.d_feat_len = None

    def free(self):
        pass

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        # print("self.num_inputs:" + str(self.num_inputs))
        # print("self.current_index:" + str(self.current_index))
        if self.current_index >= self.num_inputs:
            print("Calibrating index {:} batch size {:} exceed max input limit {:} sentences".format(self.current_index, self.batch_size, self.num_inputs))
            return None


            np_feats = np.concatenate((np_feats, feat), axis=0)

            feat_len = self.feat_len_list[self.current_index + i]
            np_feat_lens = np.concatenate((np_feat_lens, feat_len), axis=0)

        np_feats = self.feat_list[self.current_index]
        np_feat_lens = self.feat_len_list[self.current_index]
        # print(np_feats.shape)
        # print(np_feat_lens.shape)
        self.d_feat = cuda.mem_alloc(np_feats.size * 4)
        self.d_feat_len = cuda.mem_alloc(np_feat_lens.size * 4)

        print(getsizeof(np_feats))
        print(self.d_feat_len)

        cuda.memcpy_htod(self.d_feat, np_feats.ravel())
        cuda.memcpy_htod(self.d_feat_len, np_feat_lens.ravel())

        self.current_index += 1
        return [self.d_feat, self.d_feat_len]

        # t_feats = torch.from_numpy(np_feats).cuda()
        # t_feat_lens = torch.from_numpy(np_feat_lens).cuda()

        # return [t_feats.data_ptr(), t_feat_lens.data_ptr()]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            os.fsync(f)

    def get_quantile(self):
        return 0.9999

    def get_regression_cutoff(self):
        return 1.0

    def read_histogram_cache(self, length):
        return None

    def write_histogram_cache(self, ptr, length):
        return None


def main():
    c = EncoderCalibrator("/workspace/data/calibration.npz", "encoder.cache", 100)
    c.get_batch("input")
    c.get_batch("input")
    c.get_batch("input")
    c.get_batch("input")

if __name__ == '__main__':
    main()
