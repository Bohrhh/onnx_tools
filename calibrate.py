import os
import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm
from base import OnnxModel
import scipy.io as sio


MOMENTUM = 0.1


class Calibrator():
    def __init__(self, dataloader, network_tensors):
        self.dataloader = dataloader
        self.network_tensors = network_tensors

    def __call__(self, base):

        assert isinstance(base, OnnxModel), "model should be an object of OnnxModel!"

        base.add_hook_nodes()
        model = base.model
        # runtime config
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel(0)
        sess_options.log_severity_level = 3
        sess = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=sess_options, providers=['CPUExecutionProvider'])

        # prepare input_name and output_names
        input_names = [i.name for i in sess.get_inputs()]
        output_names = [o.name for o in sess.get_outputs()]
        new_output_names = []
        for on in output_names:
            if '_reduceM' in on:
                new_output_names.append(on)

        # prepare dynamic_range  {tensor_name:[min, max], ...}
        dynamic_range = {}
        for non in new_output_names:
            tensor_name = non.split('_reduceM')[0]
            if tensor_name not in dynamic_range:
                dynamic_range[tensor_name] = [0,0]

        # calibration
        for x in tqdm(self.dataloader):
            x = [x]
            y = sess.run(new_output_names, dict(zip(input_names,x)))

            for i, non in enumerate(new_output_names):
                tensor_name = non.split('_reduceM')[0]
                if '_reduceMin' in non:
                    before = dynamic_range[tensor_name][0]
                    dynamic_range[tensor_name][0] = before*(1-MOMENTUM)+y[i]*MOMENTUM
                elif '_reduceMax' in non:
                    before = dynamic_range[tensor_name][1]
                    dynamic_range[tensor_name][1] = before*(1-MOMENTUM)+y[i]*MOMENTUM
                else:
                    assert False, "invalid output"

        # generate tensorrt dynamic range file
        tensors = []
        with open(self.network_tensors, 'r') as f:
            while True:
                s = f.readline()
                if s:
                    t = s.split(':')[1].strip()
                    tensors.append(t)
                else:
                    break
        s = ''
        for t in tensors:
            drange = dynamic_range.get(t)
            if drange is not None:
                drange = max(np.abs(drange))
                s += t+': '+str(drange)+'\n'
            else:
                if t in input_names:
                    s += t+': '+'\n'
        with open('dynamic_range.txt', 'w') as f:
            f.write(s)
        print("------------- Export dynamic_range.txt --------------")


class Dataset(object):
    def __init__(self, root):
        self.root = root
        self.datas = os.listdir(root)

    def __getitem__(self, idx):
        if idx>=self.__len__():
            raise IndexError
        x = os.path.join(self.root, self.datas[idx])
        x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
        x = (cv2.resize(x, (512, 320))/255.-0.449)/0.226
        x = x[None, None].astype('float32').copy()
        return x

    def __len__(self):
        # return len(self.datas)
        return 100


def main(model, network_tensors):
    """
    parameters:
        model -- onnx model filename or onnx.ModelProto
        network_tensors -- tensor names produced by onnx2trt
    """

    #-----------------------------------------------
    # prepare model
    #-----------------------------------------------
    base = OnnxModel(model)
    dataloader = Dataset('/home/kmlee/DATA/mscoco_2014/val2014')

    #-----------------------------------------------
    # working procedur
    #-----------------------------------------------
    calib = Calibrator(dataloader, network_tensors)

    #-----------------------------------------------
    # produce
    #-----------------------------------------------
    calib(base)