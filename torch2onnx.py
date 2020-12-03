import getpass
user = getpass.getuser()
import sys
sys.path.insert(0,'/home/'+user+'/DATA/Models/segmentation/torch')
sys.path.insert(0,'/home/'+user+'/DATA/Models/stereo/torch')
sys.path.insert(0,'/home/'+user+'/DATA/Models/classification/torch')
sys.path.insert(0,'/home/'+user+'/DATA/Models/instanceSeg/torch')
sys.path.insert(0,'/home/'+user+'/DATA/Models/localFeature/torch')

import torch
import torch.nn as nn
import scipy.io as sio
from torchvision import models
import segmentation
# import stereo
import classification
import instanceSeg
import localFeature


def torch_model_onnx(
    model, 
    input, 
    export_path,
    input_names = ['input'],
    output_names = ['output'],
    dynamic_axes = {'input' : {0 : 'batch_size'},    # variable lenght axes
                    'output': {0 : 'batch_size'}}
):
    """
    parameters:
        input -- tuple of tensors or just one tensor
        export_path -- "resnet.onnx"
    """

    model.eval()
    if isinstance(input, list):
        input = tuple(input)
    if not isinstance(input, (tuple)):
        input = (input,)

    for i in input:
        i.requires_grad = True

    output = model(*input)

    torch.onnx.export(
            model,    # model being run
            input,    # model input (or a tuple for multiple inputs)
            export_path,                # where to save the model (can be a file or file-like object)
            export_params=True,         # store the trained parameter weights inside the model file
            opset_version=11,           # the ONNX version to export the model to
            do_constant_folding=True,   # whether to execute constant folding for optimization
            input_names=input_names,    # the model's input names
            output_names=output_names,  # the model's output names
            dynamic_axes=dynamic_axes,    # variable lenght axes
            # keep_initializers_as_inputs=True
    )


def haha():
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    x = torch.rand(1,3,224,224)

    torch_model_onnx(
        model, 
        [x],
        './mobilenet_v2.onnx',
        input_names = ['x'],
        output_names = ['y'],
        dynamic_axes = None
    )

    print('export mobilenet_v2.onnx sucessfully!')
        


def resnet50(device='cuda'):
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    x = torch.rand(1,3,224,224).to(device)

    torch_model_onnx(
        model, 
        [x],
        './models/resnet50.onnx',
        input_names = ['x'],
        output_names = ['y'],
        dynamic_axes = None
    )

    print('export resnet50.onnx sucessfully!')

def hardnet(device='cuda'):
    model, preprocess = segmentation.build_model('hardnet', 'cityscapes', device)
    model.eval()
    x = torch.rand(1,3,320,512).to(device)

    torch_model_onnx(
        model, 
        [x],
        './models/hardnet.onnx',
        input_names = ['img'],
        output_names = ['seg'],
        dynamic_axes = None
    )

    print('export hardnet.onnx sucessfully!')

def aslfeat_onnx(device='cuda'):
    model, _ = localFeature.build_model('aslfeat_onnx', 'v3', device=device)
    model.eval()
    img = torch.rand(1,1,320,512).to(device)
    feat = torch.rand(1,128,80,128).to(device)
    des = torch.rand(1,128,1,1024).to(device)

    torch_model_onnx(
        model, 
        [img, feat, des],
        './models/aslfeat_onnx.onnx',
        input_names = ['img', 'feat', 'des'],
        output_names = ['layer3_input', 'det_kpt_inds', 'det_kpt_scores', 'det_kpt_inds_norm', 'descriptors'],
        dynamic_axes = None
    )

    print('export aslfeat_onnx.onnx sucessfully!')


def superpoint(device='cuda'):
    model, _ = localFeature.build_model('superpoint', 'default', device=device)
    model.eval()
    img = torch.rand(1,1,320,512).to(device)
    des = torch.rand(1,256,1,1024).to(device)

    torch_model_onnx(
        model, 
        [img, des], 
        './models/superpoint.onnx',
        input_names = ['img', 'des'],
        output_names = ['scores', 'keypoints', 'keypoints_norm', 'dense_feat', 'descriptors'],
        dynamic_axes = None
    )

    print('export superpoint.onnx sucessfully!')


def dcnV2():
    class DCN(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1):
            super(DCN, self).__init__()
            out_channels = deformable_groups * 3 * kernel_size * kernel_size
            self.conv_offset_mask = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        def forward(self, input):
            out = self.conv_offset_mask(input)
            o1, o2, mask = torch.chunk(out, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)
            return offset, mask
    
    model = DCN(128, 128, 3, 1, 1)
    s = torch.load('dcn.pth')
    model.load_state_dict({'conv_offset_mask.weight':s['conv_offset_mask.weight'], 'conv_offset_mask.bias':s['conv_offset_mask.bias']})
    model.eval()

    img = torch.rand(1,128,224,224)

    torch_model_onnx(
        model, 
        img,
        './models/dcnV2.onnx',
        input_names = ['img'],
        output_names = ['offset', 'mask'],
        dynamic_axes = None
    )

    print('export dcnV2.onnx sucessfully!')