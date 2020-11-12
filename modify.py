import copy
import onnx
import torch
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx import numpy_helper
import numpy as np


def pydnet():
    model = onnx.load("models/pydnet.onnx")

    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 3
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 384
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 640
    model.graph.input[0].name = "img"
    model.graph.node.pop(0)
    model.graph.node[0].input[0] = "img"

    for i in model.graph.initializer:
        if i.name == "new_shape__120":
            model.graph.initializer.remove(i)
            a = numpy_helper.from_array(np.array([1,1,384,640]))
            a.name = "new_shape__120"
            print(a)
            model.graph.initializer.append(a)
            break

    model.graph.output[0].name = "depth"
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 1
    model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 384
    model.graph.output[0].type.tensor_type.shape.dim[3].dim_value = 640

    model.graph.node[-1].output[0] = "depth"
    onnx.save(model, "pydnet.onnx")

def MiDaS():
    model = onnx.load("models/MiDaS.onnx")
    model.graph.input[0].name = "img"
    model.graph.output[0].name = "depth"
    model.graph.node[0].input[0] = "img"

    model.graph.node.pop()
    model.graph.node[-1].output[0] = "depth"
    a = copy.deepcopy(model.graph.output[0].type.tensor_type.shape.dim[0])
    model.graph.output[0].type.tensor_type.shape.dim.insert(0,a)
    onnx.save(model, "MiDaS.onnx")


def aslfeat():
    model =  onnx.load("models/aslfeat.onnx")

    grid_sample = helper.make_node(
        'GridSample', # node name
        ['feat', 'det_kpt_inds_norm'], # inputs
        ['des'], # outputs
        mode = 1,
        padding_mode = 0,
        align_corners = True
    )

    # ===============================
    # ModulatedDeformConvPack 1
    conv1 = helper.make_node(
        'Conv',
        ['layer3_input', 'layer3.conv1.0.conv_offset_mask.weight', 'layer3.conv1.0.conv_offset_mask.bias'],
        ['layer3.conv1.0.out'],
        dilations=[1,1],
        group=1,
        kernel_shape=[3,3],
        pads=[1,1,1,1],
        strides=[1,1]
    )
    split1 = helper.make_node(
        'Split',
        ['layer3.conv1.0.out'],
        ['split1_1', 'split1_2', 'split1_3'],
        axis=1,
        split=[9,9,9]
    )
    concat1 = helper.make_node(
        'Concat',
        ['split1_1', 'split1_2'],
        ['concat1_out'],
        axis=1,
    )
    sigmoid1 = helper.make_node(
        'Sigmoid',
        ['split1_3'],
        ['sigmoid1_out'],
    )
    dcnV2_1 = helper.make_node(
        'DCNv2', # node name
        ['layer3_input', 'concat1_out', 'sigmoid1_out', 'layer3.conv1.0.weight', 'layer3.conv1.0.bias'], # inputs
        ['dcnV2_1_out'], # outputs
        deformable_group = 1,
        dilation = 1,
        groups = 1,
        padding = 1,
        stride = 1
    )

    bn_1 = helper.make_node(
        'BatchNormalization',
        ['dcnV2_1_out', 'bn_1_scale', 'bn_1_B', 'layer3.conv1.1.running_mean', 'layer3.conv1.1.running_var'],
        ['bn_1_out'],
        epsilon = 1e-5,
        momentum = 0.9
    )
    relu_1 = helper.make_node(
        'Relu',
        ['bn_1_out'],
        ['relu_1_out']
    )

    # ===============================
    # ModulatedDeformConvPack 2
    conv2 = helper.make_node(
        'Conv',
        ['relu_1_out', 'layer3.conv2.0.conv_offset_mask.weight', 'layer3.conv2.0.conv_offset_mask.bias'],
        ['layer3.conv2.0.out'],
        dilations=[1,1],
        group=1,
        kernel_shape=[3,3],
        pads=[1,1,1,1],
        strides=[1,1]
    )
    split2 = helper.make_node(
        'Split',
        ['layer3.conv2.0.out'],
        ['split2_1', 'split2_2', 'split2_3'],
        axis=1,
        split=[9,9,9]
    )
    concat2 = helper.make_node(
        'Concat',
        ['split2_1', 'split2_2'],
        ['concat2_out'],
        axis=1,
    )
    sigmoid2 = helper.make_node(
        'Sigmoid',
        ['split2_3'],
        ['sigmoid2_out'],
    )
    dcnV2_2 = helper.make_node(
        'DCNv2', # node name
        ['relu_1_out', 'concat2_out', 'sigmoid2_out', 'layer3.conv2.0.weight', 'layer3.conv2.0.bias'], # inputs
        ['dcnV2_2_out'], # outputs
        deformable_group = 1,
        dilation = 1,
        groups = 1,
        padding = 1,
        stride = 1
    )

    bn_2 = helper.make_node(
        'BatchNormalization',
        ['dcnV2_2_out', 'bn_2_scale', 'bn_2_B', 'layer3.conv2.1.running_mean', 'layer3.conv2.1.running_var'],
        ['bn_2_out'],
        epsilon = 1e-5,
        momentum = 0.9
    )
    relu_2 = helper.make_node(
        'Relu',
        ['bn_2_out'],
        ['relu_2_out']
    )


    # ===============================
    # ModulatedDeformConvPack 3
    conv3 = helper.make_node(
        'Conv',
        ['relu_2_out', 'layer3.conv3.conv_offset_mask.weight', 'layer3.conv3.conv_offset_mask.bias'],
        ['layer3.conv3.0.out'],
        dilations=[1,1],
        group=1,
        kernel_shape=[3,3],
        pads=[1,1,1,1],
        strides=[1,1]
    )
    split3 = helper.make_node(
        'Split',
        ['layer3.conv3.0.out'],
        ['split3_1', 'split3_2', 'split3_3'],
        axis=1,
        split=[9,9,9]
    )
    concat3 = helper.make_node(
        'Concat',
        ['split3_1', 'split3_2'],
        ['concat3_out'],
        axis=1,
    )
    sigmoid3 = helper.make_node(
        'Sigmoid',
        ['split3_3'],
        ['sigmoid3_out'],
    )
    dcnV2_3 = helper.make_node(
        'DCNv2', # node name
        ['relu_2_out', 'concat3_out', 'sigmoid3_out', 'layer3.conv3.weight', 'layer3.conv3.bias'], # inputs
        ['feat'], # outputs
        deformable_group = 1,
        dilation = 1,
        groups = 1,
        padding = 1,
        stride = 1
    )


    # ====== remove input ======
    to_be_removed = []
    for i in model.graph.input:
        if i.name in ["feat", "des"]:
            to_be_removed.append(i)
    for i in to_be_removed:
        model.graph.input.remove(i)

    # ====== remove output ======
    to_be_removed = []
    for i in model.graph.output:
        if i.name in ["layer3_input", "det_kpt_inds_norm"]:
            to_be_removed.append(i)
    for i in to_be_removed:
        model.graph.output.remove(i)

    # ====== load paras ======
    s = torch.load('models/aslfeat_v3.pth', map_location='cpu')['model']
    s['bn_1_scale'] = torch.ones(128)
    s['bn_1_B'] = torch.zeros(128)
    s['bn_2_scale'] = torch.ones(128)
    s['bn_2_B'] = torch.zeros(128)
    weights = ['layer3.conv1.0.conv_offset_mask.weight',
               'layer3.conv1.0.conv_offset_mask.bias',
               'layer3.conv1.0.weight', 
               'layer3.conv1.0.bias',
               'bn_1_scale',
               'bn_1_B',
               'layer3.conv1.1.running_mean',
               'layer3.conv1.1.running_var',
               'layer3.conv2.0.conv_offset_mask.weight',
               'layer3.conv2.0.conv_offset_mask.bias',
               'layer3.conv2.0.weight',
               'layer3.conv2.0.bias',
               'bn_2_scale',
               'bn_2_B',
               'layer3.conv2.1.running_mean',
               'layer3.conv2.1.running_var',
               'layer3.conv3.conv_offset_mask.weight',
               'layer3.conv3.conv_offset_mask.bias',
               'layer3.conv3.weight',
               'layer3.conv3.bias']
    for w in weights:
        if w in s:
            print('load {} into init.'.format(w))
            i = numpy_helper.from_array(s[w].numpy())
            i.name = w
            model.graph.initializer.append(i)

    model.graph.node.extend([conv1,split1,concat1,sigmoid1,dcnV2_1,bn_1,relu_1,conv2,split2,concat2,sigmoid2,dcnV2_2,bn_2,relu_2,conv3,split3,concat3,sigmoid3,dcnV2_3,grid_sample])
    onnx.save(model, 'aslfeat.onnx')


def dcn():
    model = onnx.load("models/dcnV2.onnx")
    dcnV2 = helper.make_node(
        'DCNv2', # node name
        ['img', 'offset', 'mask', 'dcn0.weight', 'dcn0.bias'], # inputs
        ['y'], # outputs
        deformable_group = 1,
        dilation = 1,
        groups = 1,
        padding = 1,
        stride = 1
    )

    y = copy.deepcopy(model.graph.output[0])
    y.type.tensor_type.elem_type = 1
    y.type.tensor_type.shape.dim[0].dim_value = 1
    y.type.tensor_type.shape.dim[1].dim_value = 128
    y.type.tensor_type.shape.dim[2].dim_value = 224
    y.type.tensor_type.shape.dim[3].dim_value = 224
    y.name = "y"

    s = torch.load('dcn.pth')
    i = numpy_helper.from_array(s['weight'].numpy())
    i.name = 'dcn0.weight'
    model.graph.initializer.append(i)
    i = numpy_helper.from_array(s['bias'].numpy())
    i.name = 'dcn0.bias'
    model.graph.initializer.append(i)

    # ====== remove output ======
    to_be_removed = []
    for i in model.graph.output:
        if i.name in ["offset", "mask"]:
            to_be_removed.append(i)
    for i in to_be_removed:
        model.graph.output.remove(i)

    model.graph.node.extend([dcnV2])
    model.graph.output.extend([y])
    onnx.save(model, 'dcnV2.onnx')



def correlation():

    # Create one input (ValueInfoProto)
    left = helper.make_tensor_value_info('left', TensorProto.FLOAT, [1,32,40,64])
    right = helper.make_tensor_value_info('right', TensorProto.FLOAT, [1,32,40,64])

    # Create one output (ValueInfoProto)
    corr = helper.make_tensor_value_info('corr', TensorProto.FLOAT, [1,7,40,64])

    # ==========================
    # produce correlation
    # ==========================
    node = helper.make_node(
        'Correlation', # node name
        ['left', 'right'], # inputs
        ['corr'], # outputs
        max_disparity = 3,
        stride = 1
    )

    # ==========================
    # produce graph
    # ==========================
    graph = helper.make_graph(
        [node],
        'correlation_test',
        [left, right],
        [corr],
    )

    # ==========================
    # produce model
    # ==========================
    model = helper.make_model(graph, producer_name='onnx-example')

    onnx.save(model, 'correlation.onnx')
    print('save correlation.onnx successfully!')


def warping():

    # Create one input (ValueInfoProto)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,32,40,64])
    disparity = helper.make_tensor_value_info('disparity', TensorProto.FLOAT, [1,1,40,64])

    # Create one output (ValueInfoProto)
    winput = helper.make_tensor_value_info('winput', TensorProto.FLOAT, [1,32,40,64])

    # ==========================
    # produce warping
    # ==========================
    node = helper.make_node(
        'Warping', # node name
        ['input', 'disparity'], # inputs
        ['winput'], # outputs
    )

    # ==========================
    # produce graph
    # ==========================
    graph = helper.make_graph(
        [node],
        'warping_test',
        [input, disparity],
        [winput],
    )

    # ==========================
    # produce model
    # ==========================
    model = helper.make_model(graph, producer_name='onnx-example')

    onnx.save(model, 'warping.onnx')
    print('save warping.onnx successfully!')


def faststereoslim():
    model = onnx.load('faststereo_sceneflow_vkitti.onnx')

    corr32 = helper.make_node(
        'Correlation', # node name
        ['853', 'right32'], # inputs
        ['corr32'], # outputs
        max_disparity = 5,
        stride = 1
    )

    wright16 = helper.make_node(
        'Warping', # node name
        ['right16', '1298'], # inputs
        ['wright16'], # outputs
    )  

    corr16 = helper.make_node(
        'Correlation', # node name
        ['869', 'wright16'], # inputs
        ['corr16'], # outputs
        max_disparity = 3,
        stride = 1
    )

    wright8 = helper.make_node(
        'Warping', # node name
        ['right8', '1365'], # inputs
        ['wright8'], # outputs
    )  

    corr8 = helper.make_node(
        'Correlation', # node name
        ['893', 'wright8'], # inputs
        ['corr8'], # outputs
        max_disparity = 3,
        stride = 1
    )

    wright4 = helper.make_node(
        'Warping', # node name
        ['right4', '1432'], # inputs
        ['wright4'], # outputs
    )  

    corr4 = helper.make_node(
        'Correlation', # node name
        ['925', 'wright4'], # inputs
        ['corr4'], # outputs
        max_disparity = 3,
        stride = 1
    )

    wright2 = helper.make_node(
        'Warping', # node name
        ['right2', '1499'], # inputs
        ['wright2'], # outputs
    )  

    corr2 = helper.make_node(
        'Correlation', # node name
        ['965', 'wright2'], # inputs
        ['corr2'], # outputs
        max_disparity = 3,
        stride = 1
    )

    for i in range(5):
        model.graph.output.pop()
        model.graph.input.pop()
    model.graph.node.extend([corr32, wright16, corr16, wright8, corr8, wright4, corr4, wright2, corr2])
    onnx.save(model, 'haha.onnx')


def faststereou():
    model = onnx.load('faststereou.onnx')

    corr32 = helper.make_node(
        'Correlation', # node name
        ['943', 'right32'], # inputs
        ['corr32'], # outputs
        max_disparity = 3,
        stride = 1
    )

    wright16 = helper.make_node(
        'Warping', # node name
        ['right16', '1425'], # inputs
        ['wright16'], # outputs
    )  

    corr16 = helper.make_node(
        'Correlation', # node name
        ['959', 'wright16'], # inputs
        ['corr16'], # outputs
        max_disparity = 3,
        stride = 1
    )

    wright8 = helper.make_node(
        'Warping', # node name
        ['right8', '1530'], # inputs
        ['wright8'], # outputs
    )  

    corr8 = helper.make_node(
        'Correlation', # node name
        ['983', 'wright8'], # inputs
        ['corr8'], # outputs
        max_disparity = 3,
        stride = 1
    )

    wright4 = helper.make_node(
        'Warping', # node name
        ['right4', '1635'], # inputs
        ['wright4'], # outputs
    )  

    corr4 = helper.make_node(
        'Correlation', # node name
        ['1015', 'wright4'], # inputs
        ['corr4'], # outputs
        max_disparity = 3,
        stride = 1
    )

    wright2 = helper.make_node(
        'Warping', # node name
        ['right2', '1740'], # inputs
        ['wright2'], # outputs
    )  

    corr2 = helper.make_node(
        'Correlation', # node name
        ['1055', 'wright2'], # inputs
        ['corr2'], # outputs
        max_disparity = 3,
        stride = 1
    )

    for i in range(5):
        model.graph.output.pop()
        model.graph.input.pop()
    model.graph.node.extend([corr32, wright16, corr16, wright8, corr8, wright4, corr4, wright2, corr2])
    onnx.save(model, 'haha.onnx')


def centermask():
    model = onnx.load('models/bbox_scores.onnx')
    graph = model.graph
    init = graph.initializer
    reshape_init_to_remove = []
    for i in init:
        if i.name in ["1294", "1307", "1268", "1281", "1350", "1363", "1324", "1337", "1406",
                      "1419", "1380", "1393", "1462", "1475", "1436", "1449", "1518", "1531",
                      "1493", "1506"]:
            reshape_init_to_remove.append(i)
    for i in reshape_init_to_remove:
        init.remove(i)

    reshape_init_to_add = []
    shape_info = {
    "1294":[1,32,10240],
    "1307":[1,32,10240],
    "1268":[1,32,10240],
    "1281":[1,32,10240],
    "1350":[1,32,2560],
    "1363":[1,32,2560],
    "1324":[1,32,2560],
    "1337":[1,32,2560],
    "1406":[1,32,640],
    "1419":[1,32,640],
    "1380":[1,32,640],
    "1393":[1,32,640],
    "1462":[1,32,160],
    "1475":[1,32,160],
    "1436":[1,32,160],
    "1449":[1,32,160],
    "1518":[1,32,48],
    "1531":[1,32,48],
    "1493":[1,32,48],
    "1506":[1,32,48]
    }

    for k, w in shape_info.items():
        i = numpy_helper.from_array(np.array(w))
        i.name=k
        init.append(i)

    onnx.save(model, "new.onnx")

def add_batchnms():
    model = onnx.load("models/mask_probs.onnx")
    batchnms = helper.make_node(
        'BatchNMS', # node name
        ['bbox', 'scores'], # inputs
        ['nms_out'], # outputs
        shareLocation = True,
        backgroundLabelId = -1,
        numClasses = 80,
        topK = 1000,
        keepTopK = 50,
        scoreThreshold = 0.5,
        iouThreshold = 0.6,
        isNormalized = True,
        clipBoxes = True
    )

    roialign = helper.make_node(
        'PyramidRoiAlign3', # node name
        ['nmsed_boxes', 'num_detections', '1035', '1004', '973'], # inputs
        ['roi_out'], # outputs
        pooled_size = 14
    )

    mask2image = helper.make_node(
        'AdaptMask2Image', # node name
        ['mask_probs', 'nmsed_boxes', 'num_detections'], # inputs
        ['mask'], # outputs
        image_height = 320,
        image_width = 512,
        mask_threshold= 0.5
    )


    mask = copy.deepcopy(model.graph.output[0])
    mask.type.tensor_type.elem_type = 6
    mask.type.tensor_type.shape.dim[0].dim_value = 1
    mask.type.tensor_type.shape.dim[1].dim_value = 1
    mask.type.tensor_type.shape.dim[2].dim_value = 320
    mask.type.tensor_type.shape.dim[3].dim_value = 512
    mask.name = "mask"

    to_be_removed = []
    for i in model.graph.output:
        if i.name in ["bbox", "scores", "mask_probs"]:
            to_be_removed.append(i)
    for i in to_be_removed:
        model.graph.output.remove(i)

    model.graph.input.pop()
    model.graph.input.pop()
    model.graph.output.append(mask)
    model.graph.node.extend([batchnms, roialign, mask2image])
    onnx.save(model, 'centermask.onnx')


def grid_sample():

    # Create one input (ValueInfoProto)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,32,224,224])
    grids = helper.make_tensor_value_info('grids', TensorProto.FLOAT, [1,1,100,2])

    # Create one output (ValueInfoProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1,32,1,100])

    # ==========================
    # produce warping
    # ==========================
    node = helper.make_node(
        'GridSample', # node name
        ['input', 'grids'], # inputs
        ['output'], # outputs
        mode=1,
        padding_mode=0,
        align_corners=True
    )

    # ==========================
    # produce graph
    # ==========================
    graph = helper.make_graph(
        [node],
        'grid_sample',
        [input, grids],
        [output],
    )

    # ==========================
    # produce model
    # ==========================
    model = helper.make_model(graph, producer_name='onnx-example')

    onnx.save(model, 'grid_sample.onnx')
    print('save grid_sample.onnx successfully!')


def superpoint():
    model = onnx.load("haha.onnx")

    grid_sample = helper.make_node(
        'GridSample', # node name
        ['dense_feat', 'keypoints_norm'], # inputs
        ['des'], # outputs
        mode = 1,
        padding_mode = 0,
        align_corners = True
    )

    # ====== remove output ======
    to_be_removed = []
    for i in model.graph.output:
        if i.name in ["dense_feat", "keypoints_norm"]:
            to_be_removed.append(i)
    for i in to_be_removed:
        model.graph.output.remove(i)

    model.graph.input.pop()
    model.graph.node.extend([grid_sample])
    onnx.save(model, 'superpoint.onnx')


def aslfeat_plain():
    model = onnx.load("haha.onnx")

    grid_sample = helper.make_node(
        'GridSample', # node name
        ['dense_feat', 'keypoints_norm'], # inputs
        ['des'], # outputs
        mode = 1,
        padding_mode = 0,
        align_corners = True
    )

    # ====== remove output ======
    to_be_removed = []
    for i in model.graph.output:
        if i.name in ["dense_feat", "keypoints_norm"]:
            to_be_removed.append(i)
    for i in to_be_removed:
        model.graph.output.remove(i)

    model.graph.input.pop()
    model.graph.node.extend([grid_sample])
    onnx.save(model, 'aslfeat_plain.onnx')