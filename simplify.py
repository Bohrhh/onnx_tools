# post process of onnx model
# Author kmlee
# Date 20200504

import os
import onnx
import copy
import numpy as np
import onnxruntime
import onnx.optimizer
from onnx import helper
from base import OnnxModel
from onnx import numpy_helper
from collections import OrderedDict


class Optimizer(object):
    def __init__(self):
        pass

    def modify_model(self, model):
        """
        some nodes are not supported by onnxruntime
        replace:
            ConvTranspose3d -- Conv3d+Resize nearest
            Resize linear -- Resize nearest 
        
        return:
            model -- modified model
            removed_nodes -- []
            added_nodes -- []
        """
        removed_nodes = []
        added_nodes = []
        nodes = model.graph.node
        for n in nodes:
            if n.op_type == 'Resize' and n.attribute[2].s == b'linear':
                removed_nodes.append(copy.deepcopy(n))
                n.attribute[2].s = b'nearest'
                n.attribute[0].s = b'asymmetric'
                added_nodes.append(n)

        return removed_nodes, added_nodes


    def add_initializers_into_inputs(self, model):
        for x in model.graph.initializer:
            input_names = [x.name for x in model.graph.input]
            if x.name not in input_names:
                shape = onnx.TensorShapeProto()
                for dim in x.dims:
                    shape.dim.extend([onnx.TensorShapeProto.Dimension(dim_value=dim)])
                model.graph.input.extend(
                    [onnx.ValueInfoProto(name=x.name,
                                         type=onnx.TypeProto(tensor_type=onnx.TypeProto.Tensor(elem_type=x.data_type,
                                                                                            shape=shape)))])
        return model

    def normal(self, model, skip_fuse_bn):
        """
        paremeters:
            model -- The onnx model
            skip_fuse_bn -- whether skip fuse bn
        return:
            The optimized onnx model
        Before simplifying, use this method to generate value_info, which is used in `forward_all`
        After simplifying, use this method to fold constants generated in previous step into initializer,
        and eliminate unused constants.
        """

        # Due to a onnx bug, https://github.com/onnx/onnx/issues/2417, we need to add missing initializers into inputs
        onnx.checker.check_model(model)
        input_num = len(model.graph.input)
        model = self.add_initializers_into_inputs(model)
        onnx.helper.strip_doc_string(model)
        onnx.checker.check_model(model)
        optimizers_list = ['eliminate_deadend', 'eliminate_identity', 'eliminate_nop_dropout',
                                                'eliminate_nop_monotone_argmax', 'eliminate_nop_pad',
                                                'extract_constant_to_initializer', 'eliminate_unused_initializer',
                                                'eliminate_nop_transpose', 'fuse_add_bias_into_conv', 
                                                # https://github.com/daquexian/onnx-simplifier/issues/31
                                                # 'fuse_consecutive_concats',
                                                'fuse_consecutive_log_softmax',
                                                'fuse_consecutive_reduce_unsqueeze', 'fuse_consecutive_squeezes',
                                                'fuse_consecutive_transposes', 'fuse_matmul_add_bias_into_gemm',
                                                'fuse_pad_into_conv', 'fuse_transpose_into_gemm']
        if not skip_fuse_bn:
            optimizers_list.append('fuse_bn_into_conv')

        model = onnx.optimizer.optimize(model, optimizers_list, fixed_point=True)
        del model.graph.input[input_num:]
        onnx.checker.check_model(model)
        return model
    
    def get_constant_nodes(self, model):
        const_nodes = []
        const_tensors = [x.name for x in model.graph.initializer]
        const_tensors.extend([node.output[0]
                            for node in model.graph.node if node.op_type == 'Constant'])
        # If one of the input of a node is produced (directly or indirectly) by nms,
        # we consider the output of this node doesn't have constant shape,
        # so we do not simplify a such node even if the node is Shape op
        tensors_nms = []
        for node in model.graph.node:
            if any(x in tensors_nms for x in node.input):
                tensors_nms.extend(node.output)
            elif node.op_type == 'Shape':
                const_nodes.append(node)
                const_tensors.extend(node.output)
            elif node.op_type == 'NonMaxSuppression':
                tensors_nms.extend(node.output)
            elif all([x in const_tensors for x in node.input]):
                const_nodes.append(node)
                const_tensors.extend(node.output)
        return copy.deepcopy(const_nodes)

    def get_value_info_all(self, model, name):
        for v in model.graph.value_info:
            if v.name == name:
                return v

        for v in model.graph.input:
            if v.name == name:
                return v

        for v in model.graph.output:
            if v.name == name:
                return v

        return None

    def get_shape(self, model, name):
        """
        Note: This method relies on onnx shape inference, which is not reliable. So only use it on input or output tensors
        """
        v = self.get_value_info_all(model, name)
        if v is not None:
            return [dim.dim_value for dim in v.type.tensor_type.shape.dim]
        raise RuntimeError('Cannot get shape of "{}"'.format(name))

    def get_elem_type(self, model, name):
        v = self.get_value_info_all(model, name)
        assert v is not None
        sizes = (None, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, str, np.bool,
                np.float16, np.float64, np.uint32, np.uint64, np.complex64, np.complex128, np.float16)
        return sizes[v.type.tensor_type.elem_type]


    def generate_rand_input(self, model):
        input_names = list(set([ipt.name for ipt in model.graph.input]) -
                        set([x.name for x in model.graph.initializer]))
        

        full_input_shapes = {ipt: self.get_shape(model, ipt) for ipt in input_names}
        for key in full_input_shapes:
            if np.prod(full_input_shapes[key]) <= 0:
                raise RuntimeError(
                    'The shape of input "{}" has dynamic size, '
                    'please determine the input size manually'.format(key))

        inputs = {ipt: np.array(np.random.rand(*full_input_shapes[ipt]),
                                dtype=self.get_elem_type(model, ipt)) for ipt in input_names}
        return inputs

    def forward(self, model, inputs=None):
        # runtime config
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel(0)
        sess_options.log_severity_level = 3
        sess = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=sess_options, providers=['CPUExecutionProvider'])
        # prepare inputs
        if inputs is None:
            inputs = self.generate_rand_input(model)

        outputs = [x.name for x in sess.get_outputs()]
        run_options = onnxruntime.RunOptions()
        run_options.log_severity_level = 3
        result = OrderedDict(zip(outputs, sess.run(outputs, inputs, run_options=run_options)))
        return result


    def forward_for_node_outputs(self, model, nodes):
        model = copy.deepcopy(model)

        # Add features to output in pb, so that ONNX Runtime will output them.
        for node in nodes:
            for output in node.output:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])

        result = self.forward(model)
        return result

    def insert_elem(self, repeated_container, index, element):
        repeated_container.extend([repeated_container[-1]])
        for i in reversed(range(index + 1, len(repeated_container) - 1)):
            repeated_container[i].CopyFrom(repeated_container[i - 1])
        repeated_container[index].CopyFrom(element)


    def eliminate_const_nodes(self, model, const_nodes, res):
        """
        :param model: the original onnx model
        :param const_nodes: const nodes detected by `get_constant_nodes`
        :param res: The dict containing all tensors, got by `forward_all`
        :return: the simplified onnx model. Redundant ops are all removed.
        """
        for i, node in enumerate(model.graph.node):
            if node in const_nodes:
                for output in node.output:
                    new_node = copy.deepcopy(node)
                    new_node.name = "node_" + output
                    new_node.op_type = 'Constant'
                    new_attr = onnx.helper.make_attribute(
                        'value',
                        onnx.numpy_helper.from_array(res[output], name=output)
                        )
                    del new_node.input[:]
                    del new_node.attribute[:]
                    del new_node.output[:]
                    new_node.output.extend([output])
                    new_node.attribute.extend([new_attr])
                    # self.insert_elem(model.graph.node, i + 1, new_node)
                    model.graph.node.insert(i+1, new_node)
                del model.graph.node[i]

        return model

    def check(self, model_opt, model_ori, n_times):
        """
        Warning: Some models (e.g., MobileNet) may fail this check by a small magnitude.
        Just ignore if it happens.
        :param input_shapes: Shapes of generated random inputs
        :param model_opt: The simplified ONNX model
        :param model_ori: The original ONNX model
        :param n_times: Generate n random inputs
        """
        onnx.checker.check_model(model_opt)
        for i in range(n_times):
            print("Checking {}/{}...".format(i, n_times))
            rand_input = self.generate_rand_input(model_opt)
            res_opt = self.forward(model_opt, inputs=rand_input)
            res_ori = self.forward(model_ori, inputs=rand_input)

            for name in res_opt.keys():
                if not np.allclose(res_opt[name], res_ori[name], rtol=1e-4, atol=1e-5):
                    print("Tensor {} changes after simplifying. The max diff is {}.".format(
                        name, np.max(np.abs(res_opt[name] - res_ori[name]))))
                    print("Note that the checking is not always correct.")
                    print("After simplifying:")
                    print(res_opt[name])
                    print("Before simplifying:")
                    print(res_ori[name])
                    print("----------------")
                    return False
        return True

    def __call__(self, base, check=3, skip_fuse_bn=True):
        """
        base -- OnnxModel
        """
        model_original = base.model
        model = copy.deepcopy(base.model)
        
        # normal optimization
        model = self.normal(model, skip_fuse_bn)
        onnx.checker.check_model(model)
        
        # replace nodes which are not supported by onnxruntime
        removed_nodes, added_nodes = self.modify_model(model)
        model_replaced = copy.deepcopy(model)


        #simplify constant nodes, e.g. shape
        const_nodes = self.get_constant_nodes(model)
        res = self.forward_for_node_outputs(model, const_nodes)
        const_nodes = [node for node in const_nodes if node.output[0] in res]
        model = self.eliminate_const_nodes(model, const_nodes, res)

        # check model
        if check>0:
            self.check(model, model_replaced, check)

        # restore the nodes which are not supported by onnxruntime
        for n in added_nodes:
            model.graph.node.remove(n)
        model.graph.node.extend(removed_nodes)
        base.model = model
        base.sort()

        # normal optimization
        model = self.normal(base.model, skip_fuse_bn)
        onnx.checker.check_model(model)
        
        base.model = model
        return base

def main(model, output_file, check=3):
    """
    parameters:
        model -- onnx model filename or onnx.ModelProto
        output_file -- output onnx model filename
        check -- check times, check whether the optimized model 
            is equal to the original
    """

    #-----------------------------------------------
    # prepare model
    #-----------------------------------------------
    model = OnnxModel(model)

    #-----------------------------------------------
    # working procedur
    #-----------------------------------------------
    opti = Optimizer()
    # quant = Quantizer()


    #-----------------------------------------------
    # produce
    #-----------------------------------------------
    model.preprocess()
    model = opti(model, check=check)


    #-----------------------------------------------
    # output
    #-----------------------------------------------
    model.save(output_file)