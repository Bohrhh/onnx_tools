import copy
import onnx
import numpy as np
from onnx import helper
from onnx import numpy_helper

QUANTINODES = ['Conv', 'BatchNormalization', 'Relu', 'ConvTranspose', 'Concat', 'Add', 'MaxPool', 'LeakyRelu']
ONNXTYPE2NUMPY = {
    onnx.TensorProto.FLOAT : np.float32,
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.INT32 : np.int32,
    onnx.TensorProto.INT64 : np.int64,
}

INF = 1000000


class OnnxModel(object):

    def __init__(self, model):
        if isinstance(model, str):
            model = onnx.load(model)
        self.model = model

    def add_node(self, node):
        self.model.graph.node.extend([node])
        
    def get_input2nodes(self):
        # mapping input_name->nodes
        input2nodes = {}
        for n in self.model.graph.node:
            for i in n.input:
                if input2nodes.get(i):
                    input2nodes[i].append(n)
                else:
                    input2nodes[i] = []
                    input2nodes[i].append(n)
        return input2nodes

    def get_output2node(self):
        # mapping output_name->node
        output2node = {}
        for n in self.model.graph.node:
            for o in n.output:
                output2node[o] = n
        return output2node

    def get_name2init(self):
        name2init = {}
        for i in self.model.graph.initializer:
            name2init[i.name] = i
        return name2init

    def get_initializer(self, name):
        for tensor in self.model.graph.initializer:
            if tensor.name == name:
                return tensor
        return None

    @staticmethod
    def replace_node_input(node, old_input_name, new_input_name):
        assert isinstance(old_input_name, str) and isinstance(new_input_name, str)
        for j in range(len(node.input)):
            if node.input[j] == old_input_name:
                node.input[j] = new_input_name

    def replace_input_of_all_nodes(self, old_input_name, new_input_name):
        for node in self.model.graph.node:
            OnnxModel.replace_node_input(node, old_input_name, new_input_name)

    @staticmethod
    def replace_node_output(node, old_output_name, new_output_name):
        assert isinstance(old_output_name, str) and isinstance(new_output_name, str)
        for j in range(len(node.output)):
            if node.output[j] == old_output_name:
                node.output[j] = new_output_name

    def replace_output_of_all_nodes(self, old_output_name, new_output_name):
        for node in self.model.graph.node:
            OnnxModel.replace_node_output(node, old_output_name, new_output_name)

    def sort(self):
        """
        sort nodes topologically 
        """
        nodes = self.model.graph.node
        name2info = {}

        for i,n in enumerate(nodes):
            if not n.name:
                n.name = "sort_"+str(i)
            name2info[n.name] = {'f':-1, 'd':-1, 'color':'white'}

        input2nodes = self.get_input2nodes()

        def dfs_visit(nodes, n, time, name2info, input2nodes):
            time[0]+=1
            name2info[n.name]['d'] = time[0]
            name2info[n.name]['color'] = 'gray'
            adj = []
            for o in n.output:
                if o in input2nodes:
                    adj += input2nodes[o]
            for v in adj:
                if name2info[v.name]['color'] == 'white':
                    dfs_visit(nodes, v, time, name2info, input2nodes)
            name2info[n.name]['color']='black'
            time[0]+=1
            name2info[n.name]['f'] = time[0]

        # sort
        time = [0]
        for n in nodes:
            if name2info[n.name]['color']=='white':
                dfs_visit(nodes, n, time, name2info, input2nodes)

        # change model.graph.node
        num_nodes = len(nodes)
        for i in range(num_nodes-1):
            for j in range(0,num_nodes-i-1):
                idx1 = name2info[nodes[j].name]['f']
                idx2 = name2info[nodes[j+1].name]['f']
                assert idx1>0 and idx2>0
                if idx1<idx2:
                    n = nodes.pop(j)
                    nodes.insert(j+1, n)

    def remove_subgraph(self, node):
        """
        remove subgraph which contains the input node
        Args:
            node -- node or node's name
        """
        nodes = self.model.graph.node
        if isinstance(node, onnx.NodeProto):
            toBeRemoved = [node]
        elif isinstance(node, str):
            for n in nodes:
                if n.name == node:
                    toBeRemoved = [n]
                    break

        # mapping input_name->nodes
        input2nodes = self.get_input2nodes()
        # mapping output_name->nodes
        output2node = self.get_output2node()
        # mapping name->model_input
        name2input = dict(zip([i.name for i in self.model.graph.input], self.model.graph.input))
        # mapping name->model_output
        name2output = dict(zip([i.name for i in self.model.graph.output], self.model.graph.output))
        # init names
        init_names = [i.name for i in self.model.graph.initializer]

        while len(toBeRemoved):
            n = toBeRemoved.pop(0)

            # rm node
            nodes.remove(n)

            for o in n.output:
                out_nodes = input2nodes.get(o)
                if out_nodes is not None:
                    for out_node in out_nodes:
                        if out_node in nodes and out_node not in toBeRemoved:
                            toBeRemoved.append(out_node)
                if o in name2output and name2output[o] in self.model.graph.output:
                    self.model.graph.output.remove(name2output[o])

            for i in n.input:
                if i in init_names:
                    continue
                input_node = output2node.get(i)
                if input_node is not None and input_node in nodes and input_node not in toBeRemoved:
                    toBeRemoved.append(input_node)
                if i in name2input:
                    out_nodes = input2nodes.get(i)
                    if out_nodes is not None:
                        for out_node in out_nodes:
                            if out_node in nodes and out_node not in toBeRemoved:
                                toBeRemoved.append(out_node)
                    if name2input[i] in self.model.graph.input:
                        self.model.graph.input.remove(name2input[i])

    def clean_unused_init(self):
        node_inputs = []
        for n in self.model.graph.node:
            node_inputs += n.input
        init_to_be_removed = []
        for i in self.model.graph.initializer:
            if i.name not in node_inputs:
                init_to_be_removed.append(i)
        for i in init_to_be_removed:
            self.model.graph.initializer.remove(i)

    def add_hook_nodes(self, quantiNodes=False):
        """
        add ReduceMax and ReduceMin node to compute dynamic range of each tensors
        and add them to graph output

        parameters:
            quantiNodes -- nodes for adding output hooks, 
                           if False for all nodes,
                           if True for QUANTINODES
        """
        extend_nodes = []
        for n in self.model.graph.node:
            if n.op_type not in ["Constant", "Gather"]:
                if not quantiNodes or n.op_type in QUANTINODES:
                    for o in n.output:
                        minNode = helper.make_node(
                            'ReduceMin',
                            inputs=[o],
                            outputs=[o+'_reduceMin'],
                            keepdims=0,
                            name=o+'_reduceMin_hook_node'
                        )
                        maxNode = helper.make_node(
                            'ReduceMax',
                            inputs=[o],
                            outputs=[o+'_reduceMax'],
                            keepdims=0,
                            name=o+'_reduceMax_hook_node'
                        )
                        self.model.graph.output.extend([
                            helper.ValueInfoProto(name=o+'_reduceMin'),
                            helper.ValueInfoProto(name=o+'_reduceMax')
                        ])
                        extend_nodes += [minNode, maxNode]

        self.model.graph.node.extend(extend_nodes)

    def remove_hook_nodes(self):

        # remove hook nodes
        hook_nodes = []
        for n in self.model.graph.node:
            if 'hook_node' in n.name:
                hook_nodes.append(n)
        for hn in hook_nodes:
            self.model.graph.node.remove(hn)

        # remove reduce output
        reduce_out = []
        for o in self.model.graph.output:
            if '_reduceM' in o.name:
                reduce_out.append(o)
        for ro in reduce_out:
            self.model.graph.output.remove(ro)

    def remove_initializer_from_input(self):
        if self.model.ir_version < 4:
            print(
                'Model with ir_version below 4 requires to include initilizer in graph input'
            )
            return

        inputs = self.model.graph.input
        name_to_input = {}
        for input in inputs:
            name_to_input[input.name] = input

        for initializer in self.model.graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])

    def clipAdjusted(self):
        # modify Clip node
        for i,n in enumerate(self.model.graph.node):
            if n.op_type == "Clip":
                attr_len = len(n.attribute)
                attr = [n.attribute.pop() for i in range(attr_len)]
                attr = {a.name:a for a in attr}
                npdtype = ONNXTYPE2NUMPY[1]
                if len(attr)==0:
                    if n.input[1] == '':
                        mintensor = numpy_helper.from_array(np.array(-INF).astype(npdtype))
                        mintensor.name = 'clipmin'+str(i)
                        n.input[1] = 'clipmin'+str(i)
                        self.model.graph.initializer.append(mintensor)
                    if n.input[2] == '':
                        maxtensor = numpy_helper.from_array(np.array(INF).astype(npdtype))
                        maxtensor.name = 'clipmax'+str(i)
                        n.input[2] = 'clipmax'+str(i)
                        self.model.graph.initializer.append(maxtensor)
                else:
                    n.input.extend(['clipmin'+str(i), 'clipmax'+str(i)])
                    dtype = 1
                    if 'min' in attr:
                        wmin = attr['min'].f
                        dtype = attr['min'].type
                    else:
                        wmin = -INF
                    
                    if 'max' in attr:
                        wmax = attr['max'].f
                        dtype = attr['max'].type
                    else:
                        wmax = INF
                    npdtype = ONNXTYPE2NUMPY[dtype]

                    mintensor = numpy_helper.from_array(np.array(wmin).astype(npdtype))
                    mintensor.name = 'clipmin'+str(i)
                    maxtensor = numpy_helper.from_array(np.array(wmax).astype(npdtype))
                    maxtensor.name = 'clipmax'+str(i)

                    self.model.graph.initializer.append(mintensor)
                    self.model.graph.initializer.append(maxtensor)

                    
    def selfAdjusted(self):
        """
        some onnx nodes generated by torch is not compatable with onnxruntime
        nodes list:[Clip, ]
        """
        self.clipAdjusted()

    def scissors(self, out1, out2, add_output=True, n1name=None, n2name=None):
        """
        Cut the connection between nodes. Out1 is used for finding node1 and out2 is used for finding node2.
        The connection is from node1 to node2.

        Args:
            out1 -- node1's output name
            out2 -- node2's output name
            add_output -- whether add node1's output to graph output
            n1name -- set node1's output name after the connection has been cut, default "{original}"
            n2name -- set node2's input name after the connection has been cut, default "{original}_cut"
        """

        output2node = self.get_output2node()
        node1 = output2node[out1]
        node2 = output2node[out2]
        outputName = [o.name for o in self.model.graph.output]
        for j,i in enumerate(node2.input):
            if i == out1:
                node2.input[j] = i+"_cut"
                if n2name is not None:
                    node2.input[j] = n2name
                if add_output:
                    if out1 in outputName and n1name is None:
                        pass
                    elif out1 in outputName and n1name is not None:
                        for o in self.model.graph.output:
                            if o.name == out1:
                                o.name = n1name
                    elif out1 not in outputName and n1name is None:
                        self.model.graph.output.append(helper.ValueInfoProto(name=out1))
                    else:
                        assert len(node1.output)==1, "Illegal node. The length of node's output should be one!"
                        node1.output[0] = n1name
                        self.model.graph.output.append(helper.ValueInfoProto(name=n1name))                       
                break

            assert j < len(node2.input)-1, "no connection between these two nodes!" 

    def convert_model_float32_to_float16(self):
        graph = self.model.graph
        initializers = graph.initializer

        for initializer in initializers:
            if initializer.data_type == 1:
                initializer.CopyFrom(
                    numpy_helper.from_array(numpy_helper.to_array(initializer).astype(np.float16), initializer.name))

        for node in graph.node:
            if node.op_type in ['Constant', 'ConstantOfShape']:
                for att in node.attribute:
                    if att.name == 'value' and att.t.data_type == 1:
                        att.CopyFrom(
                            helper.make_attribute(
                                "value", numpy_helper.from_array(numpy_helper.to_array(att.t).astype(np.float16))))
            if node.op_type == 'Cast':
                for att in node.attribute:
                    if att.name == 'to' and att.i == 1:
                        att.CopyFrom(helper.make_attribute("to", 10))

        for input_value_info in graph.input:
            if input_value_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
                initializer = self.get_initializer(input_value_info.name)
                if initializer is not None:  # for compatibility for old converter/exporter
                    input_value_info.type.tensor_type.elem_type = 10
                else:
                    cast_input = input_value_info.name
                    cast_output = input_value_info.name + '_float16'
                    self.replace_input_of_all_nodes(cast_input, cast_output)
                    cast_node = helper.make_node('Cast', inputs=[cast_input], outputs=[cast_output])
                    cast_node.attribute.extend([helper.make_attribute("to", int(onnx.TensorProto.FLOAT16))])
                    self.add_node(cast_node)

        for output_value_info in graph.output:
            if output_value_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
                cast_input = output_value_info.name + '_float16'
                cast_output = output_value_info.name
                self.replace_output_of_all_nodes(cast_output, cast_input)
                self.replace_input_of_all_nodes(cast_output, cast_input)
                cast_node = helper.make_node('Cast', inputs=[cast_input], outputs=[cast_output])
                cast_node.attribute.extend([helper.make_attribute("to", int(onnx.TensorProto.FLOAT))])
                self.add_node(cast_node)


    def preprocess(self):
        self.remove_initializer_from_input()
        self.selfAdjusted()
        self.clean_unused_init()
        self.sort()

    def save(self, output_file):
        onnx.save(self.model, output_file)
        print("# export model to {}".format(output_file))