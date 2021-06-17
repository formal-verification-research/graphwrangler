import tensorflow as tf
import numpy as np

from parse_nodes import get_inputs
from load import load_graph

def get_tensor():
    pass

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: python " + sys.argv[0] + " <graph.pb> <input_node>\n")
    else:
        sys.stderr.write("Loading " + sys.argv[1] + "\n")
        graph = load_graph(sys.argv[1])
        graph_def = graph.as_graph_def()

        with graph.as_default():
            node = graph.get_tensor_by_name(sys.argv[2] + ":0")
            print(str(node.shape))



