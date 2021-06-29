#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import sys

from parse_nodes import pick_nodes, get_inputs, get_outputs
from load import load_graph

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: python " + sys.argv[0] + " <graph.pb> <out_graph.pb> [disallow_prompt_user : bool [negative : bool]]\n")
    else:
        sys.stderr.write("Loading " + sys.argv[1] + "\n")
        graph = load_graph(sys.argv[1])
        graph_def = graph.as_graph_def()

        if len(sys.argv) > 3 and sys.argv[3].lower() in ["true", "t", "1"]:
            allow_prompt_user = False
        else:
            allow_prompt_user = True

        nodes = pick_nodes({
            "input": get_inputs(graph_def),
            "logits": get_outputs(graph_def),
        }, fallback_list=graph_def.node, outfile=sys.stderr, allow_prompt_user=allow_prompt_user)
        
        with graph.as_default():
            logits = graph.get_tensor_by_name(nodes["logits"].name + ":0")
            if len(sys.argv) > 4 and sys.argv[4].lower() in ["true", "t", "1"]:
                logits = tf.math.negative(logits, name="negative")
            x = graph.get_tensor_by_name(nodes["input"].name + ":0")
            y = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name='y_label')
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name='loss_fun')
            grad = tf.gradients(loss, x)
            grad_out = tf.identity(grad, name='gradient_out')
            tf.io.write_graph(graph, '.', sys.argv[2], as_text=False)

