from parse_nodes import pick_nodes, get_inputs, get_outputs
from test import load_graph
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: python " + sys.argv[0] + " <graph.pb> <out_graph.pb>\n")
    else:
        sys.stderr.write("Loading " + sys.argv[1] + "\n")
        graph = load_graph(sys.argv[1])
        graph_def = graph.as_graph_def()

        nodes = pick_nodes({
            "input": get_inputs(graph_def),
            "logits": get_outputs(graph_def),
        }, fallback_list=graph_def.node, outfile=sys.stderr)

        logits = graph.get_tensor_by_name(nodes["logits"])
        x = graph.get_tensor_by_name(nodes["input"])
        y = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name='y_label')
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name='loss_fun')
        grad = tf.gradients(loss, x)
        grad_out = tf.identity(grad, name='gradient_out')
        tf.io.write_graph(graph, '.', sys.argv[2], as_text=False)

