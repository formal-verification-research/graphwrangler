import tensorflow as tf
import sys

def load_pb(path_to_pb):
    with open(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def

def get_inputs(graph_def):
    return [node for node in graph_def.node if node.op == "Placeholder"]

def get_outputs(graph_def):
    node_dict = {}
    non_leaf_names = []
    for node in graph_def.node:
        node_dict[node.name] = [node, True] # True --> isLeaf
    for node in graph_def.node:
        for node_input in node.input:
            node_dict[node_input][1] = False # Outputs to something else
    leaf_nodes = []
    for node_pair in node_dict.values():
        if node_pair[1]:
            leaf_nodes.append(node_pair[0])
    return leaf_nodes

def check_with_user(prompt, option_nodes, allow_none=None, fallback_list=None, outfile=sys.stdout):
    if allow_none is None:
        allow_none = not fallback_list
    if len(option_nodes) == 0:
        return None
    elif len(option_nodes) == 1:
        return option_nodes[0]
    outfile.write("  " + prompt + " [PICK ONE]\n")
    i = 0
    for option in option_nodes:
        outfile.write("    " + str(i) + ". " + option.name + "\n")
        i += 1
    if allow_none or fallback_list:
        outfile.write("    " + str(i) + ". None of the above\n")
    outfile.write("  Type a number: ")
    idx = int(input())
    if fallback_list and idx == len(option_nodes):
        return check_with_user(prompt, fallback_list, allow_none=allow_none, outfile=outfile)
    elif allow_none and idx == len(option_nodes):
        return None
    else:
        return option_nodes[idx]

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.stderr.write("Usage: python [-i] " + sys.argv[0] + " <graph.pb>\n")
    else:
        sys.stderr.write("Loading " + sys.argv[1] + "\n")
        graph = load_pb(sys.argv[1])

        inputs = get_inputs(graph)
        outputs = get_outputs(graph)

        input_node = check_with_user("Which of these is the input node?", inputs, fallback_list=graph.node, outfile=sys.stderr) 
        output_node = check_with_user("Which of these is the output node?", outputs, fallback_list=graph.node, outfile=sys.stderr)

        if input_node is None:
            sys.stdout.write("\n")
            sys.stderr.write("No input node found!\n")
        else:
            sys.stdout.write(input_node.name + "\n")

        if output_node is None:
            sys.stdout.write("\n")
            sys.stderr.write("No output node found!\n")
        else:
            sys.stdout.write(output_node.name + "\n")

