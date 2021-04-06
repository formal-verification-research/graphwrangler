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

def check_with_user(prompt, option_nodes, outfile=sys.stdout):
    if len(option_nodes) == 0:
        return None
    elif len(option_nodes) == 1:
        return option_nodes[0]
    outfile.write("  " + prompt + " [PICK ONE]\n")
    i = 0
    for option in option_nodes:
        outfile.write("    " + str(i) + ". " + option.name + "\n")
        i += 1
    outfile.write("  Type a number: ")
    return option_nodes[int(input())]

if __name__ == "__main__":
    sys.stderr.write("Loading " + sys.argv[1] + "\n")
    graph = load_pb(sys.argv[1])

    inputs = get_inputs(graph)
    outputs = get_outputs(graph)

    input_node = check_with_user("Which of these is the input node?", inputs, outfile=sys.stderr) 
    output_node = check_with_user("Which of these is the output node?", outputs, outfile=sys.stderr)

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

