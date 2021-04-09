import tensorflow as tf
import sys
from load import load_pb

def get_inputs(graph_def):
    return [node for node in graph_def.node if node.op == "Placeholder"]

def get_outputs(graph_def):
    node_dict = {}
    non_leaf_names = []
    for node in graph_def.node:
        node_dict[node.name] = [node, True] # True --> isLeaf
    for node in graph_def.node:
        for node_input in node.input:
            node_dict[node_input.split(":")[0]][1] = False # Outputs to something else
    leaf_nodes = []
    for node_pair in node_dict.values():
        if node_pair[1]:
            leaf_nodes.append(node_pair[0])
    return leaf_nodes

def check_with_user(prompt, option_nodes, allow_none=None, fallback_list=None, outfile=sys.stdout):
    if allow_none is None:
        allow_none = not fallback_list
    if len(option_nodes) == 0:
        if fallback_list:
            return check_with_user(prompt, fallback_list, allow_none=allow_none, outfile=outfile)
        else:
            return None
    elif len(option_nodes) == 1 and not (allow_none or fallback_list):
        return option_nodes[0]
    else:
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

def pick_nodes(type_dict, allow_none=None, fallback_list=None, outfile=sys.stdout):
    ret_dict = {}
    for k, v in type_dict.items():
        ret_dict[k] = check_with_user("Which of these is the " + k + " node?", v, allow_none=allow_none, fallback_list=fallback_list, outfile=outfile)
    return ret_dict;

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.stderr.write("Usage: python [-i] " + sys.argv[0] + " <graph.pb>\n")
    else:
        sys.stderr.write("Loading " + sys.argv[1] + "\n")
        graph = load_graph(sys.argv[1])
        graph_def = graph.as_graph_def()

        ret_dict = pick_nodes({
            "input": get_inputs(graph_def),
            "output": get_outputs(graph_def),
        }, fallback_list=graph_def.node, outfile=sys.stderr)

        for k, v in ret_dict.items():
            if v is None:
                sys.stdout.write("\n")
                sys.stderr.write("No " + k + " node found!\n")
            else:
                sys.stdout.write(v.name + "\n")
