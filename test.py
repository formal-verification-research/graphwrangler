import tensorflow as tf
sess = tf.compat.v1.Session()

def modify_graph(graph_filename, node_name, f):
    with tf.Graph().as_default():
        gd = tf.GraphDef()
        with open(graph_filename, 'rb') as f:
            gd.MergeFromString(f.read())
        my_tensor = tf.import_graph_def(gd, name='', return_elements=node_name)
        f(gd, node_name)
        tf.train.write_graph(tf.get_default_graph(), '.', graph_filename[:-2] + ".mod.pb", as_text=False)


if __name__ == '__main__':
    import sys
    with tf.Graph().as_default():
        gd = tf.compat.v1.GraphDef()
        with open('mnist.pb', 'rb') as f:
            gd.MergeFromString(f.read())
        tf.compat.v1.import_graph_def(gd, name='')
        
        logits = tf.compat.v1.get_default_graph().get_tensor_by_name('Plus214_Output_0:0')
        x = tf.compat.v1.get_default_graph().get_tensor_by_name('Input3:0')
        y = tf.compat.v1.placeholder(tf.float32, shape=(None, 10), name='y_label')
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name='loss_func')
        grad = tf.gradients(loss, x)
        grad_out = tf.identity(grad, name='gradient_out')
        tf.io.write_graph(tf.compat.v1.get_default_graph(), '.', 'mnist_mod.pb', as_text=False)

