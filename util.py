import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run PGQL.")

    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--lr1', type=float, default=0.0001)                    
    parser.add_argument('--lr2', type=float, default=0.0001)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--game', type=str, default='None') 
    return parser.parse_args()

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(image):
    image = np.reshape(image,[np.prod(image.shape)]) / 255.0
    return image