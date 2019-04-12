from mpi4py import MPI
import numpy as np
import socket
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data_dir = 'data/'
batch_size = 100
comm = MPI.COMM_WORLD
size = comm.Get_size() # qde de tarefas MPI
rank = comm.Get_rank() # identificador da tarefa (rank)

'''
These four functions help to define the CNN model to be trained. 
Instead of using just a simple softmax function with a neural net, this code will 
implement a multilayer convolutional neural network. In order to create this model,
we will need many weights and biases. Since we're  using ReLU (Rectified Linear Unit) 
neurons, it is also good practice to initialize them with a slightly positive 
initial bias to avoid "dead neurons". (https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks)
Instead of doing this repeatedly while we build the model, weight_variable() and bias_variable()
can be called to return a variable or constant based on the shape provided.
'''
def weight_variable(shape):
  '''
	tf.truncated_normal returns random values from a truncated normal distribution. 
	The generated values follow a normal distribution with specified mean and standard 
	deviation, except that values whose magnitude is more than 2 standard deviations
	from the mean are dropped and re-picked.
	'''
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

'''
These next two functions perform convolution and pooling. Convolution is the process of training on portions of 
an image, and applying the features learned from that portion to the entire image. The stride indicates 
how many pixels to shift over when applying this 'mask' to the entire image. In our case, we use the 
default of 1. Pooling is a sample based discretization process. The objective is to down-sample the input 
into bins.
'''
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



def gather_jobs_with_mpi():
    # hostname em formato ubyte de cada node
    hostname = list(socket.gethostname().encode('utf8'))

    # Comunicador MPI 
    numDataPerRank = len(hostname) # tamanho do buffer de bytes do nome

    # buffers a enviar
    sendbuf = np.array(hostname, dtype=np.uint8) # transforma buffer bytes em array
    sendbuf2 = np.array([rank], dtype=np.uint8) # buffer com idenficador da tarefa

    # buffers a receber 
    recvbuf = np.empty(numDataPerRank*size, dtype=np.uint8)
    recvbuf2 = np.empty(size, dtype=np.uint8)

    # envia para todos as tarefas MPI (todas sabem o nome de todas)
    comm.Allgather(sendbuf, recvbuf)
    comm.Allgather(sendbuf2, recvbuf2)

    # Transforma novamente o array de bytes nos nomes dos hosts
    hosts = np.split(recvbuf, size)
    hosts = [bytes(list(i)).decode('utf8') for i in hosts]

    # Identificadores de cada tarefa
    ranks = recvbuf2

    # Junta os host name dos nos com a porta 222 mais o identificador da tarefa
    jobs = {'worker': ['%s:222%s' % (i, j) for (i,j) in zip(hosts, ranks) if j != 0]}
    jobs['ps'] = ['%s:2220' % (hosts[0])]

    return jobs

jobs = gather_jobs_with_mpi()
job_name = 'worker'
# Se for a tarefa com id 0
if rank == 0:
    job_name = 'ps'
    import json
    f = open("saida.out", "w+")
    f.write(json.dumps(jobs)) # Salvar os dados das tarefas em json
    f.close()
    
    
# Criar o cluster de jobs
cluster = tf.train.ClusterSpec(jobs)

# Iniciar um servidor do tensorflow para cada tarefa
server = tf.train.Server(cluster, job_name=job_name, task_index=rank)
if job_name == 'ps':
    server.join()
else:
    # Executar em cada device
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/replica:0/task:%d" % rank,
        cluster=cluster)):

        global_step = tf.Variable(0, trainable = False)
        is_chief = rank == 1


        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])

        mnist = input_data.read_data_sets(data_dir, one_hot=True)
      
        #
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.nn.softmax(tf.matmul(x, W) + b)
      
        #
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32]
      
        #
        x_image = tf.reshape(x, [-1, 28, 28, 1])
      
        #
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
      
        #
        W_conv2 = weight_variable([5, 5, 32, 64])     
        b_conv2 = bias_variable([64])    
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        #
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        #
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        #
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        #
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy, global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #
        #saver allows for saving/restoring variables to/from checkpoints during training
        saver = tf.train.Saver()
        #summary_op tracks all summaries of the graph
        summary_op = tf.summary.merge_all()
        #init_op defines the operation to initialize all tf.Variable()s
        init_op = tf.global_variables_initializer()

        sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/replica:0/task:%d" % rank])

        if is_chief:
            print("Worker %d: Initializing session..." % rank)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                rank)

        #
        sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir="train_logs",
                             summary_op=summary_op,
                             init_op=init_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)  
        server_grpc_url = "grpc://" + worker_hosts[rank]

        #
        with sv.prepare_or_wait_for_session(server_grpc_url,
                                          config=sess_config) as sess:
            step = 0
            #If anything goes wrong, sv.should_stop() will halt execution on a worker
            while (not sv.should_stop()) and (step < 5000):
                # Run a training step asynchronously.
                batch = mnist.train.next_batch(batch_size)
                if step % 10 == 0:
                train_accuracy = accuracy.eval(session=sess,feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (step, train_accuracy))
                _, step = sess.run([train_step, global_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            print('test accuracy %g' % accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        
        print("ALL FINISHED")
        sv.stop()