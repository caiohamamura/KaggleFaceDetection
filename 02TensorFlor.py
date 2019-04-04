from mpi4py import MPI
import numpy as np
import socket
import tensorflow as tf


def gather_jobs_with_mpi():
    # hostname em formato ubyte de cada node
    hostname = list(socket.gethostname().encode('utf8'))

    # Comunicador MPI 
    comm = MPI.COMM_WORLD
    size = comm.Get_size() # qde de tarefas MPI
    rank = comm.Get_rank() # identificador da tarefa (rank)
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
    jobs = {'nodes': ['%s:222%s' % (i, j) for (i,j) in zip(hosts, ranks)]}

    return jobs

jobs = gather_jobs_with_mpi()

# Se for a tarefa com id 0
if rank == 0:
    import json
    f = open("saida.out", "w+")
    f.write(json.dumps(jobs)) # Salvar os dados das tarefas em json
    f.close()
    
    
# Criar o cluster de jobs
cluster = tf.train.ClusterSpec(jobs)

# Iniciar um servidor do tensorflow para cada tarefa
server = tf.train.Server(cluster, job_name="nodes", task_index=rank)
server.start()
server.join() # Espera por instrucoes