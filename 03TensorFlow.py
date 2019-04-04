import tensorflow as tf
import json

""" 
Ler os jobs do json criado pelo script anterior 
RETORNA: um dicionario com uma chave "nodes" e
uma lista de strings com {hostname}:{porta} para
cada job calculado no script anterior.
"""
f = open("saida.out", "r")
jobs = json.loads(f.read())
f.close()


# Criar o cluster tensorflow
cluster = tf.train.ClusterSpec(jobs)

# Criar uma constante tensorflow
x = tf.constant(2)

# Na tarefa 1, executar as instrucoes
with tf.device("/job:nodes/task:1"):
    y2 = x - 66


# Na tarefa 0, executar as instrucoes
with tf.device("/job:nodes/task:0"):
    y1 = x + 300
    y = y1 + y2

# Iniciar uma sessao e retornar o valor de y
# a sessao liga em grpc://{nome do no}:{porta}
first_job_host_port = (jobs["nodes"][0])
first_job_url = "grpc://%s" % first_job_host_port

with tf.Session(first_job_url) as sess:
    result = sess.run(y)
    print(result)