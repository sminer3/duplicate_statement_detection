import tensorflow as tf 
import tensorflow_hub as hub

def get_use_embeddings(statements):
	#Input a list of sentences and get a np array of sentence embeddings
	embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
	with tf.Session() as session:
		session.run([tf.global_variables_initializer(), tf.tables_initializer()])
		embeddings = session.run(embed(statements))
	return embeddings
