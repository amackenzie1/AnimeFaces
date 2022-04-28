from Distributed.coordinator import start_coordinator
import tensorflow as tf
from anime_dataset_gen import generator
from fancy_anime_model import get_model

global_batch_size = 100 

def dataset_fn(input_context):
    dataset = tf.data.Dataset.from_generator(generator, output_types=({"input" : tf.float32, "time": tf.float32}, tf.float32))
    dataset = dataset.batch(global_batch_size)
    dataset = dataset.shard(input_context.num_input_pipelines, 
                            input_context.input_pipeline_id)
    return dataset
   
start_coordinator(get_model, dataset_fn, global_batch_size)
