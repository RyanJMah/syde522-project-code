import os
import soundfile as sf
import tensorflow.compat.v1 as tf   # type: ignore
import numpy as np
from typing import Optional

import paths
import sys
sys.path.append(paths.VGGISH_DIR)

import vggish_input
import vggish_slim
import vggish_postprocess
import vggish_params

import logging

def set_log_level(level):
    if level == "WARNING":
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.get_logger().setLevel(logging.WARNING)

    elif level == "ERROR":
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel(logging.ERROR)
    
    else:
        ValueError("Invalid log level")

set_log_level("ERROR")

def sequence_to_numpy(sequence):
    embeddings = []

    for seq in sequence:
        # Retrieve the embeddings from the sequence example.
        feature_list = seq.feature_lists.feature_list[vggish_params.AUDIO_EMBEDDING_FEATURE_NAME]

        for feature in feature_list.feature:
            # Each feature is a byte string of 128-dim embeddings
            embedding = np.frombuffer(feature.bytes_list.value[0], dtype=np.uint8)
            embeddings.append(embedding)
        
    return np.array(embeddings)


def extract_features_batch(ogg_files: list, out_files: Optional[list] = None) -> list:
    batches_data = []

    for ogg_file in ogg_files:
        data, sample_rate = sf.read(ogg_file)
        batch = vggish_input.waveform_to_examples(data, sample_rate)
        batches_data.append(batch)

    pproc = vggish_postprocess.Postprocessor(paths.VGGISH_PCA_PARAMS)

    features = []

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.

        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, paths.VGGISH_MODEL)

        features_tensor  = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        for batch in batches_data:
            # Run inference and postprocessing.
            [embedding_batch] = sess.run( [embedding_tensor],
                                        feed_dict={features_tensor: batch} )
            # print(embedding_batch)

            postprocessed_batch = pproc.postprocess(embedding_batch)
            # print(postprocessed_batch)

            # Write the postprocessed embeddings as a SequenceExample, in a similar
            # format as the features released in AudioSet. Each row of the batch of
            # embeddings corresponds to roughly a second of audio (96 10ms frames), and
            # the rows are written as a sequence of bytes-valued features, where each
            # feature value contains the 128 bytes of the whitened quantized embedding.
            seq_example = tf.train.SequenceExample(
                feature_lists=tf.train.FeatureLists(
                    feature_list={
                        vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
                            tf.train.FeatureList(
                                feature=[
                                    tf.train.Feature(
                                        bytes_list=tf.train.BytesList(
                                            value=[embedding.tobytes()]))
                                    for embedding in postprocessed_batch
                                ]
                            )
                    }
                )
            )
            # print(seq_example)

            features.append(seq_example)


    if out_files is not None:
        assert len(out_files) == len(features)

        for out_file, feature in zip(out_files, features):
            writer = tf.python_io.TFRecordWriter(out_file)
            writer.write(feature.SerializeToString())

            if writer:
                writer.close()

    return sequence_to_numpy(features)


def read_feature_file(tfrecord_file: str):
    # return sequence_to_numpy( tf.data.TFRecordDataset([tfrecord_file]) )

    embeddings = []

    for example in tf.data.TFRecordDataset([tfrecord_file]):
        # Parse the input `tf.Example` proto using the dictionary above.
        sequence_example = tf.train.SequenceExample.FromString(example.numpy())

        # Retrieve the embeddings from the sequence example.
        feature_list = sequence_example.feature_lists.feature_list[vggish_params.AUDIO_EMBEDDING_FEATURE_NAME]

        for feature in feature_list.feature:
            # Each feature is a byte string of 128-dim embeddings
            embedding = np.frombuffer(feature.bytes_list.value[0], dtype=np.uint8)
            embeddings.append(embedding)
    
    return np.array(embeddings)
