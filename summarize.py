import logging
import os
import re
from functools import lru_cache
from urllib.parse import unquote
import tensorflow as tf
import streamlit as st
from codetiming import Timer
from transformers import AutoTokenizer, pipeline

from preprocess import ArabertPreprocessor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("malmarjeh/bert2bert")

model = AutoModelForSeq2SeqLM.from_pretrained("malmarjeh/bert2bert")

generation_pipeline = pipeline("text2text-generation",model=model,tokenizer=tokenizer)


logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

preprocessor = ArabertPreprocessor(model_name="")
logger.info("Loading Pipeline...")

logger.info("Finished loading Pipeline...")


@lru_cache(maxsize=100)
def get_results(text, model=2):
    logger.info("\n=================================================================")
    logger.info(f"Question: {text}")

    reader_time = Timer("summarize", text="Time: {:.2f}", logger=logging.info)
    reader_time.start()

    text = preprocessor.preprocess(text)

    if model == 2:
        result = generate_seq2seq(text)
    else:

        result = generation_pipeline(text,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=3,
            max_length=200,
            top_p=0.9,
            repetition_penalty = 3.0,
            no_repeat_ngram_size = 3)[0]['generated_text']

    reader_time.stop()
    logger.info(f"Time spent: {reader_time.last}")
    return result

def generate_seq2seq(text):

    model_name = 'new_bert_tokenizer_20k_add'
    tokenizers = tf.saved_model.load(model_name)

    vocab_inp_size = tokenizers.text.get_vocab_size().numpy()
    vocab_tar_size = tokenizers.title.get_vocab_size().numpy()

    embedding_dim = 256
    units = 512
    BATCH_SIZE = 64

    ##### 

    class Encoder(tf.keras.Model):
      def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        ##-------- LSTM layer in Encoder ------- ##
        self.lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.enc_units,
                                                           return_sequences=True,
                                                           return_state=True,
                                                           recurrent_initializer='glorot_uniform'), merge_mode='ave')
        
      def call(self, x, hidden=None):
        x = self.embedding(x)
        output, forward_h, forward_c, backward_h, backward_c = self.lstm_layer(x, initial_state=hidden)
        h = tf.reduce_mean([forward_h, backward_h], axis=0)
        c = tf.reduce_mean([forward_c, backward_c], axis=0)
        return output, h, c

      def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))] 

    ## Test Encoder Stack

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

    # sample input
    #sample_output, sample_h, sample_c = encoder(example_input_batch)
    #print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    #print ('Encoder h vecotr shape: (batch size, units) {}'.format(sample_h.shape))
    #print ('Encoder c vector shape: (batch size, units) {}'.format(sample_c.shape))

    class Decoder(tf.keras.Model):
      def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention_type='luong'):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        #Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)
       


        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(self.dec_units, 
                                                                  None, self.batch_sz*[max_length_input], self.attention_type)
        
        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(self.batch_sz)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

        
      def build_rnn_cell(self, batch_sz):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, 
                                      self.attention_mechanism, attention_layer_size=self.dec_units)
        return rnn_cell

      def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
        # ------------- #
        # typ: Which sort of attention (Bahdanau, Luong)
        # dec_units: final dimension of attention outputs 
        # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
        # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

        if(attention_type=='bahdanau'):
          return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
        else:
          return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

      def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state


      def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[max_length_output-1])
        return outputs


    # Test decoder stack
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, 'luong')
    #sample_x = tf.random.uniform((BATCH_SIZE, sample_max_output))
    #decoder.attention_mechanism.setup_memory(sample_output)
    #initial_state = decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c], tf.float32)


    #sample_decoder_outputs = decoder(sample_x, initial_state)

    #print("Decoder Outputs Shape: ", sample_decoder_outputs.rnn_output.shape)

    optimizer = tf.keras.optimizers.Adam()

    #checkpoint_dir = './drive/MyDrive/subword_training_checkpoints' MOD1
    checkpoint_dir = '.'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    def beam_evaluate_sentence(inputs, beam_width=3):
      #sentence = dataset_creator.preprocess_sentence(sentence)

      #inputs = inp_lang.texts_to_sequences(sentence)
      #inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
      #                                                        maxlen=max_length_input,
      #                                                        padding='post',
      #                                                       truncating='post')

      #inputs = tf.convert_to_tensor(inputs)
      inference_batch_size = inputs.shape[0]
      result = ''
      
      #enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size,units))]
      enc_out, enc_h, enc_c = encoder(inputs)

      dec_h = enc_h
      dec_c = enc_c

      START_T = 2
      END_T = 3
      start_tokens = tf.fill([inference_batch_size], START_T)
      end_token = END_T

      # From official documentation
      # NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:
      # The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
      # The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
      # The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.
      enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
      decoder.attention_mechanism.setup_memory(enc_out)
      #print("beam_with * [batch_size, max_length_input, rnn_units] :  3 * [1, 16, 1024]] :", enc_out.shape)

      # set decoder_inital_state which is an AttentionWrapperState considering beam_width
      hidden_state = tfa.seq2seq.tile_batch([enc_h, enc_c], multiplier=beam_width)
      decoder_initial_state = decoder.rnn_cell.get_initial_state(batch_size=beam_width*inference_batch_size, dtype=tf.float32)

      decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)

      # Instantiate BeamSearchDecoder
      decoder_instance = tfa.seq2seq.BeamSearchDecoder(decoder.rnn_cell,beam_width=beam_width, output_layer=decoder.fc)

      decoder_embedding_matrix = decoder.embedding.variables[0]

      # The BeamSearchDecoder object's call() function takes care of everything.
      outputs, final_state, sequence_lengths = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)

      # outputs is tfa.seq2seq.FinalBeamSearchDecoderOutput object. 
      # The final beam predictions are stored in outputs.predicted_id
      # outputs.beam_search_decoder_output is a tfa.seq2seq.BeamSearchDecoderOutput object which keep tracks of beam_scores and parent_ids while performing a beam decoding step
      # final_state = tfa.seq2seq.BeamSearchDecoderState object.
      # Sequence Length = [inference_batch_size, beam_width] details the maximum length of the beams that are generated

      
      # outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width)
      # outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width)
      # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
      #print(outputs.predicted_ids.shape)
      #print(outputs.beam_search_decoder_output.scores.shape)
      
      final_outputs = tf.transpose(outputs.predicted_ids, perm=(0,2,1))
      #beam_scores = tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0,2,1))

      #return final_outputs.numpy(), beam_scores.numpy()
      return tf.squeeze(final_outputs.numpy()[:,0,:])

    def beam_translate_sent(sentence):
      inputs = tokenizers.text.tokenize(sentence)
      inputs = inputs.to_tensor()
      inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                              maxlen=max_length_input,
                                                              padding='post',
                                                             truncating='post')
      
      result = beam_evaluate_sentence(inputs)

      output = [t.decode('utf-8') for t in tokenizers.title.detokenize(result.numpy().tolist()).numpy()]
      for i in range(len(output)):
        print('Input: %s' % (sentence[i]))
        print('Predicted translation: {}'.format(output[i]))
      return result[0]

    res = checkpoint.restore('ckpt-7')
    #res = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    res.assert_consumed()

    beam_translate_sent([text])


if __name__ == "__main__":
    results_dict = get_results("ما هو نظام لبنان؟")
