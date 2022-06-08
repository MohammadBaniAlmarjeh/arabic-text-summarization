import logging
import os
import re
from functools import lru_cache
from urllib.parse import unquote

import streamlit as st
from codetiming import Timer
from transformers import pipeline
from preprocess import ArabertPreprocessor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import GPT2TokenizerFast, BertTokenizer

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger.info("Loading models...")
reader_time = Timer("loading", text="Time: {:.2f}", logger=logging.info)
reader_time.start()
#####

#tokenizer = AutoTokenizer.from_pretrained("malmarjeh/bert2bert")
#model = AutoModelForSeq2SeqLM.from_pretrained("malmarjeh/bert2bert")
#generation_pipeline = pipeline("text2text-generation",model=model,tokenizer=tokenizer)
#logger.info("BERT2BERT is loaded")

#tokenizer_mbert = BertTokenizer.from_pretrained("malmarjeh/mbert2mbert-arabic-text-summarization")
#model_mbert = AutoModelForSeq2SeqLM.from_pretrained("malmarjeh/mbert2mbert-arabic-text-summarization")
#generation_pipeline_mbert = pipeline("text2text-generation",model=model_mbert,tokenizer=tokenizer_mbert)
#logger.info("mBERT2mBERT is loaded")

tokenizer_t5 = AutoTokenizer.from_pretrained("malmarjeh/t5-arabic-text-summarization")
model_t5 = AutoModelForSeq2SeqLM.from_pretrained("malmarjeh/t5-arabic-text-summarization")
generation_pipeline_t5 = pipeline("text2text-generation",model=model_t5,tokenizer=tokenizer_t5)
logger.info("T5 is loaded")

#tokenizer_transformer = AutoTokenizer.from_pretrained("malmarjeh/transformer")
#model_transformer = AutoModelForSeq2SeqLM.from_pretrained("malmarjeh/transformer")
#generation_pipeline_transformer = pipeline("text2text-generation",model=model_transformer,tokenizer=tokenizer_transformer)
#logger.info("Transformer is loaded")

#tokenizer_gpt2 = GPT2TokenizerFast.from_pretrained("aubmindlab/aragpt2-base")
#model_gpt2 = AutoModelForCausalLM.from_pretrained("malmarjeh/gpt2")
#logger.info("GPT-2 is loaded")

reader_time.stop()

preprocessor = ArabertPreprocessor(model_name="")

logger.info("Finished loading the models...")
logger.info(f"Time spent loading: {reader_time.last}")

@lru_cache(maxsize=200)
def get_results(text, model_selected, num_beams, length_penalty):
    logger.info("\n=================================================================")
    logger.info(f"Text: {text}")
    logger.info(f"model_selected: {model_selected}")
    logger.info(f"length_penalty: {length_penalty}")
    reader_time = Timer("summarize", text="Time: {:.2f}", logger=logging.info)
    reader_time.start()
    logger.info(len(preprocessor.preprocess(text).split()))
    text = ' '.join(preprocessor.preprocess(text).split()[:150])
    
    if model_selected == 'Transformer':
#        result = generation_pipeline_transformer(text,
#            pad_token_id=tokenizer_transformer.eos_token_id,
#            num_beams=num_beams,
#            repetition_penalty=3.0,
#            max_length=200,
#            length_penalty=length_penalty,
#            no_repeat_ngram_size = 3)[0]['generated_text']
        logger.info('Transformer')
    #elif model_selected == 'Seq2Seq_LSTM':
        #result = generate_seq2seq(text, num_beams)
    #    result = "عذراً، هذا النموذج غير متاح حالياً"
    #    logger.info('Seq2Seq_LSTM')
    elif model_selected == 'GPT-2':
#        text_processed = '\n النص: ' + text + ' \n الملخص: \n '
#        tokenizer_gpt2.add_special_tokens({'pad_token': '<pad>'})
#        text_tokens = tokenizer_gpt2.batch_encode_plus([text_processed], return_tensors='pt', padding='max_length', max_length=150)
#        output_ = model_gpt2.generate(input_ids=text_tokens['input_ids'],repetition_penalty=3.0,length_penalty=length_penalty, num_beams=num_beams, max_length=240, pad_token_id=2, eos_token_id=0, bos_token_id=10611)
#        result = tokenizer_gpt2.decode(output_[0][150:], skip_special_tokens=True).strip()
        logger.info('GPT-2')
    elif model_selected == 'mBERT2mBERT':
#        result = generation_pipeline_mbert(text,
#            pad_token_id=tokenizer_mbert.eos_token_id,
#            num_beams=num_beams,
#            repetition_penalty=3.0,
#            max_length=200,
#            length_penalty=length_penalty,
#            no_repeat_ngram_size = 3)[0]['generated_text']
        logger.info('mBERT')
    elif model_selected == 'T5':
        result = generation_pipeline_t5(text,
            pad_token_id=tokenizer_t5.eos_token_id,
            num_beams=num_beams,
            repetition_penalty=3.0,
            max_length=200,
            length_penalty=length_penalty,
            no_repeat_ngram_size = 3)[0]['generated_text']
        logger.info('t5')
    elif model_selected == 'BERT2BERT':
#        result = generation_pipeline(text,
#            pad_token_id=tokenizer.eos_token_id,
#            num_beams=num_beams,
#            repetition_penalty=3.0,
#            max_length=200,
#            length_penalty=length_penalty,
#            no_repeat_ngram_size = 3)[0]['generated_text']
        logger.info('bert2bert')
    else:
        result = "الرجاء اختيار نموذج"
    #repetition_penalty = 3.0,

    reader_time.stop()
    logger.info(f"Time spent: {reader_time.last}")

    return result


if __name__ == "__main__":
    results_dict = "" #get_results("")
