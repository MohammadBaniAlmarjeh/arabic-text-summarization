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
import tokenizers

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger.info("Loading models...")
reader_time = Timer("loading", text="Time: {:.2f}", logger=logging.info)
reader_time.start()
#####

@st.cache(ttl=24*3600, hash_funcs={AutoModelForSeq2SeqLM: lambda _: None})
def load_seq2seqLM_model(model_path):
    logger.info("AutoModelForSeq2SeqLM is loading")
    return AutoModelForSeq2SeqLM.from_pretrained(model_path)
@st.cache(ttl=24*3600, hash_funcs={AutoModelForCausalLM: lambda _: None})
def load_casualLM_model(model_path):
    logger.info("AutoModelForCausalLM is loading")
    return AutoModelForCausalLM.from_pretrained(model_path)

@st.cache(ttl=24*3600, hash_funcs={tokenizers.Tokenizer: lambda _: None})
def load_autotokenizer_model(tokenizer_path):
    logger.info("AutoTokenizer is loading")
    return AutoTokenizer.from_pretrained(tokenizer_path)
@st.cache(ttl=24*3600, hash_funcs={BertTokenizer: lambda _: None})
def load_berttokenizer_model(tokenizer_path):
    logger.info("BertTokenizer is loading")
    return BertTokenizer.from_pretrained(tokenizer_path)
@st.cache(ttl=24*3600, hash_funcs={GPT2TokenizerFast: lambda _: None})
def load_gpt2tokenizer_model(tokenizer_path):
    logger.info("GPT2TokenizerFast is loading")
    return GPT2TokenizerFast.from_pretrained(tokenizer_path)

@st.cache(ttl=24*3600, allow_output_mutation=True, hash_funcs={pipeline: lambda _: None, tokenizers.Tokenizer: lambda _: None})
def load_generation_pipeline(model_path):
    logger.info("Pipeline is loading")
    if model_path == "malmarjeh/mbert2mbert-arabic-text-summarization":
        tokenizer = load_berttokenizer_model(model_path)
    else:
        tokenizer = load_autotokenizer_model(model_path)
    #model = load_seq2seqLM_model(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return pipeline("text2text-generation",model=model,tokenizer=tokenizer)

@st.cache(ttl=24*3600, hash_funcs={ArabertPreprocessor: lambda _: None})
def load_preprocessor():
    return ArabertPreprocessor(model_name="")

tokenizer = load_autotokenizer_model("malmarjeh/bert2bert")
generation_pipeline = load_generation_pipeline("malmarjeh/bert2bert")
logger.info("BERT2BERT is loaded")

tokenizer_mbert = load_berttokenizer_model("malmarjeh/mbert2mbert-arabic-text-summarization")
generation_pipeline_mbert = load_generation_pipeline("malmarjeh/mbert2mbert-arabic-text-summarization")
logger.info("mBERT2mBERT is loaded")

tokenizer_t5 = load_autotokenizer_model("malmarjeh/t5-arabic-text-summarization")
generation_pipeline_t5 = load_generation_pipeline("malmarjeh/t5-arabic-text-summarization")
logger.info("T5 is loaded")

tokenizer_transformer = load_autotokenizer_model("malmarjeh/transformer")
generation_pipeline_transformer = load_generation_pipeline("malmarjeh/transformer")
logger.info("Transformer is loaded")

tokenizer_gpt2 = load_gpt2tokenizer_model("aubmindlab/aragpt2-base")
model_gpt2 = load_casualLM_model("malmarjeh/gpt2")
logger.info("GPT-2 is loaded")

reader_time.stop()

preprocessor = load_preprocessor()

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
    if model_selected == 'GPT-2':
        number_of_tokens_limit = 80
    else:
        number_of_tokens_limit = 150
    text = preprocessor.preprocess(text)
    logger.info(f"input length: {len(text.split())}")
    text = ' '.join(text.split()[:number_of_tokens_limit])
    
    if model_selected == 'Transformer':
        result = generation_pipeline_transformer(text,
            pad_token_id=tokenizer_transformer.eos_token_id,
            num_beams=num_beams,
            repetition_penalty=3.0,
            max_length=200,
            length_penalty=length_penalty,
            no_repeat_ngram_size = 3)[0]['generated_text']
        logger.info('Transformer')
    elif model_selected == 'GPT-2':
        text_processed = '\n النص: ' + text + ' \n الملخص: \n '
        tokenizer_gpt2.add_special_tokens({'pad_token': '<pad>'})
        text_tokens = tokenizer_gpt2.batch_encode_plus([text_processed], return_tensors='pt', padding='max_length', max_length=100)
        output_ = model_gpt2.generate(input_ids=text_tokens['input_ids'],repetition_penalty=3.0, num_beams=num_beams, max_length=140, pad_token_id=2, eos_token_id=0, bos_token_id=10611)
        result = tokenizer_gpt2.decode(output_[0][100:], skip_special_tokens=True).strip()
        logger.info('GPT-2')
    elif model_selected == 'mBERT2mBERT':
        result = generation_pipeline_mbert(text,
            pad_token_id=tokenizer_mbert.eos_token_id,
            num_beams=num_beams,
            repetition_penalty=3.0,
            max_length=200,
            length_penalty=length_penalty,
            no_repeat_ngram_size = 3)[0]['generated_text']
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
        result = generation_pipeline(text,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=num_beams,
            repetition_penalty=3.0,
            max_length=200,
            length_penalty=length_penalty,
            no_repeat_ngram_size = 3)[0]['generated_text']
        logger.info('bert2bert')
    else:
        result = "الرجاء اختيار نموذج"

    reader_time.stop()
    logger.info(f"Time spent summarizing: {reader_time.last}")

    return result


if __name__ == "__main__":
    results_dict = ""
