import logging
import os
import re
from functools import lru_cache
from urllib.parse import unquote

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
def get_results(text):
    logger.info("\n=================================================================")
    logger.info(f"Question: {text}")

    reader_time = Timer("summarize", text="Time: {:.2f}", logger=logging.info)
    reader_time.start()

    text = preprocessor.preprocess(text)

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

if __name__ == "__main__":
    results_dict = get_results("ما هو نظام لبنان؟")
