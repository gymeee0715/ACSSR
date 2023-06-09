import os, sys
import logging
import numpy as np
import pandas as pd
import argparse

import torchaudio
import torch
import re
import json 
import librosa
from datasets import DatasetDict, load_metric

from transformers import (
    set_seed,
    Wav2Vec2Processor, 
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
    Trainer,
    HfArgumentParser,
    EarlyStoppingCallback
)

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import datasets
import pickle

import editdistance
import jieba
from itertools import chain

import transformers
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from args_helper import ModelArguments, DataArguments, TrainingArguments
from utils import CHARS_TO_IGNORE, remove_special_characters, tokenize_for_mer, tokenize_for_cer,plt_confusion_matrix
from data_utils import speech_file_to_array_fn, load_dataset, DataCollatorCTCWithPadding

import datasets
from datasets import load_from_disk, set_caching_enabled
 
from accelerate import find_executable_batch_size


# set_caching_enabled(True)
logger = logging.getLogger(__name__)    

logging.disable(logging.INFO)

def load_processor(model_args, training_args):
    # Load processor
    print('Load Wav2Vec2 processor...')

    try:
        pretrained_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_args.model_name_or_path)
        # pretrained_tokenizer = SpeechEncoderDecoderModel.from_pretrained(model_args.model_name_or_path)
       
        pretrained_vocab = list(map(lambda x: x[0], sorted(pretrained_tokenizer.get_vocab().items(), key=lambda x: x[1])))
    except:
        pretrained_vocab = []

    logger.info("Vocab length (initial): {}".format(len(pretrained_vocab)))
    print("Vocab length (initial):", len(pretrained_vocab))

    with open("{}/new_vocab.json".format(training_args.output_dir), "r") as new_vocab_file:
        new_vocab_list = json.load(new_vocab_file)
        logger.info("New vocabulary length: {}".format(len(new_vocab_list)))

    all_vocab = list(dict.fromkeys(pretrained_vocab + new_vocab_list))

    vocab_dict = {v: k for k, v in enumerate(all_vocab)}

    def _assign_id_to_special_tokens(vocab_dict):
        bos_token = "<s>"
        eos_token = "</s>"
        unk_token = "[UNK]"
        pad_token = "<pad>"
        word_delimiter_token = "|"

        if bos_token not in vocab_dict:
            vocab_dict[bos_token] = len(vocab_dict)

        if eos_token not in vocab_dict:
            vocab_dict[eos_token] = len(vocab_dict)

        if unk_token not in vocab_dict:
            if "<unk>" in vocab_dict:
                vocab_dict[unk_token] = vocab_dict.pop("<unk>")
            else:
                vocab_dict[unk_token] = len(vocab_dict)

        if pad_token not in vocab_dict:
            vocab_dict[pad_token] = len(vocab_dict)

        if word_delimiter_token not in vocab_dict:
            vocab_dict[word_delimiter_token] = len(vocab_dict)

        return vocab_dict

    vocab_dict = _assign_id_to_special_tokens(vocab_dict)
    print("len vocab dict", len(vocab_dict))

    with open("{}/all_vocab.json".format(training_args.output_dir), "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)
        
    tokenizer = Wav2Vec2CTCTokenizer("{}/all_vocab.json".format(training_args.output_dir), unk_token="[UNK]")

    logger.info("Vocab size (final): {}".format(tokenizer.vocab_size))
    print("Vocab size (final):", tokenizer.vocab_size)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    return processor
    
#####
# Main Functions
#####
def run(model_args, data_args, training_args):
    ###
    # Prepare Processor & Model    
    ###
    training_args.output_dir="{}/{}".format(training_args.output_dir, model_args.model_name_or_path)

    os.makedirs(training_args.output_dir, exist_ok=True)

    cache_dir_path = "./{}/{}".format(data_args.cache_dir_name, model_args.model_name_or_path)
    os.makedirs(cache_dir_path, exist_ok=True)

    print('cache_dir_path', cache_dir_path)
    if not os.path.exists("{}/preprocess_data.arrow".format(cache_dir_path)):
        ###
        # Prepare Dataset
        ###
        raw_datasets = DatasetDict()
        print('Loading train dataset...')
        raw_datasets["train"] = load_dataset(data_args.train_manifest_path, data_args.preprocessing_num_workers, 
                                        data_args.audio_column_name, data_args.text_column_name)
        print('Loading validation dataset...')
        raw_datasets["valid"] = load_dataset(data_args.valid_manifest_path, data_args.preprocessing_num_workers, 
                                        data_args.audio_column_name, data_args.text_column_name)
        print('Loading test dataset...')
        raw_datasets["test"] = load_dataset(data_args.test_manifest_path, data_args.preprocessing_num_workers, 
                                        data_args.audio_column_name, data_args.text_column_name)

        print('Preprocess dataset...')

        # Remove ignorable characters
        print('Removing ignorable characters')
        chars_to_ignore_re = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
        def remove_special_characters(batch):
            if chars_to_ignore_re is not None:
                batch[data_args.text_column_name] = re.sub(chars_to_ignore_re, "", batch[data_args.text_column_name]).upper() + " "
            else:
                batch[data_args.text_column_name] = batch[data_args.text_column_name].upper() + " "
            return batch

        with training_args.main_process_first(desc="dataset map special characters removal"):
            raw_datasets = raw_datasets.map(
                remove_special_characters,
                num_proc=data_args.preprocessing_num_workers,
                desc="remove special characters from datasets",
                load_from_cache_file=True,
                cache_file_names={
                    "train": "{}/train_clean.arrow".format(cache_dir_path),
                    "valid": "{}/valid_clean.arrow".format(cache_dir_path),
                    "test": "{}/test_clean.arrow".format(cache_dir_path),
                }
            )

        # Build vocabulary
        print('Build vocabulary...')
        def extract_all_chars(batch):
            all_text = " ".join(batch[data_args.text_column_name])
            vocab = list(set(all_text))
            return {"vocab": [vocab], "all_text": [all_text]}

        with training_args.main_process_first(desc="vocab building"):
            _vocab = raw_datasets.map(
                extract_all_chars,
                num_proc=data_args.preprocessing_num_workers,
                desc="build vocabulary",
                load_from_cache_file=True,
                cache_file_names={
                    "train": "{}/train_vocab.arrow".format(cache_dir_path),
                    "valid": "{}/valid_vocab.arrow".format(cache_dir_path),
                    "test": "{}/test_vocab.arrow".format(cache_dir_path),
                }
            )

            def flatten(vocab_split):
                return list(chain.from_iterable(list(chain.from_iterable(vocab_split))))

            vocab_list = list(set(flatten(_vocab["train"]["vocab"]) + flatten(_vocab["valid"]["vocab"]) + flatten(_vocab["test"]["vocab"])))
            # vocab_dict = {v: k for k, v in enumerate(vocab_list)}
            # vocab_dict["|"] = vocab_dict[" "]
            # vocab_dict["[UNK]"] = len(vocab_dict)
            # vocab_dict["[PAD]"] = len(vocab_dict)

            # Dump vocabulary
            with open("{}/new_vocab.json".format(training_args.output_dir), "w") as vocab_file:
                json.dump(vocab_list, vocab_file)

        # Load processor
        processor = load_processor(model_args, training_args)

        # Preprocess audio sample and label text
        print('Vectorize dataset...')

        def prepare_dataset(batch):
            # Preprocess audio
            batch["input_values"] = processor(batch["speech_sample"], sampling_rate=16000).input_values[0]

            # Preprocess text
            with processor.as_target_processor():
                batch["labels"] = processor(batch[data_args.text_column_name]).input_ids
                batch["target_text_lid"] = batch["input_length"].split(',')
                batch["target_text_lid"] = [int(i) for i in batch["target_text_lid"]]
                # batch["LID_labels"] = batch["target_text"]
                # batch["labels"] =  batch["target_text"] 
            return batch

        with training_args.main_process_first(desc="dataset map preprocessing"):
            vectorized_datasets = raw_datasets.map(
                prepare_dataset,
                remove_columns=raw_datasets["train"].column_names,
                num_proc=data_args.preprocessing_num_workers,
                desc="preprocess datasets",
                load_from_cache_file=True,
                cache_file_names={
                    "train": "{}/train_vec.arrow".format(cache_dir_path),
                    "valid": "{}/valid_vec.arrow".format(cache_dir_path),
                    "test": "{}/test_vec.arrow".format(cache_dir_path),
                }
            )
        
        vectorized_datasets.save_to_disk("{}/preprocess_data.arrow".format(cache_dir_path))
    else:
        print('Loading cached dataset...')
        vectorized_datasets = datasets.load_from_disk('{}/preprocess_data.arrow'.format(cache_dir_path))

        # Load processor
        processor = load_processor(model_args, training_args)

    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return
    
    ###
    # Prepare Data Collator and Trainer
    ###
    print('Preparing Trainer...')

    print('Load Wav2Vec2 model...')
    print('Model ID', model_args.model_name_or_path)
    config = Wav2Vec2Config.from_pretrained(model_args.model_name_or_path)
    config.update({
        "mask_time_prob": model_args.mask_time_prob,
        "mask_time_length": model_args.mask_time_length,
        "mask_feature_prob": model_args.mask_feature_prob,
        "mask_feature_length": model_args.mask_feature_length,
        "gradient_checkpointing": training_args.gradient_checkpointing,
    })
    # config.update({
    #     "vocab_size":processor.tokenizer.vocab_size
    # })
    model = Wav2Vec2ForCTC.from_pretrained(model_args.model_name_or_path, config=config,ignore_mismatched_sizes=True)
    # # print(model),ignore_mismatched_sizes=True
    # model = Wav2Vec2ForAudioFrameClassification.from_pretrained(model_args.model_name_or_path, config=config)
    model.cuda()
    
    # model.load_state_dict(torch.load("/root/new_ev/transformers/examples/pytorch/speech-recognition/ASCEND/save/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn/checkpoint-8624/pytorch_model.bin"),strict=False)
    # model.eval()

    model.freeze_feature_extractor()
    model.freeze_feature_encoder()
    def _resize_token_embeddings(model, new_num_tokens):
        old_lm_head = model.lm_head
        new_lm_head = model._get_resized_lm_head(old_lm_head, new_num_tokens)
        model.lm_head = new_lm_head
        model.config.update({"vocab_size": new_num_tokens})
        return model

    model = _resize_token_embeddings(model, processor.tokenizer.vocab_size)


    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Define compute metric function
    def compute_metrics(pred):
        logger.info("*** Compute metrics ***")
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_strs = processor.batch_decode(pred_ids)

        # we do not want to group tokens when computing the metrics
        label_strs = processor.batch_decode(pred.label_ids, group_tokens=False)
        mixed_distance, mixed_tokens = 0, 0
        char_distance, char_tokens = 0, 0
        for i, (pred_str, label_str) in enumerate(zip(pred_strs, label_strs)):
            # Calculate 
            m_pred = tokenize_for_mer(pred_str)
            m_ref = tokenize_for_mer(label_str)
            mixed_distance += editdistance.distance(m_pred, m_ref)
            mixed_tokens += len(m_ref)

            c_pred = tokenize_for_cer(pred_str)
            c_ref = tokenize_for_cer(label_str)
            char_distance += editdistance.distance(c_pred, c_ref)
            char_tokens += len(c_ref)
        mer = mixed_distance / mixed_tokens
        cer = char_distance / char_tokens

        f = open(f'{training_args.output_dir}/test.results', 'w')
        f.writelines([item+'\n' for item in pred_strs])
        f.close()
        f = open(f'{training_args.output_dir}/test.label', 'w')
        f.writelines([item+'\n' for item in label_strs])
        f.close()
        logger.info("mer: {} --- cer: {}".format(mer, cer))

        return {"mer": mer, "cer": cer} 

        # metric = load_metric("accuracy")
        
        # logits=pred.predictions
        # # predictions=logits
        # labels = pred.label_ids
        # # print(labels.shape)
        # # print(logits.shape)
        # lnew_lab=labels[:,0:len(logits[0])].flatten().tolist()
        # predictions = np.argmax(logits, axis=-1).flatten ().tolist()
        # np_lnew_lab = np.array(lnew_lab)
        # eq_letter = np.argwhere(np_lnew_lab == -100)
        
        # for i in range(len(eq_letter)):
        #     if -100 in lnew_lab:
        #         p=lnew_lab.index(-100)
        #         lnew_lab.pop(p)
        #         predictions.pop(p)  
       
        # acc = metric.compute(predictions=predictions, references=lnew_lab)
        # plt_confusion_matrix(predictions,lnew_lab,acc)
        # return acc
        
    # Initialize Trainer
    trainer = Trainer(
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["valid"],
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    ###
    # Training Phase
    ###
    print('*** Training Phase ***')
    
    # use last checkpoint if exist
    if os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    # train_result = trainer.train()
    trainer.save_model()

    # Save the feature_extractor and the tokenizer
    if is_main_process(training_args.local_rank):
        processor.save_pretrained(training_args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(vectorized_datasets["train"])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    ###
    # Evaluation Phase
    ###
    results = {}
    logger.info("*** Evaluation Phase ***")
    metrics = trainer.evaluate(eval_dataset=vectorized_datasets["valid"])
    metrics["eval_samples"] = len(vectorized_datasets["valid"])
    
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
#####
# Entry Point
#####
def main():
    # 檢查autograd
    # torch.autograd.set_detect_anomaly(True)
    ###
    # Parsing & Initialization
    ###
    # Parse argument
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    print(parser)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set random seed
    set_seed(training_args.seed)
    
    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    ###
    # Prepare logger
    ###
    # Init logging
    os.makedirs("./log", exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(
            "./log/log__{}".format(model_args.model_name_or_path.replace("/", "_")), mode="w")],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to warn of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity(transformers.logging.WARNING)
    logger.info("Training/evaluation parameters %s", training_args)
    
    ###
    # RUN RUN RUN!!!
    ###
    run(model_args, data_args, training_args)
    
if __name__ == '__main__':
    main()