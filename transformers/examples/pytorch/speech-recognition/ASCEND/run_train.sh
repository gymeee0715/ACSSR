python3 train.py --model_name_or_path=jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn  \
    --train_manifest_path=AC_NTYT_train_metadata.csv \
    --valid_manifest_path=dev_NTUTAB.csv \
    --test_manifest_path=test_NTUTAB.csv \
    --preprocessing_num_workers=8 --audio_column_name=file_name --text_column_name=transcription \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --dataloader_num_workers=8 --dataloader_pin_memory --group_by_length \
    --seed=19 --num_train_epochs=30 --learning_rate=5e-5 \
    --logging_strategy=steps --logging_steps=50 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 --eval_delay=25\
    --save_strategy=epoch --save_steps=1    --save_total_limit=1\
    --gradient_checkpointing=True  \
    --metric_for_best_model=mer --greater_is_better=False \
    --fp16 --fp16_full_eval  --fp16_backend=cuda_amp \
    --resume_from_checkpoint=True --ignore_data_skip=True --overwrite_output_dir 
    

