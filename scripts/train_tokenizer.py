import sentencepiece as spm

data_file = (
    "/scratch/project_462000319/aralikatte/cpt/data/sv_tokenizer_training_data.txt"
)
tok_name = "custom_tokenizers/sv-tokenizer-8k"

spm.SentencePieceTrainer.train(
    input=data_file,
    model_prefix=tok_name,
    vocab_size=8_000,
    model_type="bpe",
    max_sentence_length=1073741824,
    shuffle_input_sentence="true",
    character_coverage=0.9995,
    num_threads=8,
    hard_vocab_limit="false",
    input_sentence_size=10_000_000,
    split_digits="true",
)
