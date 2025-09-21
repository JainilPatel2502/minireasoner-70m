import sentencepiece as spm
spm.SentencePieceTrainer.Train(
    input="corpus.txt",
    model_prefix="tokenizer",
    vocab_size=8192,
    model_type="bpe",
    character_coverage=1.0,
    byte_fallback=True,
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
    split_digits=True,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    add_dummy_prefix=True,
    allow_whitespace_only_pieces=True,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
)

print("Tokenizer trained with Python formatting tokens preserved")