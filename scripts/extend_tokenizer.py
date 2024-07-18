import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import argparse

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import LlamaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--llama_tokenizer_dir", default=None, type=str, required=True)
parser.add_argument(
    "--custom_sp_model_file", default="./sv-tokenizer-8k.model", type=str
)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
custom_sp_model_file = args.custom_sp_model_file

# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
custom_sp_model = spm.SentencePieceProcessor()
custom_sp_model.Load(custom_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
custom_spm = sp_pb2_model.ModelProto()
custom_spm.ParseFromString(custom_sp_model.serialized_model_proto())

# print number of tokens
print(len(llama_tokenizer), len(custom_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

## Add custom tokens to LLaMA tokenizer
llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before: {len(llama_spm_tokens_set)}")
for p in custom_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set and not piece.replace("▁", "").isnumeric():
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama_spm.pieces)}")

## Save
output_sp_dir = "merged_tokenizer_sp"
output_hf_dir = "merged_tokenizer_hf"
os.makedirs(output_sp_dir, exist_ok=True)
with open(output_sp_dir + "/custom_llama.model", "wb") as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(
    vocab_file=output_sp_dir + "/custom_llama.model", legacy=False
)

tokenizer.save_pretrained(output_hf_dir)
print(f"custom-LLaMA tokenizer has been saved to {output_hf_dir}")
