import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def main():
    # モデルとトークナイザのロード
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 入力テキストの準備
    input_text = input("Enter the text to start with: ")

    # トークン化
    inputs = tokenizer(input_text, return_tensors="pt")

    # モデルを使用してテキストを生成
    outputs = model.generate(inputs.input_ids, max_length=50)

    # 生成されたトークンをテキストにデコード
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(generated_text)

if __name__ == "__main__":
    main()
