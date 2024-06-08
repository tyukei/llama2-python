from fastapi import FastAPI, Request
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = FastAPI()

# モデルとトークナイザのロード
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

@app.post("/generate/")
async def generate(request: Request):
    data = await request.json()
    input_text = data.get("text", "")
    
    # トークン化
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # モデルを使用してテキストを生成
    outputs = model.generate(inputs.input_ids, max_length=50)
    
    # 生成されたトークンをテキストにデコード
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
