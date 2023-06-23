## 使用

* **前置 deepspeed==0.7.0 pytorch-lightning==1.9.2 torch 1.13.1+cu117**
* 文件： https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v4neo (最新)
* **Prompt** for testing Q&A of LLMs. (found by minimizing ChatGPT ppls for RWKV 1.5B)
```python
prompt = f'\nQ & A\n\nQuestion:\n{qq}\n\nDetailed Expert Answer:\n' # let the model generate after this
```

### Inference

* **运行 RWKV-4 Pile models:** Download models from https://huggingface.co/BlinkDL. Set TOKEN_MODE = 'pile' in run.py and run it. It's fast even on CPU (the default mode).
* **Colab for RWKV-4 Pile 1.5B**: https://colab.research.google.com/drive/1F7tZoPZaWJf1fsCmZ5tjw6sYHiFOYVWM
* 在 browser 中运行 https://github.com/BlinkDL/RWKV-LM/issues/7
* RWKV-4 Web Demo: https://josephrocca.github.io/rwkv-v4-web/demo/ (note: only greedy sampling for now)
* 较老的 RWKV-2版本 Run run.py in https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v2-RNN.
** 在 briwser 中运行 https://github.com/BlinkDL/AI-Writer/tree/main/docs/eng https://blinkdl.github.io/AI-Writer/eng/ 

### Training / Fine-tuning
* **前置 pip install deepspeed==0.7.0 // pip install pytorch-lightning==1.9.2 // torch 1.13.1+cu117**
* **Training RWKV-4 from scratch:** run train.py, 数据集：enwik8 dataset (unzip https://data.deepai.org/enwik8.zip). 可以使用更长的 ctxLen 对模型进行微调, 它可以快速适应更长的 ctxLens
* **Fine-tuning RWKV-4 Pile models:** 首先使用 'prepare-data.py' in https://github.com/BlinkDL/RWKV-v2-RNN-Pile/tree/main/RWKV-v3 来 tokenize .txt into train.npy data. 后使用 https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/train.py 来训练
* Colab for fine-tuning RWKV-4 Pile models: https://colab.research.google.com/github/resloved/RWKV-notebooks/blob/master/RWKV_v4_RNN_Pile_Fine_Tuning.ipynb
* **Large corpus:** 使用 https://github.com/EleutherAI/gpt-neox 来将 .jsonl 转化为 .bin / .idx
```
python tools/preprocess_data.py --input ./my_data.jsonl --output-prefix ./data/my_data --vocab ./20B_tokenizer.json --dataset-impl mmap --tokenizer-type HFTokenizer --append-eod
```
* 也可以使用这个: https://github.com/Abel2076/json2binidx_tool

sample:
```
{"text": "This is the first document."}
{"text": "Hello\nWorld"}
{"text": "1+1=2\n1+2=3\n2+2=4"}
```
会产生类似代码:
```
ss = json.dumps({"text": text}, ensure_ascii=False)
out.write(ss + "\n")
```

### 在 text embedding 的时候使用 RWKV
首先收集每个向量的每个通道的平均值+标准偏差统计量, 并对它们进行归一化, 然后训练 linear classifier


## RWKV的原理
* 来源: https://arxiv.org/abs/2105.14103

* SmallInitEmb: https://github.com/BlinkDL/SmallInitEmb 提高 embedding quality

* Token-shift: https://github.com/BlinkDL/RWKV-LM#token-shift-time-shift-mixing 适用于所有 transformers, 对于 char-level models 有用

* Head-QK: https://github.com/BlinkDL/RWKV-LM#the-head-qk-trick-learning-to-copy-and-avoid-tokens 适用于所有 transformers

* Better initilization: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v2-RNN/src/model.py).

* https://www.reddit.com/r/MachineLearning/comments/umq908/r_rwkvv2rnn_a_parallelizable_rnn_with/ 将一些参数从小模型转移到大模型, 以实现更快更好的收敛

* CUDA kernel: https://github.com/BlinkDL/RWKV-CUDA to speedup training.


## RWKV-3 的特色

 R / K / V in SA and FF 的 layer 中的 TimeMix factors 不同
```python
xx = self.time_shift(x)
xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
```

使用 preLN 而不是 postLN: 
```python
if self.layer_id == 0:
	x = self.ln0(x)
x = x + self.att(self.ln1(x))
x = x + self.ffn(self.ln2(x))
```

### The GPT mode - overview
RWKV-3 GPT mode 和其他的 preLN GPT 相似, 但是区别是 embedding 后的一个 extra LN 

```python
x = self.emb(idx)  # input: idx = token indices
x = self.ln_emb(x) # extra LN after embedding
x = x + self.att_0(self.ln_att_0(x)) # preLN
x = x + self.ffn_0(self.ln_ffn_0(x))
...
x = x + self.att_n(self.ln_att_n(x))
x = x + self.ffn_n(self.ln_ffn_n(x))
x = self.ln_head(x) # final LN before projection
x = self.head(x)    # output: x = logits
```

使用了之前的原理: https://github.com/BlinkDL/SmallInitEmb.
### ATT block

```python
B, T, C = x.size() # x = (Batch,Time,Channel)

# Mix x with the previous timestep to produce xk, xv, xr
xx = self.time_shift(x) # self.time_shift = nn.ZeroPad2d((0,0,1,-1))
xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

# Use xk, xv, xr to produce k, v, r
k = self.key(xk).transpose(-1, -2)
v = self.value(xv).transpose(-1, -2)
r = self.receptance(xr)
k = torch.clamp(k, max=60) # clamp k to avoid overflow
k = torch.exp(k)
kv = k * v

# Compute the W-curve = [e^(-n * e^time_decay), e^(-(n-1) * e^time_decay), ..., 1, e^(time_first)]
self.time_w = torch.cat([torch.exp(self.time_decay) * self.time_curve.to(x.device), self.time_first], dim=-1)
w = torch.exp(self.time_w)

# Use W to mix kv and k respectively. Add K_EPS to wk to avoid divide-by-zero
if RUN_DEVICE == 'cuda':
    wkv = TimeX.apply(w, kv, B,C,T, 0)
    wk = TimeX.apply(w, k, B,C,T, K_EPS)
else:
    w = w[:,-T:].unsqueeze(1)
    wkv = F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(kv), w, groups=C)
    wk = F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(k), w, groups=C) + K_EPS

# The RWKV formula
rwkv = torch.sigmoid(r) * (wkv / wk).transpose(-1, -2)
rwkv = self.output(rwkv) # final output projection
```

 self.key, self.receptance, self.output 矩阵一开始就被初始化为零

time_mix, time_decay, time_first 向量来自更小的模型

### FFN block

三个和普通的 PGPT 不同的看

1.  time_mix

2.  sqReLU

3. 额外的 receptance-gate
```python
# Mix x with the previous timestep to produce xk, xr
xx = self.time_shift(x)
xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

# The usual FFN operation
k = self.key(xk)
k = torch.square(torch.relu(k)) # from the Primer paper
kv = self.value(k)

# Apply an extra receptance-gate to kv
rkv = torch.sigmoid(self.receptance(xr)) * kv
return rkv
```
