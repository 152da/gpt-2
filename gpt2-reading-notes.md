# GPT-2 代码精读笔记

## 阅读进度

- [x] `src/interactive_conditional_samples.py`
- [x] `src/encoder.py`
- [x] `src/sample.py`
- [x] `src/model.py`
- [ ] `src/generate_unconditional_samples.py`
- [ ] `download_model.py`

## 1. `src/interactive_conditional_samples.py`

### 文件作用

这个文件是 GPT-2 的“交互式条件生成”入口。

它会启动一个命令行循环：用户输入一段 prompt，GPT-2 根据这段 prompt 续写文本，然后把生成出来的 token ids 解码回自然语言文本。

整体流程：

```text
用户输入文本
  -> encoder.encode(...)
  -> prompt token ids
  -> sample.sample_sequence(...)
  -> TensorFlow Session 执行生成图
  -> generated token ids
  -> encoder.decode(...)
  -> 输出生成文本
```

### 重要导入

- `fire`：把 `interact_model(...)` 自动变成命令行工具。
- `json`：读取模型配置文件 `hparams.json`。
- `os`：处理模型目录路径。
- `numpy`：设置随机种子。
- `tensorflow`：构建和运行 GPT-2 推理计算图。
- `model`：定义 GPT-2 Transformer 模型结构。
- `sample`：定义自回归生成和采样逻辑。
- `encoder`：定义 byte-level BPE tokenizer，负责文本和 token id 的相互转换。

可以简单记成：

```text
encoder: 文本 <-> token id
sample: token 生成循环
model: GPT-2 Transformer 主体
```

### 主函数

```python
def interact_model(...):
```

这是这个文件的主函数，最后会通过：

```python
fire.Fire(interact_model)
```

暴露成命令行程序。

也就是说，命令行参数会被传进这个函数。

例如：

```bash
python src/interactive_conditional_samples.py --top_k 40 --temperature 0.7
```

等价于调用：

```python
interact_model(top_k=40, temperature=0.7)
```

### 关键参数

- `model_name`：使用哪个模型目录，默认是 `124M`。
- `seed`：随机种子，用于尽量复现生成结果。
- `nsamples`：每次 prompt 总共生成多少条样本。
- `batch_size`：一次并行生成多少条样本。
- `length`：生成 token 的数量。
- `temperature`：控制随机性。
- `top_k`：每一步只保留分数最高的 k 个候选 token。
- `top_p`：nucleus sampling，只保留累计概率达到 p 的候选 token。
- `models_dir`：模型文件所在的父目录。

### 路径处理

```python
models_dir = os.path.expanduser(os.path.expandvars(models_dir))
```

这行代码用于展开路径。

它支持：

```text
~/models
$HOME/models
```

这样用户传入模型路径时会更灵活。

### batch 检查

```python
if batch_size is None:
    batch_size = 1
```

如果用户没有正确传入 `batch_size`，就默认设为 1。

```python
assert nsamples % batch_size == 0
```

要求 `nsamples` 必须能被 `batch_size` 整除。

原因是后面会按 batch 分批生成：

```python
for _ in range(nsamples // batch_size):
```

例如：

```text
nsamples = 4
batch_size = 2
```

就会循环 2 次，每次生成 2 条，总共 4 条。

### 加载 tokenizer

```python
enc = encoder.get_encoder(model_name, models_dir)
```

这行会加载 GPT-2 的 tokenizer。

它通常读取两个文件：

```text
models/124M/encoder.json
models/124M/vocab.bpe
```

后面会用：

```python
enc.encode(raw_text)
enc.decode(out[i])
```

分别完成：

```text
文本 -> token ids
token ids -> 文本
```

### 加载模型超参数

```python
hparams = model.default_hparams()
```

先创建一份默认模型配置。

常见配置项：

- `n_vocab`：词表大小。
- `n_ctx`：最大上下文长度。
- `n_embd`：隐藏层维度。
- `n_head`：attention head 数量。
- `n_layer`：Transformer block 层数。

```python
with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))
```

再从模型目录里的 `hparams.json` 读取真实配置，并覆盖默认值。

面试表达：

> 代码先创建默认超参数，再从 checkpoint 对应的 `hparams.json` 读取真实配置，保证模型结构和权重文件匹配。

### 设置生成长度

```python
if length is None:
    length = hparams.n_ctx // 2
elif length > hparams.n_ctx:
    raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
```

如果用户没有指定 `length`，默认生成 `n_ctx // 2` 个 token。

如果用户指定的长度超过 `n_ctx`，就报错。

原因：

GPT-2 的位置编码只支持固定最大上下文长度。原始 GPT-2 通常是：

```text
n_ctx = 1024
```

所以生成长度不能超过模型的上下文窗口大小。

注意：

```text
length 是 token 数，不是字符数，也不是单词数。
```

### 创建 TensorFlow Session

```python
with tf.Session(graph=tf.Graph()) as sess:
```

这份代码使用 TensorFlow 1.x 的 graph mode。

核心思想是：

```text
先搭计算图
再用 sess.run(...) 执行计算图
```

可以理解成：

```text
tf.Graph(): 图纸
tf.Session(): 按图纸运行的机器
```

### 定义输入 placeholder

```python
context = tf.placeholder(tf.int32, [batch_size, None])
```

定义 prompt token ids 的输入位置。

形状：

```text
[batch_size, sequence_length]
```

其中第二维是 `None`，因为用户输入的 prompt 长度是不固定的。

例如用户输入：

```text
Hello world
```

假设编码后是：

```python
[15496, 995]
```

如果 `batch_size = 1`，喂给模型的数据形状就是：

```python
[[15496, 995]]
```

### 设置随机种子

```python
np.random.seed(seed)
tf.set_random_seed(seed)
```

设置 NumPy 和 TensorFlow 的随机种子。

生成文本时会进行随机采样，所以设置种子可以让结果更容易复现。

不过在不同硬件、不同 TensorFlow 版本下，完全一致并不一定能保证。

### 构建生成图

```python
output = sample.sample_sequence(
    hparams=hparams, length=length,
    context=context,
    batch_size=batch_size,
    temperature=temperature, top_k=top_k, top_p=top_p
)
```

这是这个文件最关键的一段。

它调用 `sample.py` 里的 `sample_sequence(...)`，构建自回归生成逻辑。

注意：

```text
这里还没有真正生成文本。
这里只是在定义“之后 sess.run 时应该怎么生成”。
```

生成图大致做这些事：

```text
输入 context token ids
  -> 调用 model.model(...)
  -> 得到 logits
  -> 用 temperature 调整 logits
  -> 用 top_k / top_p 过滤候选 token
  -> 从分布中采样下一个 token
  -> 把新 token 接到输出序列后面
  -> 循环直到达到 length
```

### 加载模型权重

```python
saver = tf.train.Saver()
```

创建一个 Saver，用来恢复 checkpoint。

```python
ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
```

找到模型目录中最新的 checkpoint。

```python
saver.restore(sess, ckpt)
```

把训练好的 GPT-2 参数加载进当前 Session。

要区分：

```text
sample.sample_sequence(...) / model.model(...): 定义模型结构
saver.restore(...): 加载训练好的模型参数
```

如果只定义结构但不加载 checkpoint，模型就没有训练好的权重。

### 进入交互循环

```python
while True:
```

启动无限循环。

程序不会生成一次就退出，而是不断等待用户输入新的 prompt。

```python
raw_text = input("Model prompt >>> ")
```

从命令行读取用户输入。

### 防止空 prompt

```python
while not raw_text:
    print('Prompt should not be empty!')
    raw_text = input("Model prompt >>> ")
```

如果用户直接回车，`raw_text` 是空字符串。

这里会提示用户重新输入，避免空 prompt。

### 文本编码

```python
context_tokens = enc.encode(raw_text)
```

把用户输入的文本转成 token ids。

例如：

```text
Hello world
```

可能被编码成：

```python
[15496, 995]
```

GPT-2 不能直接处理字符串，只能处理 token ids。

### 样本计数

```python
generated = 0
```

记录当前 prompt 已经生成了多少条样本。

每输入一个新的 prompt，这个计数都会重新从 0 开始。

### 分批生成

```python
for _ in range(nsamples // batch_size):
```

按 batch 分批生成。

例如：

```text
nsamples = 4
batch_size = 2
```

那就循环 2 次，每次并行生成 2 条，总共生成 4 条。

### 真正执行生成

```python
out = sess.run(output, feed_dict={
    context: [context_tokens for _ in range(batch_size)]
})[:, len(context_tokens):]
```

这是实际生成发生的地方。

拆开看：

```python
sess.run(output, feed_dict={...})
```

表示运行前面构建好的 TensorFlow 计算图。

```python
feed_dict={
    context: [context_tokens for _ in range(batch_size)]
}
```

表示把 prompt token ids 喂给 `context` placeholder。

如果：

```python
context_tokens = [15496, 995]
batch_size = 3
```

那么实际喂入的数据是：

```python
[
    [15496, 995],
    [15496, 995],
    [15496, 995],
]
```

也就是同一个 prompt 并行生成 3 个结果。

### 为什么要切片

```python
[:, len(context_tokens):]
```

`sample_sequence(...)` 返回的序列包含：

```text
原始 prompt tokens + 新生成 tokens
```

但是打印时通常只想打印模型新生成的部分，所以要切掉 prompt。

例如：

```python
context_tokens = [10, 20]
sess.run(...) 返回:
[
    [10, 20, 31, 42, 53]
]
```

切片：

```python
[:, 2:]
```

得到：

```python
[
    [31, 42, 53]
]
```

这就是只保留生成内容。

### 解码并打印

```python
for i in range(batch_size):
```

遍历当前 batch 中的每条生成结果。

```python
generated += 1
```

样本计数加 1。

```python
text = enc.decode(out[i])
```

把生成的 token ids 转回自然语言文本。

```python
print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
print(text)
```

打印样本编号和生成文本。

```python
print("=" * 80)
```

当前 prompt 的所有样本生成完后，打印分隔线。

### 脚本入口

```python
if __name__ == '__main__':
    fire.Fire(interact_model)
```

当这个文件被直接运行时，执行 `fire.Fire(interact_model)`。

它会把 `interact_model` 变成命令行接口。

所以可以这样运行：

```bash
python src/interactive_conditional_samples.py --temperature 0.7 --top_k 40
```

### 面试总结

`interactive_conditional_samples.py` 是 GPT-2 的条件生成推理入口。它先加载 tokenizer 和模型超参数，再用 TensorFlow 1.x graph mode 构建自回归生成图，并通过 `Saver` 从 checkpoint 恢复训练好的模型权重。运行时，用户输入的 prompt 会被 `encoder.encode(...)` 转成 token ids，通过 `feed_dict` 喂给 `context` placeholder。`sess.run(...)` 执行生成图，得到包含 prompt 和续写内容的 token 序列。最后代码切掉 prompt 部分，用 `encoder.decode(...)` 把生成 token ids 转回文本并打印。

### 当前已经理解的点

- 这个文件是推理入口，不是模型主体实现。
- `encoder.encode(...)` 负责把用户输入文本变成 token ids。
- `sample.sample_sequence(...)` 负责构建自回归生成图。
- `sess.run(...)` 才是真正执行生成的地方。
- `feed_dict` 负责把 prompt tokens 喂给 `context` placeholder。
- `output` 里包含 prompt tokens 和 generated tokens。
- `[:, len(context_tokens):]` 用来切掉 prompt，只保留模型新生成的 token。
- `encoder.decode(...)` 负责把 generated token ids 转回自然语言文本。
- `fire.Fire(interact_model)` 让这个函数可以通过命令行参数调用。

## 下一步阅读

### 推荐阅读：`src/encoder.py`

下一步读 `src/encoder.py`。

原因是 `interactive_conditional_samples.py` 里最重要的几处文本处理都依赖它：

```python
enc = encoder.get_encoder(...)
context_tokens = enc.encode(raw_text)
text = enc.decode(out[i])
```

读完 `encoder.py` 后，你应该能回答：

- 原始文本是怎么变成 token ids 的？
- GPT-2 为什么使用 byte-level BPE？
- `encoder.json` 是什么？
- `vocab.bpe` 是什么？
- `encode(...)` 的流程是什么？
- `decode(...)` 怎么恢复文本？

后续推荐顺序：

```text
src/encoder.py
  -> src/sample.py
  -> src/model.py
```

这个顺序更自然：

```text
交互入口
  -> 文本如何变成 token
  -> token 如何逐步生成
  -> Transformer 内部如何计算
```

## 2. `src/encoder.py`

### 文件作用

`encoder.py` 是 GPT-2 的 tokenizer 实现。

它负责两件事：

```text
encode: 原始文本 -> token ids
decode: token ids -> 原始文本
```

在 `interactive_conditional_samples.py` 里，我们已经看到它被这样使用：

```python
context_tokens = enc.encode(raw_text)
text = enc.decode(out[i])
```

所以 `encoder.py` 是文本进入 GPT-2 模型之前的第一道处理流程。

### GPT-2 为什么用 byte-level BPE

普通词表如果没有见过某个字符，可能会产生 `<UNK>`。

GPT-2 的做法是：

```text
任意文本
  -> UTF-8 bytes
  -> 每个 byte 都是 0~255 之间的数字
  -> byte-to-unicode 映射
  -> BPE 合并
  -> token id
```

这样做的好处是：只要文本能被 UTF-8 表示，就一定能变成 bytes；而 byte 的取值只有 256 种，所以 GPT-2 可以覆盖非常多语言、符号和 emoji，尽量避免 unknown token。

### `bytes_to_unicode()`

这个函数建立一个可逆映射：

```text
byte 数字 -> Unicode 字符
```

byte 是 8 个 bit，所以一共有：

```text
2^8 = 256
```

种可能，也就是：

```text
0, 1, 2, ..., 255
```

例如：

```text
"A" 的 UTF-8 byte 是 65
"!" 的 UTF-8 byte 是 33
"\n" 的 byte 是 10
"你" 的 UTF-8 bytes 是 [228, 189, 160]
```

`bytes_to_unicode()` 会给 0 到 255 的每个 byte 都分配一个唯一 Unicode 字符。

为什么不直接用原始 byte 对应的字符？

因为有些 byte 对应空格、换行、制表符、控制字符。这些字符会干扰 BPE 文件格式和字符串处理。例如换行会破坏行结构，空格会和 BPE merge 规则里的分隔符冲突。

所以代码先选择一批安全可见字符：

```python
bs = list(range(ord("!"), ord("~")+1)) \
   + list(range(ord("\xa1"), ord("\xac")+1)) \
   + list(range(ord("\xae"), ord("\xff")+1))
```

然后把剩下不安全的 byte 映射到 256 以上的 Unicode 字符。

最终效果是：

```text
每个 byte 都有一个安全、可逆、适合字符串处理的 Unicode 表示。
```

### `get_pairs(word)`

`get_pairs(...)` 用来找一个 word 里面所有相邻符号对。

例如：

```python
word = ("l", "o", "w")
```

得到：

```python
{("l", "o"), ("o", "w")}
```

BPE 的核心操作就是不断选择一个相邻 pair，然后把它合并成一个更大的符号。

### `Encoder.__init__`

初始化时主要建立几张表：

```python
self.encoder = encoder
```

保存：

```text
BPE token 字符串 -> token id
```

它来自 `encoder.json`。

```python
self.decoder = {v:k for k,v in self.encoder.items()}
```

建立反向映射：

```text
token id -> BPE token 字符串
```

用于 decode。

```python
self.byte_encoder = bytes_to_unicode()
self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
```

建立 byte 和 Unicode 字符之间的双向映射。

```python
self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
```

建立 BPE merge pair 的优先级表。

`bpe_ranks` 的优先级来自 `vocab.bpe` 中 merge 规则的顺序：

```text
越靠前，rank 越小，优先级越高。
```

假设 `vocab.bpe` 内容是：

```text
#version: 0.2
l o
lo w
e r
low er
```

那么会得到：

```python
{
    ("l", "o"): 0,
    ("lo", "w"): 1,
    ("e", "r"): 2,
    ("low", "er"): 3,
}
```

### 正则切分规则

```python
self.pat = re.compile(...)
```

这个正则会先把原始文本切成粗粒度片段。

它会识别：

- 英文缩写，比如 `'s`、`'t`、`'re`。
- 字母串。
- 数字串。
- 标点符号。
- 空白字符。

重要特点：

```text
GPT-2 经常把空格和后面的词绑定在一起处理。
```

例如：

```text
"Hello world!"
```

可能被切成概念上的：

```python
["Hello", " world", "!"]
```

注意 `" world"` 前面带空格。

这也是为什么 GPT-2 里 `"world"` 和 `" world"` 可能是不同 token。

### `bpe(token)`

`bpe(...)` 的作用是：对一个 byte-to-unicode 后的字符串执行 BPE 合并。

流程：

```text
1. 如果 token 已经处理过，直接返回 cache。
2. 把 token 拆成字符 tuple。
3. 找所有相邻 pair。
4. 从 pair 中选择 bpe_ranks 最小的 pair。
5. 如果这个 pair 在 merge 表中，就合并。
6. 合并后重新计算 pair。
7. 重复直到不能继续合并。
8. 用空格连接最终 BPE token。
```

具体例子：

假设当前 token 是：

```python
"low"
```

初始：

```python
("l", "o", "w")
```

相邻 pair：

```python
{("l", "o"), ("o", "w")}
```

如果 `bpe_ranks` 中：

```python
("l", "o") -> 0
("lo", "w") -> 1
```

那么先合并：

```text
("l", "o", "w")
  -> ("lo", "w")
```

再合并：

```text
("lo", "w")
  -> ("low")
```

最终可能得到：

```python
"low"
```

如果某些 pair 没有可用 merge，就会停下来，保留多个 BPE 子词。

### `encode(text)`

`encode(...)` 把自然语言文本变成 token ids。

核心流程：

```text
原始文本
  -> 正则切分
  -> UTF-8 bytes
  -> byte-to-unicode 字符串
  -> BPE 合并
  -> 查 encoder.json
  -> token ids
```

具体例子，假设输入：

```text
Hello 你
```

第一步，正则切分，概念上可能得到：

```python
["Hello", " 你"]
```

第二步，每个片段转成 UTF-8 bytes：

```text
"Hello" -> [72, 101, 108, 108, 111]
" 你"   -> [32, 228, 189, 160]
```

注意中文 `"你"` 在 UTF-8 中是 3 个 byte。

第三步，把每个 byte 通过 `byte_encoder` 映射成 Unicode 字符串。

第四步，对这些 Unicode 字符串执行 BPE 合并。

第五步，对每个 BPE token 查 `encoder.json`，得到 token id。

最终得到类似：

```python
[15496, 220, ...]
```

这里具体 id 要以真实 `encoder.json` 为准。

### `decode(tokens)`

`decode(...)` 把 token ids 还原成文本。

流程和 encode 相反：

```text
token ids
  -> 查 decoder 得到 BPE token 字符串
  -> 拼接成 byte-to-unicode 字符串
  -> 用 byte_decoder 转回 byte 数字
  -> bytearray
  -> UTF-8 decode
  -> 原始文本
```

例如，假设 token ids 对应的 BPE 字符串最终还原出 bytes：

```text
[72, 101, 108, 108, 111, 32, 228, 189, 160]
```

UTF-8 解码后就是：

```text
Hello 你
```

### `get_encoder(model_name, models_dir)`

这个函数从模型目录加载 tokenizer 文件。

默认会读取：

```text
models/124M/encoder.json
models/124M/vocab.bpe
```

`encoder.json`：

```text
BPE token 字符串 -> token id
```

`vocab.bpe`：

```text
BPE merge 规则，越靠前优先级越高
```

代码：

```python
bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
```

含义：

- `[1:-1]` 跳过第一行版本号和最后一个空行。
- 每一行 merge 规则被 split 成一个 pair。

例如：

```text
l o
```

变成：

```python
("l", "o")
```

最后返回：

```python
Encoder(encoder=encoder, bpe_merges=bpe_merges)
```

### 面试总结

`encoder.py` 实现的是 GPT-2 的 byte-level BPE tokenizer。编码时，它先用正则把文本切成片段，再把片段转成 UTF-8 bytes。由于每个 byte 都是 0 到 255 的数字，代码通过 `bytes_to_unicode()` 把每个 byte 映射成一个安全且可逆的 Unicode 字符。随后根据 `vocab.bpe` 中的 merge 顺序执行 BPE 合并，并通过 `encoder.json` 把 BPE token 映射成 token id。解码时则反过来，从 token id 找回 BPE token，再还原成 bytes，最后 UTF-8 解码成文本。

### 当前已经理解的点

- GPT-2 不直接对原始 Unicode 字符做 BPE，而是先转成 UTF-8 bytes。
- byte 的范围是 0 到 255，所以只需要覆盖 256 种基础 byte。
- `bytes_to_unicode()` 建立 byte 到 Unicode 字符的可逆映射。
- 这样做可以避免空格、换行、控制字符直接干扰 BPE 文件格式。
- `vocab.bpe` 决定 BPE pair 的合并顺序。
- `bpe_ranks` 是由 `vocab.bpe` 的行号顺序生成的。
- `encoder.json` 负责把 BPE token 字符串映射到 token id。
- `encode(...)` 是文本到 token ids。
- `decode(...)` 是 token ids 到文本。

## 3. `src/sample.py`

### 文件作用

`sample.py` 负责 GPT-2 的自回归生成逻辑。

它不实现 Transformer 内部结构，而是负责：

```text
已有 token ids
  -> 调用 model.model(...)
  -> 得到 logits
  -> temperature / top-k / top-p
  -> 采样下一个 token
  -> 拼回输出序列
  -> 循环生成
```

也就是说：

```text
encoder.py 负责“文本怎么变成 token”
sample.py 负责“token 怎么一个接一个生成”
model.py 负责“Transformer 怎么算 logits”
```

### `top_k_logits(logits, k)`

这个函数实现 top-k 过滤。

输入：

```text
logits: 每个 token 的未归一化分数
k: 只保留分数最高的 k 个 token
```

例子：

```python
logits = [1.2, 0.1, 3.5, 2.0, -1.0]
k = 2
```

分数最高的两个是：

```text
3.5 和 2.0
```

过滤后变成：

```python
[-1e10, -1e10, 3.5, 2.0, -1e10]
```

`-1e10` 是一个非常小的数，softmax 后概率几乎为 0。

所以 top-k 的含义是：

```text
每一步只允许从最可能的 k 个 token 中采样。
```

如果：

```python
k = 0
```

就表示不做 top-k 限制。

### `top_p_logits(logits, p)`

这个函数实现 top-p，也叫 nucleus sampling。

top-k 是固定保留 k 个 token，top-p 是保留累计概率达到 p 的一组 token。

例子：

```text
排序后的 token 概率：
A: 0.50
B: 0.20
C: 0.15
D: 0.10
E: 0.05
```

如果：

```text
p = 0.8
```

那么会保留：

```text
A + B + C = 0.85
```

后面的 D 和 E 被过滤。

代码流程：

```text
1. 按 logits 从大到小排序。
2. 对排序后的 logits 做 softmax。
3. 计算累计概率 cumulative_probs。
4. 找到累计概率边界。
5. 用边界 logit 作为阈值。
6. 低于阈值的 token 设为 -1e10。
```

top-p 的好处是候选集合大小会动态变化：

```text
模型很确定时，候选 token 少。
模型不确定时，候选 token 多。
```

### `sample_sequence(...)`

这是 `sample.py` 的主函数。

它负责构建完整生成图。

参数重点：

- `hparams`：模型超参数。
- `length`：生成长度。
- `start_token`：无条件生成时的起始 token。
- `context`：条件生成时的 prompt token ids。
- `batch_size`：一次生成几条。
- `temperature`：控制随机性。
- `top_k`：top-k 过滤。
- `top_p`：top-p 过滤。

这里要求：

```text
start_token 和 context 必须二选一。
```

两种模式：

```text
条件生成：使用 context，也就是用户 prompt。
无条件生成：使用 start_token，通常是 <|endoftext|>。
```

### `step(hparams, tokens, past=None)`

`step(...)` 表示执行一次模型前向计算。

核心代码：

```python
lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
```

输入：

```text
tokens: 当前喂给模型的 token ids
past: 历史 KV cache
```

输出：

```text
logits: 当前 token 对下一个 token 的预测分数
presents: 当前 step 新产生的 KV cache
```

`reuse=tf.AUTO_REUSE` 很重要，因为生成循环会多次调用 `model.model(...)`，但每次都应该复用同一套模型权重。

### `past` 和 `present`

在生成过程中：

```text
past: 之前步骤缓存下来的 key/value
present: 当前步骤新算出来的 key/value
```

下一轮会把：

```text
旧 past + 当前 present
```

拼起来，作为新的 past。

这样模型不用每次重新计算完整 prompt 的 attention key/value。

它的作用可以理解成：

```text
KV cache，用空间换时间，加速自回归生成。
```

### `body(past, prev, output)`

`body(...)` 是生成循环每一步的主体。

三个循环状态：

```text
past: 历史 KV cache
prev: 上一步生成的 token，下一步要喂给模型
output: 当前完整输出序列
```

每一步做：

```text
1. 用 prev 和 past 调用模型。
2. 取最后一个位置的 logits。
3. 除以 temperature。
4. 应用 top-k。
5. 应用 top-p。
6. 用 tf.multinomial 采样一个 token。
7. 更新 past。
8. 把采样 token 拼到 output 后面。
```

### 为什么取 `[:, -1, :]`

代码：

```python
logits = next_outputs['logits'][:, -1, :] / tf.to_float(temperature)
```

`model.model(...)` 会对序列中每个位置都输出 logits。

形状：

```text
[batch, sequence, vocab]
```

但生成下一个 token 时，只需要最后一个位置的预测。

所以：

```python
[:, -1, :]
```

表示取每个 batch 中最后一个 token 位置的 logits。

### temperature 的作用

代码：

```python
logits / temperature
```

效果：

```text
temperature < 1: 分布更尖锐，生成更保守。
temperature = 1: 使用原始分布。
temperature > 1: 分布更平坦，生成更随机。
```

### `tf.multinomial`

代码：

```python
samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
```

它从 logits 对应的概率分布里随机采样一个 token。

输出形状：

```text
[batch, 1]
```

例如：

```python
[[31]]
```

表示当前 batch 采样到了 token id `31`。

### `tf.while_loop`

`tf.while_loop` 用来在 TensorFlow 1.x graph mode 中循环生成 token。

循环变量：

```text
past
prev
output
```

每轮调用：

```python
body(past, prev, output)
```

然后得到新的：

```text
past
prev
output
```

`shape_invariants` 告诉 TensorFlow：

```text
sequence 维度会随着生成过程变长。
```

`back_prop=False` 表示不需要反向传播，因为这是推理生成，不是训练。

### 一个完整生成例子

假设：

```python
context = [[10, 20]]
length = 4
batch_size = 1
```

第 1 步，先手动执行：

```python
past, prev, output = body(None, context, context)
```

输入：

```text
prev = [[10, 20]]
```

模型采样出：

```text
[[31]]
```

于是：

```text
output = [[10, 20, 31]]
prev = [[31]]
```

然后 `tf.while_loop` 再生成 `length - 1` 次。

可能得到：

```text
第 2 个新 token: 42
第 3 个新 token: 53
第 4 个新 token: 64
```

最终：

```python
[[10, 20, 31, 42, 53, 64]]
```

注意返回结果包含：

```text
原始 prompt tokens + 生成 tokens
```

所以在 `interactive_conditional_samples.py` 中要用：

```python
[:, len(context_tokens):]
```

切掉 prompt，只保留生成部分。

### 面试总结

`sample.py` 实现 GPT-2 的自回归采样生成。它接收 prompt token ids 或 start token，每一步调用 `model.model(...)` 得到 logits 和 present KV cache。生成时只取最后一个位置的 logits，通过 temperature 调整分布，再用 top-k 和 top-p 过滤候选 token，最后用 `tf.multinomial` 采样下一个 token。采样出的 token 会拼接到 output 后面，同时 present 会和历史 past 拼接成新的 KV cache，用于下一步生成。整个过程通过 TensorFlow 1.x 的 `tf.while_loop` 完成。

### 当前已经理解的点

- `sample.py` 是生成策略，不是 Transformer 结构本身。
- `top_k_logits(...)` 保留分数最高的 k 个 token。
- `top_p_logits(...)` 保留累计概率达到 p 的候选集合。
- `temperature` 控制生成分布的尖锐或平坦程度。
- `sample_sequence(...)` 是自回归生成主函数。
- `step(...)` 每次调用模型，拿到 logits 和 KV cache。
- `past/present` 是 KV cache，用来减少重复计算。
- `body(...)` 是每一步生成逻辑。
- `tf.while_loop` 在图模式中循环生成 token。
- `sample_sequence(...)` 返回的是 prompt tokens 加 generated tokens。

## 最新下一步阅读

### 推荐阅读：`src/model.py`

现在已经读完：

```text
interactive_conditional_samples.py
  -> encoder.py
  -> sample.py
```

下一步读：

```text
src/model.py
```

原因是 `sample.py` 里最关键的一句是：

```python
lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
```

也就是说，真正计算 logits 和 KV cache 的 Transformer 主体在 `model.py` 里。

读 `model.py` 时重点关注：

- `default_hparams()`：模型超参数。
- `conv1d(...)`：实际是线性层。
- `norm(...)`：layer normalization。
- `attention_mask(...)`：causal mask。
- `attn(...)`：multi-head masked self-attention。
- `mlp(...)`：前馈网络。
- `block(...)`：一个 Transformer block。
- `model(...)`：完整 GPT-2 前向流程。

读完 `model.py`，你就能把 GPT-2 的主链路完整串起来：

```text
文本
  -> tokenizer
  -> token ids
  -> 自回归生成循环
  -> Transformer 前向计算
  -> logits
  -> 采样 token
  -> decode 回文本
```

## 4. `src/model.py`

### 文件作用

`model.py` 是 GPT-2 的模型主体实现。

它负责把 token ids 计算成 logits：

```text
token ids
  -> token embedding + position embedding
  -> 多层 Transformer block
  -> final LayerNorm
  -> vocabulary logits
```

在 `sample.py` 中最关键的一句就是调用这里：

```python
lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
```

`model.py` 输出：

```text
logits: 每个位置对下一个 token 的预测分数
present: 当前 forward 产生的 KV cache
```

### GPT-2 是 decoder-only，不是 encoder-decoder

GPT-2 没有 Transformer 原论文里的 Encoder。

它的结构是：

```text
decoder-only Transformer
```

原因是 GPT-2 的任务是自回归语言建模：

```text
给定前文 token_1 ... token_n
预测下一个 token_{n+1}
```

它不需要像机器翻译那样：

```text
源语言句子 -> Encoder
目标语言句子 -> Decoder
```

GPT-2 的输入和输出属于同一条文本序列，只是错开一位：

```text
输入: Once upon a time
目标: upon a time there
```

所以它只需要 masked self-attention：

```text
当前位置只能看自己和左边的 token，不能看未来 token。
```

### `default_hparams()`

默认超参数：

```python
n_vocab=0
n_ctx=1024
n_embd=768
n_head=12
n_layer=12
```

含义：

- `n_vocab`：词表大小，真实值从 `hparams.json` 覆盖。
- `n_ctx`：最大上下文长度。
- `n_embd`：hidden size。
- `n_head`：attention head 数量。
- `n_layer`：Transformer block 层数。

GPT-2 small 常见配置：

```text
n_layer = 12
n_head = 12
n_embd = 768
head_dim = 768 / 12 = 64
```

### `shape_list(x)`

TensorFlow 1.x 中有静态 shape 和动态 shape。

例如：

```python
context = tf.placeholder(tf.int32, [batch_size, None])
```

第二维是 `None`，只有运行时才知道。

`shape_list(...)` 的作用是：

```text
如果静态 shape 已知，就用静态 shape；
如果某一维是 None，就用 tf.shape(x) 动态获取。
```

### `softmax(x)`

这是手写 softmax。

代码先做：

```python
x = x - tf.reduce_max(x, axis=axis, keepdims=True)
```

目的是数值稳定，避免 `exp(x)` 太大溢出。

在 attention 中，它会把 attention score 转成概率。

### `gelu(x)`

GELU 是 GPT-2 MLP 里的激活函数。

MLP 结构大致是：

```text
Linear
  -> GELU
  -> Linear
```

GELU 比 ReLU 更平滑，是 GPT 系列常见激活函数。

### `norm(x, scope, ...)`

这是 LayerNorm。

它做：

```text
x -> 减均值 -> 除以标准差 -> 乘 g -> 加 b
```

其中：

```python
g = tf.get_variable('g', ...)
b = tf.get_variable('b', ...)
```

是可学习参数。

在 GPT-2 block 中使用的是 pre-norm：

```text
先 LayerNorm，再进入 attention / MLP。
```

### `split_states(x, n)` 和 `merge_states(x)`

这两个函数用于处理多头注意力的维度。

例子：

```text
[batch, seq, 768]
```

`split_states(x, 12)` 后：

```text
[batch, seq, 12, 64]
```

因为：

```text
768 / 12 = 64
```

`merge_states(...)` 做反向操作：

```text
[batch, seq, 12, 64] -> [batch, seq, 768]
```

### `conv1d(x, scope, nf)`

虽然名字叫 `conv1d`，但这里实际等价于逐 token 的线性层。

如果输入：

```text
x: [batch, seq, nx]
```

输出：

```text
[batch, seq, nf]
```

内部核心是矩阵乘法：

```text
[batch * seq, nx] @ [nx, nf]
```

所以它本质上是：

```text
Linear layer
```

GPT-2 原版代码沿用了 `conv1d` 这个命名。

### `attention_mask(nd, ns)`

这个函数生成 causal mask。

作用：

```text
让每个位置只能看自己和之前的位置，不能看未来。
```

例子，如果没有 past，序列长度是 4，mask 类似：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

第 1 个 token 只能看第 1 个 token。

第 4 个 token 可以看前 4 个 token。

这就是 GPT-2 自回归生成不能偷看未来的关键。

### `attn(...)`

这是 multi-head masked self-attention。

输入：

```text
x: [batch, sequence, hidden]
```

内部先通过一层线性层同时生成 Q/K/V：

```python
c = conv1d(x, 'c_attn', n_state*3)
q, k, v = map(split_heads, tf.split(c, 3, axis=2))
```

如果：

```text
hidden = 768
```

那么：

```text
c: [batch, seq, 2304]
q/k/v: [batch, seq, 768]
split heads 后: [batch, heads, seq, head_dim]
```

然后计算 scaled dot-product attention：

```text
score = Q @ K^T / sqrt(head_dim)
```

再应用 causal mask：

```text
未来位置分数变成极小值，softmax 后概率接近 0。
```

最后：

```text
attention_probs @ V
```

得到 attention 输出。

### `past` 和 `present`

在 attention 中：

```python
present = tf.stack([k, v], axis=1)
```

保存当前 step 产生的 key/value。

单层形状：

```text
[batch, 2, heads, sequence, head_dim]
```

其中：

```text
2 = key + value
```

如果传入了 `past`：

```python
pk, pv = tf.unstack(past, axis=1)
k = tf.concat([pk, k], axis=-2)
v = tf.concat([pv, v], axis=-2)
```

就把历史 key/value 和当前 key/value 拼起来。

这样生成下一个 token 时不用重新计算完整上下文的 K/V。

这就是 KV cache。

### `mlp(...)`

Transformer block 中的前馈网络。

在 block 里调用时：

```python
m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
```

所以 hidden size 会先扩展 4 倍：

```text
768 -> 3072 -> 768
```

结构：

```text
Linear
  -> GELU
  -> Linear
```

### `block(...)`

这是一个 GPT-2 Transformer block。

代码：

```python
a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
x = x + a
m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
x = x + m
return x, present
```

对应结构：

```text
x
  -> LayerNorm
  -> Masked Multi-Head Self-Attention
  -> Residual Add
  -> LayerNorm
  -> MLP
  -> Residual Add
```

重点：

```text
这是 pre-norm block。
```

也就是 LayerNorm 在 attention 和 MLP 之前。

### `past_shape(...)`

完整模型的 KV cache shape：

```text
[batch, n_layer, 2, n_head, sequence, head_dim]
```

GPT-2 small 中：

```text
[batch, 12, 2, 12, sequence, 64]
```

这个 shape 会在 `sample.py` 里的 `tf.while_loop` 中用到。

### `positions_for(tokens, past_length)`

这个函数负责生成 position ids。

如果没有 past：

```text
past_length = 0
tokens 长度 = 3
positions = [0, 1, 2]
```

如果已经生成过 5 个 token：

```text
past_length = 5
tokens 长度 = 2
positions = [5, 6]
```

这样增量生成时，位置编号会连续，不会每次都从 0 开始。

### `model(...)`

这是完整 GPT-2 前向入口。

输入：

```text
X: token ids, shape [batch, sequence]
past: 历史 KV cache，可选
```

第一步，创建 embedding：

```python
wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd])
wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd])
```

含义：

```text
wte: token embedding
wpe: position embedding
```

第二步，查 embedding 并相加：

```python
h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))
```

形状：

```text
X: [batch, sequence]
token embedding: [batch, sequence, n_embd]
position embedding: [batch, sequence, n_embd]
h: [batch, sequence, n_embd]
```

第三步，循环通过所有 Transformer block：

```python
for layer, past in enumerate(pasts):
    h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
    presents.append(present)
```

第四步，堆叠每层 present：

```python
results['present'] = tf.stack(presents, axis=1)
```

形状：

```text
[batch, n_layer, 2, n_head, sequence, head_dim]
```

第五步，final LayerNorm：

```python
h = norm(h, 'ln_f')
```

第六步，输出 logits：

```python
h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
logits = tf.matmul(h_flat, wte, transpose_b=True)
logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
```

这里用了 weight tying：

```text
输入 token embedding wte
输出 vocabulary projection 也用 wte 的转置
```

输出 logits 形状：

```text
[batch, sequence, vocab]
```

### 一个完整前向例子

假设：

```text
batch = 1
sequence = 3
n_embd = 768
n_head = 12
n_layer = 12
n_vocab = 50257
```

输入：

```python
X = [[10, 20, 30]]
```

embedding 后：

```text
h: [1, 3, 768]
```

进入第 1 层 block：

```text
Q/K/V: [1, 12, 3, 64]
attention 输出: [1, 3, 768]
MLP 输出: [1, 3, 768]
```

经过 12 层后：

```text
h: [1, 3, 768]
```

最终 logits：

```text
[1, 3, 50257]
```

表示：

```text
每个位置都对整个词表给出一个预测分数。
```

在生成时，`sample.py` 会取最后一个位置：

```python
logits[:, -1, :]
```

用它采样下一个 token。

### 面试总结

`model.py` 实现 GPT-2 的 decoder-only Transformer 前向计算。输入 token ids 后，模型先查 token embedding `wte` 和 position embedding `wpe` 并相加。随后 hidden states 经过多层 pre-norm Transformer block。每个 block 包含 masked multi-head self-attention、残差连接、MLP 和残差连接。attention 通过 causal mask 防止当前位置看到未来 token，并通过 `past/present` 实现 KV cache。最后模型经过 final LayerNorm，并使用输入 embedding `wte` 的转置作为输出投影，得到每个位置对整个词表的 logits。

### 当前已经理解的点

- GPT-2 是 decoder-only Transformer，没有原始 Transformer 的 encoder。
- `wte` 是 token embedding，`wpe` 是 position embedding。
- `block(...)` 是 GPT-2 的核心层。
- `attn(...)` 实现 masked multi-head self-attention。
- causal mask 防止模型看到未来 token。
- `mlp(...)` 是两层前馈网络，中间用 GELU。
- block 使用 pre-norm 和 residual connection。
- `past/present` 是 KV cache。
- `positions_for(...)` 保证增量生成时位置编号连续。
- 输出层复用 `wte`，这是 weight tying。
- `logits` 形状是 `[batch, sequence, vocab]`。

## 架构阅读状态

到这里，GPT-2 推理代码主架构已经读完了。

已经完成的主链路：

```text
interactive_conditional_samples.py
  -> encoder.py
  -> sample.py
  -> model.py
```

对应理解：

```text
命令行入口
  -> 文本/token 转换
  -> 自回归采样生成
  -> Transformer 前向计算
```

剩下两个文件可以作为扫尾阅读：

```text
src/generate_unconditional_samples.py
download_model.py
```

它们不是架构核心：

- `generate_unconditional_samples.py`：无 prompt 生成入口，和交互式生成很像。
- `download_model.py`：下载模型文件。

面试重点仍然是：

```text
encoder.py
sample.py
model.py
```
