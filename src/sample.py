import tensorflow as tf

import model
#sample.py 负责 GPT-2 的生成策略：
# 给定 prompt token ids，反复调用模型得到 logits，再通过 temperature、top-k、top-p 采样下一个 token。
def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        #如果 k=0，表示不做 top-k 限制，直接返回原 logits。
        return logits
#logits: 每个候选 token 的原始分数
#k: 只保留分数最高的 k 个 token
    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)#取出每一行logits中最大的k个值，第二个返回值是对应token的下标，因为不需要，所以省略
        min_values = values[:, -1, tf.newaxis]#取 top-k 里面最小的那个值，作为阈值,tf.newaxis 是为了保持维度，方便后面和 logits 广播比较。
       #tf.newaxis 是为了保持维度，方便后面和 logits 广播比较,相当于if-else
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    #TensorFlow 图模式下的条件判断。

    #如果 k == 0，返回原 logits。

    #否则执行 _top_k()。
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )

#Top-K 是固定取前 $k$ 个词，但这不够灵活（有时候前 2 个词就占了 99% 的概率，取前 40 个反而引入了噪音）。
# Top-P（Nucleus Sampling）的思想是：把词按概率从高到低加起来，只要累加概率刚超过 $p$（比如 0.9），后面的词就全部丢弃。
def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape.as_list()#提取出batch_size
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)#把logit从大到小排序
    # 2. 计算累加概率：先对排序后的 logits 做 softmax 变成百分比概率，
    # 然后用 tf.cumsum 计算前缀和（累加）。
    # 比如概率是 [0.5, 0.3, 0.1, 0.1]，累加后变成 [0.5, 0.8, 0.9, 1.0]
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)#对排序后的logits做softmax，得到概率
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),#根据累计概率得到边界，然后top-p滤掉
    ], axis=-1)
    #取出边界作为阈值
    min_values = tf.gather_nd(sorted_logits, indices)
    #将小于阈值的变成-1e10
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)
    #step表示执行一次前向计算
    #输入:当前为给模型的token ids，之前生成过程中的KV cache
    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
# 拿到模型预测的词表分布 (logits)
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        # 在主循环里：
        # 把过去的老缓存 (past) 和刚算出的新缓存 (next_outputs['presents']) 拼接在一起，
        # 变成下一轮更长的新 past！
        #这样就不用每一步自回归都去算KV矩阵，直接用上一步KV矩阵的结果即可
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        def body(past, prev, output):
            next_outputs = step(hparams, prev, past=past)
            #因为语言模型每个位置都会预测下一个 token。生成时只关心当前序列最后一个位置对下一个 token 的预测。
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            #从logits对应的概率分布中随机采样一个token
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.concat([output, samples], axis=1)
            ]

        past, prev, output = body(None, context, context)

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens
