import tensorflow as tf

import model
#sample.py 负责 GPT-2 的自回归生成策略：
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
       #相当于if-else
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )

    #如果 k == 0，返回原 logits。
    #否则执行 _top_k()。
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )

#Top-K 是固定取前 k 个词，但这不够灵活（有时候前 2 个词就占了 99% 的概率，取前 40 个反而引入了噪音）。
# Top-P的思想是：把词按概率从高到低加起来，只要累加概率刚超过 p（比如 0.9），后面的词就全部丢弃。

def top_p_logits(logits, p):

    
    # 获取批次大小 (batch_size)
    # logits.shape.as_list() 返回形如 [batch, vocab_size] 的列表，提取 batch 备用
    batch, _ = logits.shape.as_list()
    
    # 将原始得分 (logits) 按从大到小的顺序进行排序
    # direction='DESCENDING' 表示降序排列，axis=-1 表示在最后一个维度（词汇表维度）上操作
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    
    # 计算排序后词汇的累计概率，计算前缀和
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    
    # 寻找阈值边界的二维坐标 
    # 我们需要精确定位到累计概率恰好达到 p 的那个词的位置
    indices = tf.stack([
        tf.range(0, batch),
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1) 
    
    #提取边界值
    min_values = tf.gather_nd(sorted_logits, indices)
    
    # 过滤掉低于阈值的词汇，条件判断
    return tf.where(
        logits < min_values,
 
        tf.ones_like(logits) * -1e10,
        
        logits,
    )

def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'#有条件生成，必须有prompt
    else:
        assert context is None, 'Specify exactly one of start_token and context!'#无条件生成
        context = tf.fill([batch_size, 1], start_token)#用tf.fill创建起始context
    #step表示执行一次前向计算
    #输入:当前为给模型的token ids，之前生成过程中的KV cache
    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)#这里reuse的意思就是之前生成的前面的token的KV矩阵直接放到cache里面直接拿出来结果用，不用再算一遍
# 拿到模型预测的词表分布 (logits)
        logits = lm_output['logits'][:, :, :hparams.n_vocab]#logits的形状大概是[batch,sequence,vocab_size]，分别代表了哪一句话，哪一个token，这个token有多少个选项
        presents = lm_output['present']
        #更新KVcache
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        def body(past, prev, output):
            next_outputs = step(hparams, prev, past=past)
            #每个位置都会预测下一个 token。生成时只关心当前序列最后一个位置对下一个 token 的预测。
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)#除以temperature经过softmax后会影响概率分布，temperature<1:分布更尖锐，temperature:分布更平坦，更随机
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            #从logits对应的概率分布中随机采样一个token
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),#如果已经有历史缓存，就把旧 past 和当前 presents 在 sequence 维度拼起来。
                samples,
                tf.concat([output, samples], axis=1)#把新生成 token 拼到输出序列末尾。
            ]

        past, prev, output = body(None, context, context)#先手动执行生成第一步

        def cond(*args):#相当于while True
            return True
        #跑循环
        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,#也就是说条件永远允许继续，但 TensorFlow 会最多循环固定次数。
            loop_vars=[#循环变量是，past，prev，output，循环体是body，最后返回tokens，所以sample_sequence这个函数的作用是把prompt的token输入进来，然后拼接上输出的token并输出
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
