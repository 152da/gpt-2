import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
#decoder only transformer，因为GPT2不需要输入，或者说输入就是输出的一部分，根据输出继续往后写
#transformer模型的输出是自回归机制的，输入是算好的，然后输出一个一个往外蹦，加上利用KV机制，就减小计算量，但是kv机制有缓存最大限制
def default_hparams():
    return HParams(
        n_vocab=0,#词表大小
        n_ctx=1024,#最大上下文长度
        n_embd=768,#hidden size
        n_head=12,#头的数量
        n_layer=12,# Transformer 层的数量（Block数量）
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]
#手写softmax，减去最大值，所有的指数项都被强制压缩在了0到1之间，防止套上指数之后爆炸
def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)
#激活函数GELU
def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
# 层归一化（Layer Normalization）的实现
def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)# 计算均值
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)# 计算方差
        x = (x - u) * tf.rsqrt(s + epsilon)# 归一化 (减去均值，除以标准差)
        x = x*g + b# 应用可学习的缩放和平移
        return x
#多头注意力的分头，就是把一个长的向量切成短的向量去训练
def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])
#把头和并起来，#[batch, seq, 12, 64] -> [batch, seq, 768]
def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])
#全连接层
def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c
#生成一个下三角 mask，保证当前位置只能看自己和之前的位置，不能看未来。
def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0 #hidden size必须能被head数整除
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]
    #分头
    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])
    #合头
    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))
#掩码机制
    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)# 将注意力权重矩阵的右上角（未来信息）替换为极小值（-1e10），这样Softmax后这些位置的概率就变成了 0。
        return w
#多头注意力机制
    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        # 计算注意力分数: Q * K^T
        w = tf.matmul(q, k, transpose_b=True)
        # 缩放 (Scale)：除以根号下维度大小，防止点积结果过大导致梯度消失
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))
        #掩码，遮蔽未来信息
        w = mask_attn_weights(w)
        w = softmax(w)
        # 转化为概率分布
        a = tf.matmul(w, v)# 乘以 V (Value) 得到最终的注意力输出
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))#生成q，k，v矩阵，并劈成三份
        present = tf.stack([k, v], axis=1)
        if past is not None:#如果有历史 cache，就把历史 key/value 和当前 key/value 拼起来。
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present

# Transformer 层中的前馈神经网络。将维度放大（通常是4倍），经过 GELU，再压缩回原始维度。
def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2

# 这是一个完整的 Transformer Decoder 层 (Block)。
def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        # Pre-LayerNorm 架构（GPT独有，与原始论文不同）：先做 LayerNorm，再做 Attention
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        #前馈神经网络
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        #输出结果
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)

#完整模型入口
#X: token ids，形状 [batch, sequence]
#past: 历史 KV cache，可选
def model(hparams, X, past=None, scope='model', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)
        #词嵌入，wpe，position embedding
        #wte，token embedding
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
