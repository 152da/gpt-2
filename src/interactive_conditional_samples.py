#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='124M',
    seed=None,
    nsamples=1,#针对你输入的同一段提示词（Prompt），你希望模型总共为你生成多少个不同的结果。
    batch_size=1,
    length=None,#指的是模型生成的token的最大长度，与我输入的token的数量加起来不能超过n_ctx
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',#模型所在目录文件夹
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))#处理模型目录路径，先解析环境变量，再解析用户主目录波浪号。
    if batch_size is None:#保证batch_size不少于1
        batch_size = 1
    assert nsamples % batch_size == 0#保证每批次生成样本同样多

    enc = encoder.get_encoder(model_name, models_dir)#加载tokenizer，负责将文本转成token->encoder,以及decoder
    hparams = model.default_hparams()#加载模型默认超参数
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:#打开models_dir/model_name/hparams.josn的目标路径
        hparams.override_from_dict(json.load(f))#json.load是把json文件转化为python的字典值
#hparams.override_from_dict是先给一组默认值，然后用实际模型目录里的配置覆盖它，也就是json文件里面是模型真正的参数
    if length is None:
        length = hparams.n_ctx // 2#如果不指定，就按最大上下文的一半来计
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    #生成一块计算图
    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])#占位符
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        #这一段的作用是把训练好的模型参数加载进来
        saver = tf.train.Saver()#实例化一个参数搬运
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))#找到最新的权重文件路径
        saver.restore(sess, ckpt)#把权重放到刚刚建立的网络中，现在的程序运行都是基于sess的


        while True:#循环生成
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={#真正去运行程序
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])#toker->文本
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)#打印文本
            print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)

