"""Byte pair encoding utilities"""
#分词器，BPE
import os
import json
import regex as re
from functools import lru_cache
@lru_cache()
#把输入的字节转成对应的unicode字符
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    #一个byte有8bit，共有256种结果，然后一个文字可能对应多个byte的组合，而每个byte对应的结果是固定的，类似acssic编码
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))#把安全的byte编码先放进来，UTF-8 编码后的字节值，每个值范围是 0~255

    cs = bs[:]
    n = 0
    for b in range(2**8):
        #如果某个byte不在前面选好的安全字符集合里，就给他分配一个新的unicode字符，就是把危险的先用别的替代（危险的指的是换行，空格这类影响正则表达的)
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]#把数字code point 转成真正的Unicode字符
    return dict(zip(bs, cs))#最后出来的格式是，一个byte值，对应一个unicode

def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()#用集合存pairs，去重
    prev_char = word[0]#把一个词中所有的相邻字母存下来
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))#给BPE Merge规则编号，越靠前的merge,优先级越高，rank越小，就是把vocab_merger里的东西变成类似数组键值的东西，方便比较大小
        self.cache = {}#已经encoder的在cache中找，不重复

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")#文本切分正则。它会把原始文本先切成一段一段的 token-like 片段。



    def bpe(self, token):
        # 缓存机制：如果这个单词以前处理过，直接从字典里拿结果，提高速度
        if token in self.cache:
            return self.cache[token]
        
        # 初始化：把输入的单词拆成单个字符的元组。比如 'low' 变成 ('l', 'o', 'w')
        word = tuple(token)
        # 获取当前字符序列中所有的相邻字符对。比如 ('l', 'o') 和 ('o', 'w')
        pairs = get_pairs(word)

        if not pairs:
            return token

        # 不断寻找最优组合并合并，直到无法合并为止
        while True:
            # 在当前所有的字符对(pairs)中，去预先训练好的 bpe_ranks 字典里查找。
            # 找到 rank 值最小的（即优先级最高、在训练语料中最常出现的组合）。
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            # 如果当前找到的最优组合不在 bpe_ranks 里（说明所有的配对都不合法了），则停止合并
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            
            # 合并：遍历当前的 word 元组，把所有的 first 和 second 合并成 first+second
            while i < len(word):
                try:
                    # 直接寻找下一个 first 出现的位置，跳过无关字符
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    # 如果找不到了，说明剩下的字符里没有 first 了，把剩下的字符全部加进来，结束这轮遍历
                    new_word.extend(word[i:])
                    break

                # 确认找到的 first 后面紧跟着的是不是 second
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    # 如果是，就把它们拼起来当成一个新词根加进去
                    new_word.append(first+second)
                    i += 2 # 因为合并了两个字符，所以索引往前跳 2 步
                else:
                    # 如果不是（比如只有 first，后面跟着的不是 second），就原样加进去
                    new_word.append(word[i])
                    i += 1
                    
            # 更新 word 为合并后的新元组
            new_word = tuple(new_word)
            word = new_word
            
            # 如果整个单词已经合并成了一个完整的词，就不需要再合了
            if len(word) == 1:
                break
            else:
                # 重新计算合并后的字符对，进入下一轮循环
                pairs = get_pairs(word)
                
        # 格式化输出与缓存：把元组用空格连起来，比如 ('low', 'est') 变成 'low est'
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
            # 正则预切分

        for token in re.findall(self.pat, text):
            # UTF-8 字节映射
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
        # BPE 合并 与 ID 查表
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        # ID 转字符串
        text = ''.join([self.decoder[token] for token in tokens])
        #逆向字节映射 与 UTF-8 解码
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

def get_encoder(model_name, models_dir):
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )
