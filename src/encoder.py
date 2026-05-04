"""Byte pair encoding utilities"""
#分词器，BPE
import os
import json
import regex as re
from functools import lru_cache
#总结就是按照预先设定好的bpe规则，去分词，分完词之后去设定对应token_id
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
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))#吧安全的byte编码先放进来，UTF-8 编码后的字节值，每个值范围是 0~255

    cs = bs[:]
    n = 0
    for b in range(2**8):
        #如果某个byte不在前面选好的安全字符集合里，就给他分配一个新的unicode字符，就是把危险的先用别的替代（危险的指的是换行，空格这类影响正则表达的)
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
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
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):#这个BPE的作用是吧全部的字母合到合适的程度，不一定是整个单词，也不一定是一个一个字母，现在很好的开源
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))#在所有的pair里，找BPE优先级最好的pair
            if bigram not in self.bpe_ranks:#如果当前所有的pair都不在merge列表中，就停止合并，这个bpe_ranks是预先处理好的存放在vocab.bpe中了
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
            # 步骤 1：正则预切分

        for token in re.findall(self.pat, text):
            # 步骤 2：UTF-8 字节映射
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
        # 步骤 3 & 4：BPE 合并 与 ID 查表
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        # 步骤 1：ID 转字符串
        text = ''.join([self.decoder[token] for token in tokens])
        # 步骤 2 & 3：逆向字节映射 与 UTF-8 解码
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
