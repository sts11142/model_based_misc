import os
import nltk
import json
import torch
import pickle
import logging
import numpy as np
from tqdm.auto import tqdm
from src.utils import config
import torch.utils.data as data
# from src.utils.common import save_config
from nltk.corpus import wordnet, stopwords
from src.utils.constants import DATA_FILES
from src.utils.constants import EMO_MAP as emo_map
from src.utils.constants import MAP_EMO as map_emo
from src.utils.constants import WORD_PAIRS as word_pairs
from src.utils.constants import STRATEGY_MAP as strategy_map
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
emotion_lexicon = json.load(open("data/NRCDict.json"))[0]
"""
emotion_lexicon の中身...配列．[0]が配列で[1]が辞書．lexicon: 辞書
[
    ["語1", "語2", ...],
    {
        "語1": 0 or 1 or 2,
        "語2": 0 or 1 or 2,
        ...
    }
]
"""
stop_words = stopwords.words("english")


class Lang:
    """
    quated:
        load_dataset(),
    usage:
        インスタンスに与えられた引数から，語彙とインデックスの辞書を作成
    """
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def get_wordnet_pos(tag):
    """
    quated:
        encode_ctx()
    usage:
        語彙の品詞を判定
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def process_sent(sentence):
    """
    quated:
        get_commonsense(), encode_ctx(), encode()
    usage:
        ワードペア(src.utils.constants WORD_PAIRS)で文を置換した後，tokenizeする．
    """
    sentence = sentence.lower() # str.lower(): 文字列を小文字に変換したものを返す
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)   # str.replace('a', 'b'): 文字列中の'a'を'b'に置換する
    sentence = nltk.word_tokenize(sentence)
    return sentence


def get_commonsense(comet, item, data_dict):
    """
    quated:
        encode_ctx()
    usage:
        Commetに文を投入し，各relationに対応する生成を得たのち，data_dictに追加する
        参考: data_dict ... in encode()
            data_dict = {
                "context": [],
                "target": [],
                "emotion": [],
                "situation": [],
                "emotion_context": [],
                "utt_cs": [],
            }
    """
    cs_list = []
    input_event = " ".join(item)
    for rel in relations:
        cs_res = comet.generate(input_event, rel)
        cs_res = [process_sent(item) for item in cs_res]
        cs_list.append(cs_res)

    data_dict["utt_cs"].append(cs_list)


def encode_ctx(vocab, items, data_dict, comet):
    """
    quated:
        encode()
    usage:
        データの中身をトークナイズして，各語の品詞を判定した後，辞書にdata_dictに加える
        参考: items はデータファイルの中の文章で，1文or複数文が配列になったタプル
            items = files[i]
            files = [
                (['hoge huga']),
                (['hoge huga', 'hoge huga', 'foo bar']),
                ...
            ]

    """
    for ctx in tqdm(items):
        ctx_list = []
        e_list = []
        for i, c in enumerate(ctx):
            item = process_sent(c)
            ctx_list.append(item)
            vocab.index_words(item)
            ws_pos = nltk.pos_tag(item)  # pos  ←Part of Speech: PoS
            """
            品詞タグを取得．配列の中のタプル
                ws_pos ... [('Hi', 'NNP'), (',', ','), ('I', 'PRP), ...]
            """
            for w in ws_pos:
                w_p = get_wordnet_pos(w[1])
                if w[0] not in stop_words and (
                    w_p == wordnet.ADJ or w[0] in emotion_lexicon
                ):
                    e_list.append(w[0])
            if i == len(ctx) - 1:
                get_commonsense(comet, item, data_dict)

        data_dict["context"].append(ctx_list)
        data_dict["emotion_context"].append(e_list)


def encode(vocab, files):
    """
    quated:
        read_file()
    usage:
        data_dict 辞書の中身を満たしていく
        data_dictの中身は，トークナイズされた要素
        files: ex) train_files = ["dialogue", "target", "emotion", "situation"]
    """
    from src.utils.comet import Comet

    data_dict = {
        "context": [],
        "target": [],
        "emotion": [],
        "situation": [],
        "strategy_label": [],
        "emotion_context": [],
        "utt_cs": [],
    }
    comet = Comet("data/Comet", config.device)

    for i, k in enumerate(data_dict.keys()):
        items = files[i]
        if k == "context": # items -> "sys_dialogue_texts.○○.np" のload後の中身
            encode_ctx(vocab, items, data_dict, comet) # 形態素をヨウ素にもつタプルで返す
        elif k == "emotion" or k == "strategy_label": # items -> "sys_emoition_texts.○○.np" のload後の中身
            data_dict[k] = items
        # elif k == "strategy_label":
        #     data_dict[k] = items
        else: # items -> "target", "situation"
            for item in tqdm(items): # "target", "situation"の各ファイルの1行
                item = process_sent(item) # 語のトークナイズ
                data_dict[k].append(item)
                vocab.index_words(item)
        if i == 4: # "situation"まで回ったら終了
            break
    assert (
        len(data_dict["context"])
        == len(data_dict["target"])
        == len(data_dict["emotion"])
        == len(data_dict["situation"])
        == len(data_dict["strategy_label"])
        == len(data_dict["emotion_context"])
        == len(data_dict["utt_cs"])
    )

    return data_dict

def _norm_text(text):
    """ from MISC """
    emo, r, t, *toks = text.strip().split()
    try:
        emo = int(emo)
        r = int(r)
        t = int(t)
        toks = ' '.join(toks[:len(toks)])
    except Exception as e:
        raise e
    return emo, r, t, toks

def _get_inputs_from_text(text):
    """ from MISC """
    srcs = text.strip()
    inputs = []
    emotion = None
    targets = []
    roles = []
    turns = []
    strategy_labels = []
    srcs = srcs.split(" EOS")
    srcs_len = len(srcs)
    """
    srcs:
        ex) ['3 0 0 Hi there, can you help me? ', " 3 1 1 [Question] I'll do my best. What do you need help with? ", ' 3 0 2 I feel depressed because I had to quit my job and stay home with my kids because of their remote school. ', ' 3 1 3 [Reflection of feelings] I can understand why that would make you feel depressed. ', ' 3 0 4 Do you have any advice on how to feel better? ', " 3 1 5 [Providing Suggestions] Yes of course. It's good that you are acknowledging your feelings. To improve your mood you could practice hobbies or other things you enjoy doing."]
    """

    for idx, src in enumerate(srcs):
        if src == "":
            continue
        src_emo, src_role, src_turn, src = _norm_text(src)
        if emotion is None:
            emotion = src_emo

        if src_role == 1:
            try:
                label = "[" + src.split("[")[1].split("]")[0] + "]"  # ex) [ "[Question]" ] → [ "[", "Question]" ] → [ "[", "Question", "]" ]
                src = src.split('[')[-1].split(']')[-1].strip()     # ラベルを剥がす
            except Exception as e:
                strategy_labels.append(8)
            else:
                strategy_labels.append(label)
        else:
            strategy_labels.append(8)
        
        inputs.append(src)
        roles.append(src_role)
        turns.append(src_turn)

        if idx == (srcs_len - 1):
            targets.append(inputs[-1])
            inputs = inputs[0:(srcs_len - 1)]
            target_label = strategy_labels[-1]

    return inputs, emotion, targets, roles, turns, target_label

def _make_emotion_fdata(emo_list):
    emo_fdata = []
    for (idx, emo) in enumerate(emo_list):
        emo_fdata.append(emo_map[emo])
    emo_fdata = np.array(emo_fdata, dtype='U12')

    return emo_fdata

def _make_target_fdata(lists):
    # [ ["hoge"], ["huga"], ... ] → [ "hoge", "huga", ... ]
    target_fdata = []
    for list in lists:
        target_fdata.append(list[0])
    target_fdata = np.array(target_fdata, dtype=str)

    return target_fdata

def construct_conv_ESD(arr, file_type=None):
    contexts_fdata = []
    target_data = []
    emotion_data = []
    situation_data = []
    others_data = {
        "roles": [],
        "turns": [],
        "strategy_labels": []
    }

    with open("data/ESConv" + "/" + file_type + "Situation.txt", "r", encoding="utf-8") as f:
        situation = f.read().split("\n")

    # for row in arr:
    for (row, situ) in zip(arr, situation):
        inputs, emotion, targets, roles, turns, strategy_label = _get_inputs_from_text(row) 
        contexts_fdata.append(inputs)
        target_data.append(targets)
        # emotion_data.append(emotion)
        emotion_data.append(map_emo[emotion])
        situation_data.append(situ)
        others_data["roles"] = roles
        others_data["turns"] = turns
        others_data["strategy_labels"].append(strategy_label)
        # others_data["strategy_labels_id"].append(strategy_map[strategy_label])

    
    contexts_fdata = np.array(contexts_fdata, dtype=object)
    target_fdata = _make_target_fdata(target_data)
    # emotion_fdata = _make_emotion_fdata(emotion_data)
    emotion_fdata = np.array(emotion_data, dtype=str)
    situation_fdata = np.array(situation_data, dtype=str)

    strategy_labels = others_data["strategy_labels"]
    strategy_fdata = np.array(strategy_labels)

    return contexts_fdata, target_fdata, emotion_fdata, situation_fdata, strategy_fdata

def setup_fdata(file_type):
    # with open(config.data_dir + "/" + file_type + "WithStrategy_short.tsv", "r", encoding="utf-8") as f:
    with open("data/ESConv" + "/" + file_type + "WithStrategy_short.tsv", "r", encoding="utf-8") as f:
        df_trn = f.read().split("\n")
    # contexts, targets, emotions, situations, _ = construct_conv_ESD(df_trn[:-1], file_type=file_type)
    contexts, targets, emotions, situations, strategy_labels = construct_conv_ESD(df_trn[:-1], file_type=file_type)

    print(strategy_labels)

    fdata = []
    fdata.append(contexts)
    fdata.append(targets)
    fdata.append(emotions)
    fdata.append(situations)
    fdata.append(strategy_labels)

    fdata = np.array(fdata, dtype=object)

    return fdata


def _debug_ESC():
    train_files = setup_fdata('train')
    # dev_files = setup_fdata('dev')
    # test_files = setup_fdata('test')

    print(f"ES: {train_files}\n")

    return


def read_files(vocab):
    """
    quated:
        load_dataset()
    usage:
        ファイルの中身を元にencode()で作成した辞書を返す．train, dev, testの3つ分．
    """
    #### train_files = ["dialogue", "target", "emotion", "situation"]
    # files = DATA_FILES(config.data_dir)
    # train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
    # dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
    # test_files = [np.load(f, allow_pickle=True) for f in files["test"]]

    # 書き換え
    # files = DATA_FILES("data/ESConv")

    # 各ファイルの読み込み
    train_files = setup_fdata('train')
    dev_files = setup_fdata('dev')
    test_files = setup_fdata('test')

    # ファイルを読み込んでencodeする
    # encode: emotionとかcontextとか毎にデータを分けた後，各々をトークナイズ
    #         返り値は辞書
    data_train = encode(vocab, train_files)
    data_dev = encode(vocab, dev_files)
    data_test = encode(vocab, test_files)

    return data_train, data_dev, data_test, vocab


def load_dataset():
    """
    quated:
        prepare_data_seq()
    usage:
        キャッシュファイルがあればそれを読み込む．なければ新規で読み込んでキャッシュを作成する．
        読み込んだファイルの中身を返す(4つの値)
    """
    data_dir = config.data_dir
    # data_dir = "data/ESConv"

    cache_file = f"{data_dir}/dataset_preproc.p"
    # """
    if os.path.exists(cache_file):
        # print("LOADING empathetic_dialogue")
        print("LOADING Emotional Support Conversation")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        # data_tra他: 辞書
        data_tra, data_val, data_tst, vocab = read_files(
            vocab=Lang(
                {
                    config.UNK_idx: "UNK",
                    config.PAD_idx: "PAD",
                    config.EOS_idx: "EOS",
                    config.SOS_idx: "SOS",
                    config.USR_idx: "USR",
                    config.SYS_idx: "SYS",
                    config.CLS_idx: "CLS",
                }
            )
        )
        with open(cache_file, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")
    # """

        # print("Building dataset...")
        # # data_tra他: 辞書
        # data_tra, data_val, data_tst, vocab = read_files(
        #     vocab=Lang(
        #         {
        #             config.UNK_idx: "UNK",
        #             config.PAD_idx: "PAD",
        #             config.EOS_idx: "EOS",
        #             config.SOS_idx: "SOS",
        #             config.USR_idx: "USR",
        #             config.SYS_idx: "SYS",
        #             config.CLS_idx: "CLS",
        #         }
        #     )
        # )
        # with open(cache_file, "wb") as f:
        #     pickle.dump([data_tra, data_val, data_tst, vocab], f)
        #     print("Saved PICKLE")

    for i in range(3):
        print("[situation]:", " ".join(data_tra["situation"][i]))
        print("[emotion]:", data_tra["emotion"][i])
        print("[context]:", [" ".join(u) for u in data_tra["context"][i]])
        print("[target]:", " ".join(data_tra["target"][i]))
        print("[strategy_label]:", " ".join(data_tra["strategy_label"][i]))
        print(" ")
    
    # 返却する値は，各々辞書形式で，各キーに対してcontextとかemotionのデータが格納されている
    return data_tra, data_val, data_tst, vocab

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    """
    Pytorchではデータの組([入力データ, ラベル])を返すDatasetを
    torch.utils.data.DataLoader()に渡す必要がある
    そのためのもの．
    自前のデータを使用する場合，__len__()と__getitem__()を定義する必要がある．
    """

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        # data には辞書(data_dict)が入る
        self.vocab = vocab
        self.data = data
        self.emo_map = emo_map
        self.strategy_map = strategy_map
        self.analyzer = SentimentIntensityAnalyzer()

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["situation_text"] = self.data["situation"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["emotion_context"] = self.data["emotion_context"][index]

        item["context_emotion_scores"] = self.analyzer.polarity_scores(
            " ".join(self.data["context"][index][0])
        )

        item["context"], item["context_mask"] = self.preprocess(item["context_text"])
        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(
            item["emotion_text"], self.emo_map
        )
        (
            item["emotion_context"],
            item["emotion_context_mask"],
        ) = self.preprocess(item["emotion_context"])

        # strategy
        item["strategy"], item["strategy_label"] = self.preprocess_strategy(
            self.data["strategy_label"][index], self.strategy_map
        )

        item["cs_text"] = self.data["utt_cs"][index]
        item["x_intent_txt"] = item["cs_text"][0]
        item["x_need_txt"] = item["cs_text"][1]
        item["x_want_txt"] = item["cs_text"][2]
        item["x_effect_txt"] = item["cs_text"][3]
        item["x_react_txt"] = item["cs_text"][4]

        item["x_intent"] = self.preprocess(item["x_intent_txt"], cs=True)
        item["x_need"] = self.preprocess(item["x_need_txt"], cs=True)
        item["x_want"] = self.preprocess(item["x_want_txt"], cs=True)
        item["x_effect"] = self.preprocess(item["x_effect_txt"], cs=True)
        item["x_react"] = self.preprocess(item["x_react_txt"], cs="react")

        return item

    def preprocess(self, arr, anw=False, cs=None, emo=False):
        """Converts words to ids."""
        if anw:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in arr
            ] + [config.EOS_idx]

            return torch.LongTensor(sequence)
        elif cs:
            sequence = [config.CLS_idx] if cs != "react" else []
            for sent in arr:
                sequence += [
                    self.vocab.word2index[word]
                    for word in sent
                    if word in self.vocab.word2index and word not in ["to", "none"]
                ]

            return torch.LongTensor(sequence)
        elif emo:
            x_emo = [config.CLS_idx]
            x_emo_mask = [config.CLS_idx]
            for i, ew in enumerate(arr):
                x_emo += [
                    self.vocab.word2index[ew]
                    if ew in self.vocab.word2index
                    else config.UNK_idx
                ]
                x_emo_mask += [self.vocab.word2index["CLS"]]

            assert len(x_emo) == len(x_emo_mask)
            return torch.LongTensor(x_emo), torch.LongTensor(x_emo_mask)

        else:
            x_dial = [config.CLS_idx]
            x_mask = [config.CLS_idx]
            for i, sentence in enumerate(arr):
                x_dial += [
                    self.vocab.word2index[word]
                    if word in self.vocab.word2index
                    else config.UNK_idx
                    for word in sentence
                ]
                spk = (
                    self.vocab.word2index["USR"]
                    if i % 2 == 0
                    else self.vocab.word2index["SYS"]
                )
                x_mask += [spk for _ in range(len(sentence))]
            assert len(x_dial) == len(x_mask)

            return torch.LongTensor(x_dial), torch.LongTensor(x_mask)

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]

    def preprocess_strategy(self, strategy_label, strategy_map):
        program = [0] * len(strategy_map)
        program[strategy_map[strategy_label]] = 1
        return program, strategy_map[strategy_label]


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).long()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths = merge(item_info["context"])
    mask_input, mask_input_lengths = merge(item_info["context_mask"])
    emotion_batch, emotion_lengths = merge(item_info["emotion_context"])

    ## Target
    target_batch, target_lengths = merge(item_info["target"])

    input_batch = input_batch.to(config.device)
    mask_input = mask_input.to(config.device)
    target_batch = target_batch.to(config.device)

    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["mask_input"] = mask_input
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    d["emotion_context_batch"] = emotion_batch.to(config.device)

    ##program
    d["target_program"] = item_info["emotion"]
    d["program_label"] = item_info["emotion_label"]

    ##strategy
    d["strategy"] = item_info["strategy"]
    d["strategy_label"] = item_info["strategy_label"]

    ##text
    d["input_txt"] = item_info["context_text"]
    d["target_txt"] = item_info["target_text"]
    d["program_txt"] = item_info["emotion_text"]
    d["situation_txt"] = item_info["situation_text"]

    d["context_emotion_scores"] = item_info["context_emotion_scores"]

    relations = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]
    for r in relations:
        pad_batch, _ = merge(item_info[r])
        pad_batch = pad_batch.to(config.device)
        d[r] = pad_batch
        d[f"{r}_txt"] = item_info[f"{r}_txt"]

    return d


def prepare_data_seq(batch_size=32):
# def prepare_data_seq(batch_size=11): # ESConv ver.
    """
    quated:
        main.py
    usage:
        バッチサイズに固めた3つのデータ(tra, valid, test)とvocabとlen
        の5つを返す
    """

    # それぞれが辞書
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    # torch.utils.data.DataLoader(datase=, ...)
    #    PytorchのDataLoaderにDatasetインスタンス([入力, ラベル]の組を1つ返すもの)
    #    を渡して，データをバッチサイズに固めて戻す

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    save_config()
    return (
        data_loader_tra,
        data_loader_val,
        data_loader_tst,
        vocab,
        len(dataset_train.emo_map),
    )


if __name__ == "__main__":
    _debug_ESC()