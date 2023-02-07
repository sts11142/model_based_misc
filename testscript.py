import os
import pickle
from tqdm import tqdm, trange
import torch

from src.transformers import (BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration, BlenderbotSmallConfig)

class InputFeatures_train(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids,
                role_ids, lm_labels, cls_position, cls_label, strategy_ids, input_len=None):
        self.conv_id = conv_id
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.role_ids = role_ids
        self.lm_labels = lm_labels
        self.cls_position = cls_position
        self.cls_label = cls_label
        self.strategy_ids = strategy_ids
        if input_len is None:
            self.input_len = len(input_ids)
        else:
            self.input_len = input_len


class InputFeatures_blender(object):
    def __init__(self, encoder_feature, decoder_feature, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask):
        self.conv_id = encoder_feature.conv_id
        self.input_ids = encoder_feature.input_ids
        self.position_ids = encoder_feature.position_ids
        self.token_type_ids = encoder_feature.token_type_ids
        self.role_ids = encoder_feature.role_ids
        self.lm_labels = encoder_feature.lm_labels
        self.cls_position = encoder_feature.cls_position
        self.cls_label = encoder_feature.cls_label
        self.strategy_ids = encoder_feature.strategy_ids
        self.decoder_input_ids = decoder_feature.input_ids
        self.decoder_position_ids = decoder_feature.position_ids
        self.decoder_token_type_ids = decoder_feature.token_type_ids
        self.decoder_role_ids = decoder_feature.role_ids
        self.decoder_lm_labels = decoder_feature.lm_labels
        self.decoder_cls_position = decoder_feature.cls_position
        self.decoder_cls_label = decoder_feature.cls_label
        self.decoder_strategy_ids = decoder_feature.strategy_ids
        self.comet_ids = comet_ids
        self.comet_mask = comet_mask
        self.emotion = emotion
        self.comet_st_ids = comet_st_ids
        self.comet_st_mask = comet_st_mask


def process_row_to_comet_query(row):
    sents = row.strip().split('EOS')
    n_sent = len(sents)
    all_seeker_uttrs = []
    for i in range(n_sent-1, -1, -1):
        # print(sents[i].strip().split(' '))
        tokens = sents[i].strip().split(' ')
        if int(tokens[1]) == 0:
            if int(tokens[1]) == 0:
                return ' '.join(tokens[3:])
                # all_seeker_uttrs.append(' '.join(tokens[3:]))
    # return '\t'.join(all_seeker_uttrs)


def summary(test_file_path, generate_file_path, reference_file_path, summary_file_path, all_top_k_blocks, all_top_k_blocks_st, chat_texts, test_situation_file_path):
    with open(test_file_path, "r", encoding="utf-8") as f:
        ctx = f.read().split("\n")
    with open(test_situation_file_path, "r", encoding="utf-8") as f:
        st = f.read().split("\n")
    ctx = ctx[:-1]
    st = st[:-1]
    with open(generate_file_path, "r", encoding="utf-8") as f:
        gen_rep = json.load(f)
    with open(reference_file_path, "r", encoding="utf-8") as f:
        ref_rep = json.load(f)
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        for (ctx_row, ref_rep_row, gen_rep_row, top_k_blocks, top_k_blocks_st, chat_text, st_row) in zip(ctx, ref_rep, gen_rep, all_top_k_blocks, all_top_k_blocks_st, chat_texts, st):
            query = process_row_to_comet_query(chat_text)
            if query is None:
                query = ""
            line = '[contxt]\t' + ctx_row + '\n[reference_response]\t' + ref_rep_row + '\n[hypothesis_response]\t' + gen_rep_row + '\n[comet query]\t' + query + '\n[comet blocks (attention top5)]\t' + '  '.join(top_k_blocks) +'\n[situation]\t' + st_row + '\n[situation comet blocks (attention top5)]\t' + '  '.join(top_k_blocks_st) + '\n' * 2
            f.writelines(line)

def extract_top_k_attention_comet_block(mutual_attentions, comet_rows, k):
    all_top_k_blocks = []
    num_block = len(mutual_attentions[0])
    for mutual_attention, comet_row in zip(mutual_attentions, comet_rows):
        comet_blocks = comet_row.split('__EOS__')[:-1]
        if len(comet_blocks) < num_block:
            comet_blocks += (['[PAD]'] * (num_block - len(comet_blocks)))
        index = torch.topk(mutual_attention, k).indices
        top_k_blocks = [comet_blocks[i] for i in index.numpy().tolist()]
        all_top_k_blocks.append(top_k_blocks)
    return all_top_k_blocks

def _get_comet_input(comet_row, tokenizer, max_num_attr=30, max_len_attr=10):
    attrs = comet_row.split('__EOS__')[:-1]
    comet_ids = []
    comet_mask = [] #对每一个comet attr + tail 的mask
    for ids, attr in enumerate(attrs):
        if ids == max_num_attr:
            break
        comet_attr_ids = tokenizer.encode(attr)
        if len(comet_attr_ids) < max_len_attr:
            comet_attr_ids += [tokenizer.pad_token_id]*(max_len_attr - len(comet_attr_ids)) 
        else:
            comet_attr_ids = comet_attr_ids[:max_len_attr]
        comet_ids.append(comet_attr_ids)
        comet_mask.append(1)

    if len(comet_ids) < max_num_attr:
        comet_ids += ([[tokenizer.pad_token_id]*max_len_attr]) * (max_num_attr - len(comet_ids))
        comet_mask += [0] * (max_num_attr - len(comet_mask))
    # print(attrs) 
    # print(comet_ids)
    # print(comet_mask)
    # print(error)
    
    assert len(comet_ids) == max_num_attr
    assert len(comet_mask) == max_num_attr
    return comet_ids, comet_mask


def _make_feature(id_, sents, rls, ts, eos, pad=False, block_size=512, strategy_labels=None, evaluate=False, str_embd=False, generation=False):
    # we did't use role label and turn number in modeling as they did't carry significant improvement. However, codes still remain here.
    # sents: inputs("EOS"で区切った複数の入力文)
    if len(sents) == 0:
        return InputFeatures_train([], [], [], [], [],
                            [], [] , [], [])
    input_ids = [i for s in sents for i in s+[eos]]
    # s+[eos]: [ある行のあるターンにおける1発話(tokenized)をencodeしたもの + eos_token_id]
    # input_ids = [ 201, 204, ..., 401, eos_token_id, ..., ... ]

    input_ids = input_ids
    lm_labels = []
    token_type_ids = []
    roles = []
    strategy_ids = []

    for i, s in enumerate(sents):
        token_type_ids += [ts[i]] * (len(s) + 1)
        flag_str = -1
        if str_embd: #use for strategy embed but currently we treat strategy as token
            strategy_ids += [strategy_labels[-1]] * (len(s) + 1)
        else:
            strategy_ids += [8] * (len(s) + 1)
        if i < len(sents) - 1:
            lm_labels += [-100] * (len(s) + 1)
            roles += [rls[i]] * (len(s) + 1)
        else:
            lm_labels += (  s + [eos])
            roles += [rls[i]] * (len(s) + 1)

    i = len(lm_labels) - 1
    if len(input_ids) == 1:
        print(input_ids, lm_labels, token_type_ids, roles)
    while i >= 0:
        if lm_labels[i] != -100:
            break
        i -= 1
    input_ids = input_ids[:i+1]
    lm_labels = lm_labels[:i+1]
    token_type_ids = token_type_ids[:i+1]
    roles = roles[:i+1]
    if not str_embd:
        strategy_ids = [8]*len(input_ids) # strategy is not used
    else:
        strategy_ids = strategy_ids[:i+1]
    if len(input_ids) == 1:
        print(input_ids, lm_labels, token_type_ids, roles)


    assert (len(input_ids) == len(token_type_ids)
            == len(lm_labels) == len(roles) == len(strategy_ids))
    # cut according to block size
    if len(input_ids) > block_size:
        cut_index = input_ids.index(eos,-512) + 1
        input_ids = input_ids[cut_index: ]

        token_type_ids = token_type_ids[cut_index: ]
        lm_labels = lm_labels[cut_index: ]
        roles = roles[cut_index: ]
        strategy_ids = strategy_ids[cut_index: ]
    # pad to multiples of 8
    if pad:
        while len(input_ids) % 8 != 0:
            input_ids.append(0)
            token_type_ids.append(0)
            lm_labels.append(-100)
            roles.append(0)
            strategy_ids.append(8)
        assert len(input_ids) % 8 == 0
    position_ids = list(range(len(input_ids)))
    assert (len(input_ids) == len(position_ids) == len(token_type_ids)
            == len(lm_labels) == len(roles) == len(strategy_ids))
    if len(input_ids) == 0:
        import pdb
        pdb.set_trace()
    elif len(input_ids) == 1:
        print(input_ids, lm_labels, token_type_ids, roles)
    if True:
        # if it is for generation, the last sentence of context is the last sentence
        cls_position = len(input_ids)-1-input_ids[::-1].index(eos)
    else:
        # if not, the last sentence of context is the second last sentence
        cls_position = len(input_ids)-1-input_ids[::-1].index(eos,input_ids[::-1].index(eos)+1)
    if evaluate and strategy_labels[-1]!=8:
        try:
            lm_labels[lm_labels.index(strategy_labels[-1]+50257+4687)] = -100
        except Exception:
            pass

    feature = InputFeatures_train(id_, input_ids, position_ids, token_type_ids, roles,
                            lm_labels, cls_position , strategy_labels[-1], strategy_ids)
    return feature

def _norm_text(text):
    """
    strip()は，文字列の頭と最後の空白文字を削除する．文字間の空白は削除しない．
    その後splitで分ける．
    * はjsのスプレッド構文みたいなもの．unpacking
    例えばtrainWithStrategy.tsv の1行目
        3 0 0 Hi there, can you ...
            emo(tion): 3
            r(ole): 0,
            t(urn): 0,
            *toks: ["Hi", "there", ",", "can", "you", ...]
    """
    emo, r, t, *toks = text.strip().split()
    try:
        emo = int(emo)
        r = int(r)
        t = int(t)
        toks = ' '.join(toks[:len(toks)])   # 文末に空白を入れる？
    except Exception as e:
        raise e
    return emo, r, t, toks

def _get_inputs_from_text(text, tokenizer, strategy=True, cls = False):
    srcs = text.strip()
    inputs = []
    roles = []
    turns = []
    strategy_labels=[]
    srcs = srcs.split(" EOS")   # "EOS"で区切る
    emotion = None
    for idx, src in enumerate(srcs):

        if src =="":
            continue
        src_emo, src_role, src_turn, src = _norm_text(src)
        if emotion is None:
            emotion = src_emo

        context_id = tokenizer.encode(src)  # src: ex) [" ", "Hi", "there", ",", "can", ...]

        # ここは基本的にはpassになりそう
        if not strategy:    # strategy は基本有りな感じ
            context_id = [i  for i in context_id if i< 50257+4687]
        elif cls:   # clsは基本無い感じ
            context_id = tokenizer.cls + [i for i in context_id if i< 50257+4687]
            # [2, 2342, 2193, ...]   ← トークンごとにidを割り振り．先頭の2はclsトークンのつもり
        else:
            pass

        # 会話は src_role==1 の発話で終了する．つまりsrc_role==1 → 相手発話？
        #   src_role==1: 相手．strategy_labels は label のencode値
        #   src_role==0: 自分．strategy_labels は others
        # ちなみにtry-except-elseは，tryで例外処理が発生したらexcept，例外処理が発生しなかったらelseに飛ぶ
        if src_role==1:
            try:
                label = "["+src.split("[")[1].split("]")[0]+"]"     # [Question] I'll do my best. What ... のラベルを取り出し
            except Exception as e:
                strategy_labels.append(8)
            else:
                strategy_labels.append(tokenizer.encode([label])[0] - 50257-4687)
        else:
            strategy_labels.append(8)

        inputs.append(context_id)
        roles.append(src_role)
        turns.append(src_turn)

    """
    0: 自分, 1: 相手  ?
    inputs: encoded,
    roles: 0 or 1,
    turns: 0 or 1,
    strategy_labels: encoded,
    emotion: 0 ~ 7
    """
    return inputs, roles, turns, strategy_labels, emotion

def construct_conv_ESD(idx, row, comet_row, comet_st_row, tokenizer, eos = True, pad=True, cls=False, evaluate=False, strategy=True, generation=False):

    #  process input text
    inputs, roles, turns, strategy_labels, _ = _get_inputs_from_text("EOS".join(row.split("EOS")[:-1]), tokenizer, strategy=strategy)
    # process output (decoder input) text
    d_inputs, d_roles, d_turns, d_strategy_labels, emotion = _get_inputs_from_text(row.split("EOS")[-1], tokenizer, strategy=strategy)

    # make feature for input text
    feature = _make_feature(idx, inputs, roles, turns, tokenizer.eos_token_id, pad=pad, strategy_labels=strategy_labels, evaluate=evaluate, str_embd=True, generation=generation)
    # make feature for output (decoder input) text
    d_feature = _make_feature(idx, d_inputs, d_roles, d_turns, tokenizer.eos_token_id, pad=pad, strategy_labels=d_strategy_labels, evaluate=evaluate, str_embd=True, generation=generation)
    comet_ids, comet_mask = _get_comet_input(comet_row, tokenizer)
    comet_st_ids, comet_st_mask = _get_comet_input(comet_st_row, tokenizer, max_num_attr=20)
    feature = InputFeatures_blender(feature, d_feature, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask)
    # InputFeature_blender() は，引数の情報をまとめて保持するためのオブジェクト
    # feature は↑のオブジェクトそのもの．生成した多数の情報を持っている
    return feature








def main():
    cached_features_file = os.path.join(
        "./cached", "trn_mymodel_cached_lm_512")
    with open(cached_features_file, "rb") as handle:
        features = pickle.load(handle)
        position_ids = [f.position_ids for f in features]

    with open("./dataset"+"/"+"testWithStrategy_short.tsv","r") as f:
        chat_texts = f.read().split("\n")
    with open("./dataset"+"/"+ "testComet_st.txt", "r", encoding="utf-8") as f:
        comet_st = f.read().split("\n")

    with open("./dataset"+"/"+ "testcomet.txt", "r", encoding="utf-8") as f:
        comet = f.read().split("\n")

    with open("./dataset/"+"dataset_preproc.p", "rb") as f:
        [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    
    keys = [
        "cs_text",
        "x_intent_txt",
        "x_need_txt",
        "x_want_txt",
        "x_effect_txt",
        "x_react_txt",
        "x_intent",
        "x_need",
        "x_want",
        "x_effect",
        "x_react",
        ]
    
    def merge(sequences):
        lengths = [len(sequences)]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).long()  ## padding index 1
        end = lengths
        padded_seqs[:, :end] = sequences[:end]
        return padded_seqs, lengths

    my_dicts = []
    
    for cs_row in data_tst["utt_cs"]:
        my_dict = {}
        my_dict["cs_text"] = cs_row

        my_dict["x_intent_txt"] = my_dict["cs_text"][0]
        my_dict["x_need_txt"] = my_dict["cs_text"][1]
        my_dict["x_want_txt"] = my_dict["cs_text"][2]
        my_dict["x_effect_txt"] = my_dict["cs_text"][3]
        my_dict["x_react_txt"] = my_dict["cs_text"][4]

        my_dict["x_intent"] = preprocess(my_dict["x_intent_txt"], vocab, cs=True)
        my_dict["x_need"] = preprocess(my_dict["x_need_txt"], vocab, cs=True)
        my_dict["x_want"] = preprocess(my_dict["x_want_txt"], vocab, cs=True)
        my_dict["x_effect"] = preprocess(my_dict["x_effect_txt"], vocab, cs=True)
        my_dict["x_react"] = preprocess(my_dict["x_react_txt"], vocab, cs="react")

        my_dicts.append(my_dict)

    additional_special_tokens = ["[Question]", "[Reflection of feelings]", "[Information]",
                                 "[Restatement or Paraphrasing]", "[Others]", "[Self-disclosure]",
                                 "[Affirmation and Reassurance]", "[Providing Suggestions]"]
    # comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]"]
    comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]",
                                       "[oEffect]", "[oReact]"]

    
    tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M", cache_dir='./blender-small')
    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.add_tokens(comet_additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})

    assert len(comet) == len(chat_texts) == len(comet_st)
    gts = []
    refs = []
    mutual_attentions = []
    mutual_attentions_st = []
    strategy_logit_str = []
    # Let's chat for 5 lines
    strategy_hits = []
    strategy_record = []
    strategy_hits_topk = [[] for _ in range(8)]
    for idx, (c_text, comet_row, comet_st_row, my_di) in tqdm(enumerate(zip(chat_texts[:-1], comet[:-1], comet_st[:-1], my_dicts[:-1])), desc="Testing"):
        if "EOS" not in c_text:     # "EOS"が含まれていなければスキップ
            continue
        # if idx>=100:
        #     break
        # tokens = c_text.split("EOS")[-1].strip().split(" ")[3:]
        # print(tokens)
        # gts.append(" ".join(tokens[1:]))
        # = max(tokenizer.encode(tokens[0]))
        chat_history = c_text
        # f: feature. 
        f = construct_conv_ESD(idx, chat_history, comet_row, comet_st_row, tokenizer, eos = True, pad=False, cls=False, strategy=False, generation=True)
        if len(f.input_ids) >= 512:
            f.input_ids = f.input_ids[-512:]
            f.input_ids[0] = tokenizer.encode(tokenizer.cls_token)[0]
        else:
            f.input_ids = tokenizer.encode(tokenizer.cls_token) + f.input_ids
        next_strategy_id = f.decoder_strategy_ids[0]
        decoder_strategy_ids = torch.tensor([f.decoder_strategy_ids], dtype=torch.long)
        decoder_strategy_ids = decoder_strategy_ids[:, 0]
        # print(decoder_strategy_ids)
        # print(1/0)

        d = {}
        relations = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]
        for r in relations:
            # pad_batch, _ = merge(my_di[r])
            # d[r] = pad_batch
            d[r] = my_di[r]
            d[f"{r}_txt"] = my_di[f"{r}_txt"]

        gts.append(tokenizer.decode(f.decoder_input_ids, skip_special_tokens=True))

        emotion = torch.tensor([f.emotion], dtype=torch.long)
        comet_ids = torch.tensor([f.comet_ids], dtype=torch.long)
        comet_mask = torch.tensor([f.comet_mask], dtype=torch.long)
        comet_ids_st = torch.tensor([f.comet_st_ids], dtype=torch.long)
        comet_mask_st = torch.tensor([f.comet_st_mask], dtype=torch.long)

        batch_size, n_attr, len_attr = comet_ids.shape
        comet_ids = comet_ids.view(-1, len_attr)
        v = comet_ids.ne(tokenizer.pad_token_id)
        # v: [ [True, True, False, False, False, ...], [...], ... ]
        # comet_embs = model.model.encoder(comet_ids, attention_mask=comet_ids.ne(tokenizer.pad_token_id))[0][:, 0, :]
        # comet_embs = comet_embs.view(batch_size, n_attr, -1)

        batch_size, n_attr, len_attr = comet_ids_st.shape
        comet_ids_st = comet_ids_st.view(-1, len_attr)
        # comet_embs_st = model.model.encoder(comet_ids_st, attention_mask=comet_ids_st.ne(tokenizer.pad_token_id))[0][:, 0, :]
        # comet_embs_st = comet_embs_st.view(batch_size, n_attr, -1)

def preprocess(arr, vocab, anw=False, cs=None, emo=False):
    """Converts words to ids."""
    if anw:
        sequence = [
            vocab.word2index[word]
            if word in vocab.word2index
            else 0
            for word in arr
        ] + [2]

        return torch.LongTensor(sequence)
    elif cs:
        sequence = [6] if cs != "react" else []
        for sent in arr:
            sequence += [
                vocab.word2index[word]
                for word in sent
                if word in vocab.word2index and word not in ["to", "none"]
            ]

        return torch.LongTensor(sequence)
    elif emo:
        x_emo = [6]
        x_emo_mask = [6]
        for i, ew in enumerate(arr):
            x_emo += [
                vocab.word2index[ew]
                if ew in vocab.word2index
                else 0
            ]
            x_emo_mask += [vocab.word2index["CLS"]]

        assert len(x_emo) == len(x_emo_mask)
        return torch.LongTensor(x_emo), torch.LongTensor(x_emo_mask)

    else:
        x_dial = [6]
        x_mask = [6]
        for i, sentence in enumerate(arr):
            x_dial += [
                vocab.word2index[word]
                if word in vocab.word2index
                else 0
                for word in sentence
            ]
            spk = (
                vocab.word2index["USR"]
                if i % 2 == 0
                else vocab.word2index["SYS"]
            )
            x_mask += [spk for _ in range(len(sentence))]
        assert len(x_dial) == len(x_mask)

        return torch.LongTensor(x_dial), torch.LongTensor(x_mask)

if __name__ == "__main__":
    main()