/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */
#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#ifdef _USE_UNICODE_
#include <unicode/uchar.h>
#endif
#include <memory>
#include <unordered_map>

#include "../Manifold/gLLM.hpp"
#include "Dictionary.hpp"

#ifdef _USE_SENTENCEPIECE_
#include <sentencepiece_processor.h>
{
    sentencepiece::SentencePieceProcessor processor;
    const auto status = processor.Load(sTokenJsonPath);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        // error
    }
}
#endif

/**
 * - single sequence: `[CLS] X [SEP]`
   - pair of sequences: `[CLS] A [SEP] B [SEP]`
 */

std::string wstring_to_utf8(const std::wstring& wstr) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    return converter.to_bytes(wstr);
}

std::wstring utf8_to_wstring(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    return converter.from_bytes(str);
}

template <std::ctype_base::mask mask>
class IsNot {
    std::locale myLocale;
    std::ctype<char> const* myCType;

   public:
    IsNot(std::locale const& l = std::locale()) : myLocale(l), myCType(&std::use_facet<std::ctype<char>>(l)) {}
    bool operator()(char ch) const { return !myCType->is(mask, ch); }
};

typedef IsNot<std::ctype_base::space> IsNotSpace;

#ifdef _USE_UNICODE_
bool isPunctuation(UChar32 charCode) {
    auto tp                = u_charType(charCode);
    UCharCategory category = static_cast<UCharCategory>(tp);

    switch (category) {
        case U_DASH_PUNCTUATION:
        case U_START_PUNCTUATION:
        case U_END_PUNCTUATION:
        case U_CONNECTOR_PUNCTUATION:
        case U_OTHER_PUNCTUATION:
        case U_INITIAL_PUNCTUATION:
        case U_FINAL_PUNCTUATION:
            return true;
        default:
            return false;
    } /**/
}

bool _is_punctuation(UChar32 c) {
    if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) {
        return true;
    }
    if (isPunctuation(c)) {
        return true;
    }
    return false;
}

bool _is_chinese_char(UChar32 c) {
    // This defines a "Chinese character" as anything in the CJK Unicode block:
    // https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    //
    // Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    // despite its name. The modern Korean Hangul alphabet is a different block,
    // as is Japanese Hiragana and Katakana. Those alphabets are used to write
    // space-separated words, so they are not treated specially and handled
    // like all of the other languages.

    if ((c >= 0x4E00 && c <= 0x9FFF) ||    // CJK Unified Ideographs
        (c >= 0x3400 && c <= 0x4DBF) ||    // CJK Unified Ideographs Extension A
        (c >= 0x20000 && c <= 0x2A6DF) ||  // CJK Unified Ideographs Extension B
        (c >= 0x2A700 && c <= 0x2B73F) ||  // CJK Unified Ideographs Extension C
        (c >= 0x2B740 && c <= 0x2B81F) ||  // CJK Unified Ideographs Extension D
        (c >= 0x2B820 && c <= 0x2CEAF) ||  // CJK Unified Ideographs Extension E
        (c >= 0xF900 && c <= 0xFAFF) ||    // CJK Compatibility Ideographs
        (c >= 0x2F800 && c <= 0x2FA1F)) {  // CJK Compatibility Ideographs Supplement
        return true;
    }
    return false;
}

wstring pad_chinese_chars(const wstring& text) {
    vector<wchar_t> vec_padded_chars;
    for (auto& c : text) {
        if (_is_chinese_char(static_cast<UChar32>(c))) {
            vec_padded_chars.push_back(L' ');  // wide-character representation of space
            vec_padded_chars.push_back(c);
            vec_padded_chars.push_back(L' ');
        } else {
            vec_padded_chars.push_back(c);
        }
    }
    return wstring(vec_padded_chars.begin(), vec_padded_chars.end());
}

vector<wstring> run_split_on_punctuation(const wstring& text, bool split_specials, const vector<wstring>& special_tokens) {
    if (!split_specials && find(special_tokens.begin(), special_tokens.end(), text) != special_tokens.end()) {
        // we do not want to split special tokens and we found the text in the vector of special tokens
        return vector<wstring>{text};
    }
    size_t i            = 0;
    bool start_new_word = true;
    vector<vector<wchar_t>> output;

    while (i < text.length()) {
        wchar_t c = text[i];
        if (_is_punctuation(static_cast<UChar32>(c))) {
            vector<wchar_t> s;
            s.push_back(c);
            output.push_back(s);
            start_new_word = true;
        } else {
            if (start_new_word) {
                vector<wchar_t> empty_str;
                output.push_back(empty_str);
            }
            start_new_word = false;
            output.back().push_back(c);
        }
        i++;
    }

    vector<wstring> out_str;
    for (size_t i = 0; i < output.size(); i++) {
        wstring s(output[i].begin(), output[i].end());
        out_str.push_back(s);
    }
    return out_str;
}
#else
vector<wstring> run_split_on_punctuation(const wstring& text, bool split_specials, const vector<wstring>& special_tokens) {
    vector<wstring> output;
    return output;
}
wstring pad_chinese_chars(const wstring& text) { return L""; }
#endif

bool TOKEN_Special::Has(const int tok, int flag) const {
    if (tok == pad)
        return true;
    if (tok == sep)
        return true;
    if (tok == cls)
        return true;
    if (tok == mask)
        return true;
    if (tok == bos)
        return true;
    if (tok == eos)
        return true;
    if (tok == eot)
        return true;
    return false;
}

bool GTokenizer::isSpecialTok(const int tok, int flag) const { return S.Has(tok, flag); }

std::string GTokenizer::Decode(const TOKENS& ids, bool skip_pad, bool skip_special_tokens) const {
    string line;
    int nV = nVocab(), i = 0;
    for (auto id : ids) {
        assert(id < nV);
        if (skip_pad && id == S.pad)
            continue;
        if (skip_special_tokens && isSpecialTok(id))
            continue;

        string tok = decode_one(0, id);
        line += tok;
        i++;
    }
    return line;
}

bool GTokenizer::DoSomeTest(int flag) {
    if (!GTokenizer::isValid())
        return false;
    string sA     = "What is the capital of Shanghai?";
    sA            = "天命玄鸟,降而生生. 玄鸟是什么鸟?尚书·商书·胤征";
    TOKENS tokens = Encode(sA);
    string sD     = Decode(tokens);
    // assert(sA == sD);

    return true;
}

bool TOKEN_Special::isValid(int flag) const {
    /*if (eot_id < 0) {
                _WARN("[DICT] \"%s\" invalid eot_id=%d!\n", name.c_str(), eot_id);
                return false;
            }
            if (id_im_start < 0) {
                _WARN("[DICT] \"%s\" invalid id_im_start=%d!\n", name.c_str(), id_im_start);
                return false;
            }
            if (id_im_end < 0) {
                _WARN("[DICT] \"%s\" invalid id_im_end=%d!\n", name.c_str(), id_im_end);
                return false;
            }
            if (id_think_open < 0) {
                _WARN("[DICT] \"%s\" invalid id_think_open=%d!\n", name.c_str(), id_think_open);
                return false;
            }
            if (id_think_close < 0) {
                _WARN("[DICT] \"%s\" invalid id_think_close=%d!\n", name.c_str(), id_think_close);
                return false;
            }
            if (S.pad < 0) {  //
                _WARN("[DICT] \"%s\" invalid S.pad=%d!\n", name.c_str(), S.pad);
                return false;
            }
            // if (mask_id < 0) {  //
            //     _WARN("[DICT] \"%s\" invalid mask_id=%d!\n", name.c_str(), mask_id);
            //     return false;
            // }
            if (S.eos < 0) {  //
                _WARN("[DICT] \"%s\" invalid S.eos=%d!\n", name.c_str(), S.eos);
                return false;
            }
            if (assist_id < 0) {
                _WARN("[DICT] \"%s\" invalid assist_id=%d!\n", name.c_str(), assist_id);
                return false;
            }*/
    return true;
}

bool GTokenizer::isValid(bool allowEmpty, int flag) const {
    if (nVocab() <= 0) {
        _WARN("[DICT] Invalid \"%s\" with null vocab!", name.c_str());
        return allowEmpty;
    }
    std::string sFirst = T2STR(TOKEN_ID(0));
    if (sFirst.empty()) {  // sometimes, hDict->vocab.resize(151936) only has an empty vocab table
        _WARN("[DICT] \"%s\" is an empty vocab table(len=%d)!", name.c_str(), nVocab());
        if (!allowEmpty)
            return false;
    }

    if (sFirst.empty()) {
    } else if (!S.isValid()) {
        return false;
    }
    return true;
}
TOKENS GTokenizer::Encode(const std::wstring& wtext, bool encode_bos, bool encode_eos) {
    using convert_type = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_type, wchar_t> converter;
    // use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
    std::string text = converter.to_bytes(wtext);
    // string text(wtext.begin(),wtext.end());
    return Encode(text, encode_bos);
}

int GTokenizer_Heap::sLookup(const char* str, int flag) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    int vocab_size         = nVocab();
    struct TokenIndex tok  = {str, -1};  // acts as the key to search for
    struct TokenIndex* res = (struct TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(struct TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}
int GTokenizer_Heap::Lookup(const std::string& word, int flag) { return sLookup(word.c_str(), flag); }

int GTokenizer_Heap::merge_tokens_tryadd(struct Merge* heap, int n_heap, int lpos, int lid, int rpos, int rid) {
    assert(0);
    /*char str_buffer[MAX_TOKEN_LENGTH * 2 + 1];
    strcpy(str_buffer, vocab[lid].c_str());
    strcat(str_buffer, vocab[rid].c_str());
    int id = sLookup(str_buffer);
    if (id != -1) {
        float s            = scores == nullptr ? 0 : scores[id];
        struct Merge merge = {lpos, lid, rpos, rid, id, s};
        heap_insert(heap, n_heap++, merge);
    }*/
    return n_heap;
}

int GTokenizer_Heap::merge_tokens(std::vector<TOKEN_ID>& tokens, int flag) {
    // create heap for all token merge pairs
    size_t n_tokens = tokens.size(), nV = nVocab();
    struct Merge* heap = new Merge[2 * n_tokens];  // malloc(2 * n_tokens * sizeof(struct Merge));
    int n_heap         = 0;

    // insert all initial pairs
    for (int i = 0; i < n_tokens - 1; i++) {
        assert(tokens[i] < nV);
        n_heap = merge_tokens_tryadd(heap, n_heap, i, tokens[i], i + 1, tokens[i + 1]);
    }

    // merge all pairs
    while (n_heap > 0) {
        struct Merge merge = heap[0];
        heap_poptop(heap, n_heap--);

        if (tokens[merge.lpos] != merge.lid || tokens[merge.rpos] != merge.rid) {
            continue;  // this pair was already merged, skip it
        }

        // merge
        tokens[merge.lpos] = merge.resid;
        tokens[merge.rpos] = TOKEN_MAX;

        // we might have new pairs to merge
        for (int i = merge.lpos - 1; i >= 0; i--) {
            if (tokens[i] != TOKEN_MAX) {
                n_heap = merge_tokens_tryadd(heap, n_heap, i, tokens[i], merge.lpos, merge.resid);
                break;
            }
        }

        for (int i = merge.rpos + 1; i < n_tokens; i++) {
            if (tokens[i] != TOKEN_MAX) {
                n_heap = merge_tokens_tryadd(heap, n_heap, merge.lpos, merge.resid, i, tokens[i]);
                break;
            }
        }
    }

    free(heap);

    // compact tokens
    int nm_tokens = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (tokens[i] != TOKEN_MAX) {
            assert(tokens[i] < nV);
            tokens[nm_tokens++] = tokens[i];
        }
    }
    tokens.resize(nm_tokens);

    return nm_tokens;
}

TOKENS GTokenizer::Encode(const std::string& text, bool encode_bos, bool encode_eos) { assert(0 && "Encode is not Implemented ..."); }

GTokenizer::GTokenizer(Fish* dolphin, int flag) {
    config = dolphin->config;
    if (dolphin->config.model.isLoadCard()) {
        bool bRet = this->InitHF(dolphin, flag);
    }
    int nCurrentVocab = this->nVocab();
    if (nCurrentVocab != dolphin->config.model.pad_vocab_size) {
        _WARN("GTokenizer nVocab mismatch! \"vocab_size\"(in config.json)=%d.", dolphin->config.model.pad_vocab_size);
    }
}

std::string TOKEN_Special::Dump(int type, int flag) const {
    char buf[KOIFISH_MOST_LOG];
    switch (type) {
        default:
            sprintf(buf, "bos=%d,eos=%d,sep=%d,pad=%d,cls=%d,mask=%d", bos, eos, sep, pad, cls, mask);
    }
    return buf;
}

GTokenizer_GPT2::GTokenizer_GPT2(Fish* dolphin, int flag) {
    config = dolphin->config;
    assert(!dolphin->config.model.isLoadCard());
    // some init value to pass check in isValid
    S.im_start = 3, S.im_end = 4;
    S.think_open = 5, S.think_close = 6;  //"<think>")"</think>"
    S.eot     = 7;                        //  End of Tool / End of Text block
    S.unk     = 8;                        //<unk>
    S.newline = 9, S.newline2 = 10;
    S.pad    = 11;
    S.assist = 12;
}

// todo - call gpt2 tokenizer in next version
std::string GTokenizer_GPT2::T2STR(TOKEN_ID tok, int flag) const { return std::to_string((int)(tok) % 10); }

bool GTokenizer::CheckSpecialTokens(bool isAllowNone, int flag) {
    if (S.im_start < 0) {
        S.im_start = STR2T("<|im_start|>");
    }
    if (S.im_end < 0) {
        S.im_end = STR2T("<|im_end|>");
    }
    if (S.think_open < 0) {
        S.think_open = 151667;  // STR2T("<think>");
    }
    if (S.think_close < 0) {
        S.think_close = 151668;  // STR2T("</think>");
    }
    //  In Qwen (and many GPT-family models), <pad>and <eot>(end-of-text) are mapped to the same token — <|endoftext|>!!! because during pretraining there
    //  is no separate "padding" concept in the LM objective.
    if (S.pad < 0) {  //
        S.pad = STR2T("<|endoftext|>");
    }
    if (S.eos < 0) {  //
        S.eos = STR2T("<|endoftext|>");
    }
    if (S.assist < 0) {  //    <|im_start|>assistant
        S.assist = STR2T("assistant");
    }
    if (S.mask < 0) {  //    <|im_start|>assistant
        S.mask = STR2T("<M>");
    }
    if (S.noise < 0) {  //    <|im_start|>assistant
        S.noise = STR2T("<noise>");
    }
    S.newline = STR2T("\n");  // 198
    if (S.newline < 0)        //  hack
        S.newline = 198;
    S.newline2 = STR2T("\n\n");  // 271
    if (S.newline2 < 0)          //  hack
        S.newline2 = 271;
    if (S.eot < 0) {  //  <|end|>is NOT a special token in Qwen / Qwen2 / Qwen3.
        S.eot = S.eos;
    }
    return true;
}
/*
    Qwen uses a Byte Pair Encoding (BPE)​ tokenizer trained from scratch, and it has a unique characteristic:
    1. Qwen uses <|endoftext|>as both EOS and padding token, and it doesn't have a separate BOS token​ in the traditional sense.
        Any token can be the first token !
        More flexible for continuation tasks
        Matches real-world usage where text can start mid-document
        Eliminates special handling for sequence starts
    2.    "additional_special_tokens": [
                "<|im_start|>",
                "<|im_end|>",
                "<|object_ref_start|>",
                "<|object_ref_end|>",
                "<|box_start|>",
                "<|box_end|>",
                "<|quad_start|>",
                "<|quad_end|>",
                "<|vision_start|>",
                "<|vision_end|>",
                "<|vision_pad|>",
                "<|image_pad|>",
                "<|video_pad|>"
            ],
    3. "added_tokens_decoder":  @tokenizer_config.json
*/
GTokenizer_QWEN3::GTokenizer_QWEN3(Fish* dolphin, int flag) : HF_Tokenizer(dolphin, flag) {
    name = "TOKENIZER_QWEN3";
    /*config = dolphin->config;
    if (config.model.isLoadCard()) {
    } else {
    }
    bool bRet = false;
    // Although bin(@config.model.sTokenBinPath) is much faster than json file, it's deprecated since 20260721
    // LoadBin(dolphin, flag);
    if (!bRet)
        bRet = InitHF(dolphin, flag);*/

    if (nVocab() == 0) {       // vocab.clear();
        ReserveVocab(151936);  // LoadBin(dolphin, flag);
        assert(nVocab() == 151936);
        _WARN("[QWEN3] tokenizer resize to %lld(an empty vocab table).\n", nVocab());
    } else {
    }
}

GTokenizer_Heap::GTokenizer_Heap(Fish* dolphin, int flag) {
    config = dolphin->config;
    // assert(!dolphin->config.model.isLoadCard());
}

bool GTokenizer_Heap::Prepare(int flag) {
    assert(0);
    /**int vocab_size = config.model.pad_vocab_size;
    assert(vocab_size < TOKEN_MAX);

    sorted_vocab = new TokenIndex[vocab_size];
    for (int i = 0; i < vocab_size; ++i) {
        // vocab[i]            = tokens + off;
        sorted_vocab[i].str = vocab[i].c_str();
        sorted_vocab[i].id  = i;
    }

    qsort(sorted_vocab, vocab_size, sizeof(struct TokenIndex), compare_tokens);

    byte_fallback = sLookup("<0x00>");

    if (byte_fallback >= 0) {
        for (int i = 0; i < 256; i++) {
            byte_pieces[i][0] = (char)i;
            byte_pieces[i][1] = '\0';
        }
    }
    jVocab.clear();
    S.pad = sLookup("<|endoftext|>");  //  151643
    _INFO("\n[Tokenizer_HEAP] Init from \"%s\", n_vocab=%d S.pad=%d\n", config.model.sTokenJsonPath.c_str(), vocab_size, S.pad);*/

    return true;
}

bool GTokenizer::ReserveVocab(int nReserve, int flag) {
    assert(nReserve > 0);
    vocab.reserve(nReserve);
    for (int i = 0; i < nReserve; ++i) {
        vocab[std::to_string(i)] = i;
    }
    _WARN("[Tokenizer] reserve to %d(an empty vocab table).\n", nReserve);
    return true;
}

bool GTokenizer::InitHF(Fish* dolphin, int flag) {
    assert(0 && "many bugs in this function, so Deprecate!");
    return false;
}
/*
void load_single_template(char *buffer, size_t buffer_size, const string &dir_path, const char *filename) {
    string full_path = dir_path + filename;
    // construct_path(full_path, sizeof(full_path), dir_path, filename);

    memset(buffer, 0, buffer_size);
    FILE *file = fopen(full_path.c_str(), "rb");
    if (!file) {
        fprintf(stderr, "Error: Couldn't load template file %s\n", full_path.c_str());
        exit(EXIT_FAILURE);
    }
    // Read up to buffer_size - 1 to ensure null termination
    fread(buffer, 1, buffer_size - 1, file);
    fclose(file);
}*/

std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        std::cerr << "Cannot open " << path << std::endl;
        exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
}

bool GTokenizer_QWEN3::LoadBin(Fish* dolphin, int flag) {
    try {
        char tmp_word[MAX_TOKEN_LENGTH];
        string sRoot          = config.model.sCardPath;
        string tokenizer_path = config.model.sTokenBinPath;
        int vocab_size        = config.model.pad_vocab_size;
        if (!VERIFY_DIR_EXIST(tokenizer_path)) {
            _WARN("[DICT] tokenizer_path@ (\"%s\") is invalid! This would not affect the training, but the lack of tokenizer would make decode impossible.\n",
                  tokenizer_path.c_str());
            // exit(KOIFISH_LOAD_TOKENIZER);
            return false;
        }

        assert(vocab_size > 0);
        FILE* file = fopen(tokenizer_path.c_str(), "rb");
        if (file == NULL) {
            _ERROR("[DICT] Couldn't load tokenizer model @\"%s\"\n", tokenizer_path.c_str());
            exit(KOIFISH_LOAD_TOKENIZER);

            //  max_token_length = max(len(t) for t in all_tokens)
        } else {
            scores = (float*)malloc(vocab_size * sizeof(float));
            int len, nz = 0, max_token_length;
            fread(&max_token_length, sizeof(int), 1, file);  //  512?
            assert(max_token_length <= MAX_TOKEN_LENGTH);
            fread(&S.bos, sizeof(int), 1, file);
            fread(&S.eos, sizeof(int), 1, file);

            for (int i = 0; i < vocab_size; i++) {
                if (fread(scores + i, sizeof(float), 1, file) != 1) {
                    // vocab[i] = (char *)malloc(1);
                    // vocab[i][0] = 0;
                    tmp_word[0] = '\0';
                    nz++;
                } else {
                    fread(&len, sizeof(int), 1, file);
                    assert(len <= max_token_length);
                    fread(tmp_word, 1, len, file);
                    tmp_word[len] = '\0';
                }
                std::string word = tmp_word;
                vocab[word]      = i;
                // vocab.push_back(word);
            }
            fclose(file);
        }
        // GTokenizer_Heap::Prepare(0x0);
        return true;
    } catch (...) {
        return false;
    }
}

int GTokenizer::Lookup(const std::string& word, int flag) {
    // for (int i = 0; i < vocab.size(); i++)
    //     if (vocab[i] == word)
    //         return i;
    // return -1;
    return vocab[word];
}

std::vector<TOKEN_ID> GTokenizer_Heap::Encode(const std::string& text, bool encode_bos, bool encode_eos) {
    TOKENS out_tokens;
    if (encode_bos) {
        out_tokens.push_back(S.bos);
    }
    // process the raw (UTF-8) byte sequence of the input string
    char* c = (char*)text.c_str();
    while (*c != '\0') {
        char codepoint[5] = {};
        codepoint[0]      = *c++;

        if (codepoint[0] == '<' && *c == '|') {  // special token, skip until '|>'
            char* e = c + 1;
            while (*e && !(e[0] == '|' && e[1] == '>')) {
                e++;
            }
            if (e[0] == '|' && e[1] == '>' && e - c + 3 <= MAX_TOKEN_LENGTH) {
                // we found the end of the special token, try to encode it as is
                char special[MAX_TOKEN_LENGTH + 1];
                memcpy(special, c - 1, e - c + 3);
                special[e - c + 3] = '\0';
                int sid            = sLookup(special);  // sorted_vocab
                if (sid != -1) {
                    // we found special codepoint in vocab, add it as a token
                    out_tokens.push_back(sid);  // tokens[n_tokens++] = sid;
                    c = e + 2;
                    continue;
                }
            }
        }

        // this byte is a leading byte (11...), so it's a multi-byte UTF8 codepoint
        if ((codepoint[0] & 0xC0) == 0xC0) {
            for (int i = 1; i < 4 && (*c & 0xC0) == 0x80; ++i) {
                codepoint[i] = *c++;
            }
        }

        int id = sLookup(codepoint);
        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            out_tokens.push_back(id);  // tokens[n_tokens++] = id;
        } else if (byte_fallback >= 0) {
            // byte_fallback encoding: just encode each byte as a token
            for (char* fb = codepoint; *fb != '\0'; ++fb) {
                out_tokens.push_back((unsigned char)*fb + byte_fallback);  // tokens[n_tokens++] = (unsigned char)*fb + byte_fallbacks;
            }
        }
    }

    // optimized heap-based merge
    int n_tokens = merge_tokens(out_tokens);
    // add optional EOS token, if desired
    if (encode_eos) {                 // flags & TF_ENCODE_EOS
        out_tokens.push_back(S.eos);  // tokens[n_tokens++] = S.eos;
    }

    // assert(n_tokens <= tokenizer_bound(strlen(text)));
    return out_tokens;
}

//      void Encode(char *text, int *tokens, int *n_tokens,int flag);
TOKENS GTokenizer_QWEN3::Encode(const std::string& sText, bool encode_bos, bool encode_eos) {
    return HF_Tokenizer::Encode(sText, encode_bos, encode_eos);
    /*assert(isValid());
    // TOKENS tokens_heap = GTokenizer_Heap::Encode(sText, encode_bos, encode_eos);
    // return tokens_heap;  //result is different, so strange!

    TOKENS tokens;
    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char*)malloc((MAX_TOKEN_LENGTH * 2 + 1 + 2) * sizeof(char));
    char special_token[64 + 1];
    const char* text = sText.c_str();
    int nPass = 0, turn = 0, dump = 0;
    // start at 0 tokens
    // *n_tokens = 0;
    // _INFO("\n [cat] %s", sText.c_str());
    // process the raw (UTF-8) byte sequence of the input string
    for (const char* c = text; *c != 0; c++) {
        int id, found_special_token = 0;
        str_buffer[0] = *c;
        str_buffer[1] = 0;
        if (*c == '<') {  // special tokens begin with < and end with >
            int end_of_token_pos = -1;
            found_special_token  = 0;
            for (int k = 0; *c != 0 && k < 64; k++) {
                if (c[k] == '>') {
                    end_of_token_pos = k;
                    break;
                }
            }

            if (end_of_token_pos != -1) {
                strncpy(special_token, c, end_of_token_pos + 1);
                special_token[end_of_token_pos + 1] = 0;
                id                                  = Lookup(special_token);
                if (id != -1) {
                    c += end_of_token_pos;
                    found_special_token = 1;
                }
            }
        }

        // not a special token, just look up the single character
        if (!found_special_token)
            id = Lookup(str_buffer);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens.push_back(id);
            // tokens[(*n_tokens)++] = id;
        } else {
            _WARN("Warning: unknown character code point %d in input, skipping.\n", *str_buffer);
            nPass++;  //(*n_tokens)++;
        }
    }
    if (dump) {
        printf("\n[cat] tokens=%ld\t", tokens.size());
        for (int i = 0; i < tokens.size(); i++) {
            printf("%d ", tokens[i]);
        }
    }

    while (1) {            // merge the best consecutive pair each iteration
        if (turn == 13) {  // 151668
            int debug = 0;
        }
        float best_score = -1e10;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < tokens.size() - 1; i++) {
            string tt = vocab[tokens[i]] + vocab[tokens[i + 1]];
            int id    = Lookup(tt);

            if (id != -1 && scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = scores[id];
                best_id    = id;
                best_idx   = i;
            }
        }
        if (best_idx == -1) {
            break;
        }

        string tt        = vocab[tokens[best_idx]] + vocab[tokens[best_idx + 1]];
        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1);
        if (dump) {
            printf("[merge]_%d n=%ld best=%d(%d,%.4g)\t", turn++, tokens.size(), best_idx, best_id, best_score);
            for (int i = 0; i < tokens.size(); i++) {
                printf("%d ", tokens[i]);
            }
            printf("\n");
        }
    }

    free(str_buffer);
    return tokens;*/
}

int GTokenizer::nVocab(int flag) const {
    assert(vocab.size() >= 0);
    if (!isDialect) {
        return (int)(vocab.size());
    } else {
        assert(!mapT2T.empty());
        return (int)(mapT2T.size());
    }
}

bool GTokenizer::isInRange(const int* inp, size_t nz, int flag) {
    int t0 = min(0, S.bos), t1 = nVocab();  // some token maybe -1 in some case
    for (size_t i = 0; i < nz; i++, inp++) {
        if (*inp == S.bos)
            continue;
        if (*inp < 0 || *inp > t1)
            return false;
    }
    return true;
}

std::string GTokenizer::decode_one(int prev_token, int token) const {
    const std::string& piece = T2STR(token);  // vocab[token];
    // if following BOS token, sentencepiece decoder strips any leading whitespace
    if (prev_token == S.bos && piece[0] == ' ') {
        return piece.substr(1);
    }
    // return byte piece for byte fallback tokens (<0x00>, <0x01>, ..., <0xFF>)
    if (byte_fallback_start >= 0 && token >= byte_fallback_start && (token - byte_fallback_start) < 256) {
        return byte_pieces[token - byte_fallback_start];
    }
    return piece;
}
/**
std::string GTokenizer::encoding_to_debug_string(const std::vector<TOKEN_ID>& encoding) const {
    std::string token_encoding_debug_str = "";
    for (int token_id : encoding) {
        if (token_id == S.bos) {
            token_encoding_debug_str += "[<s>:" + std::to_string(token_id) + "]";
        } else if (token_id == S.eos) {
            token_encoding_debug_str += "[</s>:" + std::to_string(token_id) + "]";
        } else {
            token_encoding_debug_str += "[" + vocab[token_id] + ":" + std::to_string(token_id) + "]";
        }
    }
    return token_encoding_debug_str;
} */

static const char* LLM_KV_GENERAL_NAME         = "general.name";
static const char* LLM_KV_GENERAL_ARCHITECTURE = "general.architecture";
static const char* LLM_KV_GENERAL_FILE_TYPE    = "general.file_type";
static const char* LLM_KV_VOCAB_SIZE           = "%s.vocab_size";
static const char* LLM_KV_CONTEXT_LENGTH       = "%s.context_length";
static const char* LLM_KV_EMBEDDING_LENGTH     = "%s.embedding_length";

static const char* LLM_KV_BLOCK_COUNT                 = "%s.block_count";
static const char* LLM_KV_FEED_FORWARD_LENGTH         = "%s.feed_forward_length";
static const char* LLM_KV_ATTENTION_HEAD_COUNT        = "%s.attention.head_count";
static const char* LLM_KV_ATTENTION_LAYERNORM_RMS_EPS = "%s.attention.layer_norm_rms_epsilon";
static const char* LLM_KV_ROPE_DIMENSION_COUNT        = "%s.rope.dimension_count";
static const char* LLM_KV_ROPE_FREQ_BASE              = "%s.rope.freq_base";  // TODO load in llama.cpp
static const char* LLM_KV_ROPE_SCALE_LINEAR           = "%s.rope.scale_linear";
static const char* LLM_KV_ATTENTION_HEAD_COUNT_KV     = "%s.attention.head_count_kv";

static const char* LLM_KV_TOKENIZER_MODEL      = "tokenizer.ggml.model";
static const char* LLM_KV_TOKENIZER_LIST       = "tokenizer.ggml.tokens";
static const char* LLM_KV_TOKENIZER_TOKEN_TYPE = "tokenizer.ggml.token_type";
static const char* LLM_KV_TOKENIZER_SCORES     = "tokenizer.ggml.scores";
static const char* LLM_KV_TOKENIZER_MERGES     = "tokenizer.ggml.merges";
static const char* LLM_KV_TOKENIZER_BOS_ID     = "tokenizer.ggml.bos_token_id";
static const char* LLM_KV_TOKENIZER_EOS_ID     = "tokenizer.ggml.eos_token_id";
static const char* LLM_KV_TOKENIZER_UNK_ID     = "tokenizer.ggml.unknown_token_id";
static const char* LLM_KV_TOKENIZER_SEP_ID     = "tokenizer.ggml.seperator_token_id";
static const char* LLM_KV_TOKENIZER_PAD_ID     = "tokenizer.ggml.padding_token_id";

// { LLM_KV_DICT_VAE_LAYERS,               "dict.vae.layers"       },
// { LLM_KV_DICT_LATENT_DIM,                  "%s.dict_latent_dim"},
static const char* LLM_KV_DICT_VAE_LAYERS = "dict.vae.layers";
static const char* LLM_KV_DICT_LATENT_DIM = "%s.dict_latent_dim";

static const char* arch_str = "gruai";  // llm_arch_from_string
static char keybuf[512];
const char* kv(const char* key) {
    snprintf(keybuf, 512, key, arch_str);
    return keybuf;
};

string DictVAE::__repr__(string& suffix, string& prefix, int flag) {
    char buf[5012]     = "\0";
    const char* _ops[] = {
        "ONLY_LOAD",
        "RND_GRAD",
        "LOAD_GRAD,",
        "LOAD_GRAD_norm",
    };
    const char* tab = prefix.c_str();
    sprintf(buf + strlen(buf), "\n%s[%s]:resi=%d tpNorm=%d opOut=\"%s\" nLevel=%d\n", prefix.c_str(), "DictVAE", (int)(reserve_x), tpNorm, _ops[opOut], nLevel);

    _T_repr_(tok_embeddings, tab, buf);
    _T_repr_(_norm.w, tab, buf);
    _T_repr_(_norm.b, tab, buf);
    _T_repr_(_output.w, tab, buf);
    _T_repr_(_output.b, tab, buf);
    _T_repr_(out_u, tab, buf);
    _T_repr_(out_d, tab, buf);
    _T_repr_(out_v, tab, buf);
    if (nLevel > 0) {
        // sprintf(buf+strlen(buf),"%s\tdims=",tab);

        string s = "\n", p = prefix + "\t";
        auto vae = MAEC[0];
        sprintf(buf + strlen(buf), "%s  [%s] x %ld\tdims=", tab, vae->name.c_str(), MAEC.size());
        for (auto dim : dims) {
            sprintf(buf + strlen(buf), "%d ", dim);
        }
        sprintf(buf + strlen(buf), "%s", vae->__repr__(s, p, 0x0).c_str());
    }
    // sprintf(buf+strlen(buf),"\n");

    sprintf(buf + strlen(buf), "%s", suffix.c_str());
    if (flag > 0)
        _INFO("%s", buf);
    return buf;
}

DictVAE::DictVAE(Fish* dolphin, int flag) : VariationaAE(), dolphin(dolphin) {
    assert(dolphin->isValid());
    config = dolphin->config;
    // isDialect = config.dict_dialect == "on";
    isSVD = config.dict_logits == "svd";
    if (dolphin->wikis.size() > 0)
        wiki_tutor = dolphin->wikis[0];
    // assert(wiki_tutor!=nullptr);

    _norm.Init(dolphin);
    _output.Init(dolphin);
    reserve_x  = true;
    isMirror   = false;
    lama_embed = config.nEmbed();

    latent_dim = config.nEmbed();
    if (dolphin->config.nabla > 3)
        assert(0);
    if (!dolphin->config.vae.empty()) {
        // if(dolphin->config.nabla==3){
        dims = {(int)config.nEmbed(), 256};
        // dims = {config.nEmbed(), 1024, 256};
        // dims = {config.nEmbed(),1024,256,64};       //little difference with {config.nEmbed(),1024,256,128}
        nLevel     = dims.size() - 1;
        latent_dim = dims[nLevel];
        _INFO("%s symmetric=%d resi=%d tpNorm=%d opOut=%d nLevel=%d dims= ", __func__, (int)(isMirror), (int)(reserve_x), tpNorm, opOut, nLevel);
    } else { /**/
        if (dolphin->config.wiki_actor != "copy") {
            if (DEBUG.dict_latent_dim > 0)
                latent_dim = DEBUG.dict_latent_dim;
        }
        _INFO("%s latent_dim=%d Dialect=%s", __func__, latent_dim, "OFF");  // isDialect?"ON":"OFF"
    }
    if (dolphin->config.wiki_actor != "copy") {
        // dolphin->config.nEmbed() = latent_dim;   //Reset n_embd just like nLayerX
        // dolphin->config.SetHead(latent_dim);   // ???????
    }
    for (auto dim : dims) {
        _INFO("%d ", dim);
    }
    _INFO("\n");
}

void DictVAE::InitVAE(int flag) {
    if (nLevel == 0) {
    } else if (nLevel >= 1) {
        isLoadTokenEmbed = true;
        InitMAEC(dolphin->GetGGCTX(), dims);
        // hVarCoder hCoder = std::make_shared<VarCoder>(dolphin->GetGGCTX(), config.nEmbed(), latent_dim);
        // MAEC.push_back(hCoder);
        // encoder = TENSO(dolphin->GetGGCTX(), typNUMBER::F32, config.nEmbed(), latent_dim);
        // decoder = TENSO(dolphin->GetGGCTX(), typNUMBER::F32, latent_dim, config.nEmbed());
    }
}

void DictVAE::CreateEmbeddings(int flag) {
    assert(dolphin != nullptr);
    int n_embd = latent_dim, n_out = hDict->nVocab();
    // auto lama = dolphin->GetRawModel( );
    auto ctx = dolphin->GetGGCTX();
    if (nLevel == 0) {
    } else {
        const int last_dim = dims[dims.size() - 1];
        if (isLoadTokenEmbed) {
            const int n1 = isMirror ? n_embd : last_dim;
            if (opOut == RND_GRAD) {
                _norm.w   = GT(this, typNUMBER::F32, {n1});
                _output.w = GT(this, typNUMBER::F32, {n1, n_out});
            } else if (opOut == LOAD_GRAD_norm) {
                _output.w = GT(this, typNUMBER::F32, {n1, n_out});
            }
            return;
        }
    }
    int group      = config.Get({"model_v0", "target_group"}, 1);
    tok_embeddings = GT(this, typNUMBER::F32, {n_embd, n_out});
    _norm.w        = GT(this, typNUMBER::F32, {n_embd});
    if (!isSVD) {
        if (false) {  // TO_DO maybe useful
            _output.w = tok_embeddings;
        } else {
            if (group == 1)
                _output.w = GT(this, typNUMBER::F32, {n_embd, n_out});
            else {
                assert(n_embd % group == 0);
                _output.w = GT(this, typNUMBER::F32, {n_embd / group, n_out});
            }
        }

    } else {
        out_u = GT(this, typNUMBER::F32, {lo_rank, n_embd});
        out_v = GT(this, typNUMBER::F32, {lo_rank, n_out});
        out_d = GT(this, typNUMBER::F32, {lo_rank, lo_rank});
    }
}

hGTensor DictVAE::Embed2Output(void* ctx, hGTensor t33, int flag) {
    hGTensor tOutput = nullptr;
#ifdef _TENSOR_G_
#else
    int group  = config.Get({"model_v0", "target_group"}, 1);
    int n_embd = latent_dim, n_out = n_vocab, n_tokens = t33->ne[1], g_embd = n_embd / group;
    size_t nb0 = t33->nb[0], offset = 0;
    assert(nb0 == 4);
    assert(n_embd % group == 0);
    if (_output.w != nullptr) {  // 1024 32000
        if (group > 1) {
            if (false) {  // expert version
                for (int i = 0; i < group; i++) {
                    hGTensor embd   = ggml_view_2d(ctx, t33, g_embd, n_tokens, t33->nb[1], nb0 * i * g_embd);             // ne0,ne1,nb1,offset
                    hGTensor w      = ggml_view_2d(ctx, _output.w, g_embd, n_vocab, _output.w->nb[1], nb0 * i * g_embd);  // ne0,ne1,nb1,offset
                    hGTensor expert = ggml_mul_mat(ctx, w, embd);
                    // wB = _repeat(ctx,wB,expert);
                    tOutput = i == 0 ? expert : ggml_add(ctx, tOutput, expert);
                }
            } /*else{
                 assert(n_vocab%group==0);
                 int ne1 = n_vocab/group;
                 for(int i=0;i<group;i++){
                     hGTensor embd = ggml_view_2d(ctx, t33, g_embd, n_tokens, t33->nb[1], nb0*i*g_embd);  //ne0,ne1,nb1,offset
                     hGTensor w = ggml_view_2d(ctx, _output.w, g_embd, ne1,_output.w->nb[1], offset);  //ne0,ne1,nb1,offset
                     offset += tELEM(w)*nb0;
                     hGTensor expert = ggml_mul_mat(ctx, w, embd);
                     // wB = _repeat(ctx,wB,expert);
                     tOutput = i==0 ? expert : ggml_concat(ctx,tOutput,expert,0);
                 }
             }*/
            hGTensor embd = ggml_reshape_3d(ctx, t33, n_embd / group, group, n_tokens);
            strcpy(embd->name, "");
            ;
            gTN(embd, "%s_group%d", t33->name, group);
            embd = ggml_permute(ctx, embd, 0, 2, 1, 3);
            assert(_output.w->ne[0] == n_embd / group);
            hGTensor w = ggml_reshape_3d(ctx, _output.w, _output.w->ne[0], n_vocab / group, group);
            tOutput    = ggml_mul_mat(ctx, w, embd);
            tOutput    = ggml_cont(ctx, ggml_permute(ctx, tOutput, 0, 2, 1, 3));
            tOutput    = ggml_reshape_2d(ctx, tOutput, n_vocab, n_tokens);  // n_vocab, n_tokens
        } else
            tOutput = ggml_mul_mat(ctx, _output.w, t33);
    } else {
        hGTensor dv  = ggml_mul_mat(ctx, out_d, out_v);
        hGTensor svd = ggml_mul_mat(ctx, out_u, dv);
        tOutput      = ggml_mul_mat(ctx, svd, t33);
    }

    gTN(tOutput, "_output.w");
    // assert_shape_2d(t34, n_vocab, N*n_batch);
#endif
    return tOutput;
}

void DictVAE::Update_0(struct random_normal_distribution* rnd, int flag) {
#ifdef _TENSOR_G_
#else
    const uint32_t n_embd = config.nEmbed();
    auto lama             = dolphin->GetRawModel();
    if (isLoadTokenEmbed) {
        bool isParam = false;
        // get tensors from llama_model (possibly mmapped)
        tok_embeddings = llama_get_model_tensor(lama, TN(LLM_TENSOR_TOKEN_EMBD));
        if (isParam)
            nParams += tELEM(tok_embeddings);
        _norm.w = llama_get_model_tensor(lama, TN(LLM_TENSOR_OUTPUT_NORM));
        if (isParam)
            nParams += tELEM(_norm.w);
        _output.w = llama_get_model_tensor(lama, TN(LLM_TENSOR_OUTPUT));
        if (isParam)
            nParams += tELEM(_output.w);
    } else {
        auto ctx = dolphin->GetGGCTX();

        dolphin->InitGensor(ctx, tok_embeddings, TN(LLM_TENSOR_TOKEN_EMBD), rnd);
        dolphin->InitGensor(ctx, _norm.w, TN(LLM_TENSOR_OUTPUT_NORM), rnd);
        if (_output.w != nullptr) {
            if (_output.w != tok_embeddings)
                dolphin->InitGensor(ctx, _output.w, TN(LLM_TENSOR_OUTPUT), rnd);
        } else {
            dolphin->InitGensor(ctx, out_u, "out_u", rnd);
            dolphin->InitGensor(ctx, out_v, "out_v", rnd);
            dolphin->InitGensor(ctx, out_d, "out_d", rnd);
        }
    }
    // ggml_tensor_dequant(ctx_build,gensor,typNUMBER::F32);
    if (0) {
        assert_shape_2d(tok_embeddings, config.nEmbed(), n_vocab);
        assert_shape_1d(_norm.w, config.nEmbed());
        assert_shape_2d(_output.w, config.nEmbed(), n_vocab);
    } else {
    }
#endif
}

void DictVAE::Update_1(struct random_normal_distribution* rnd, int flag) {
    const uint32_t n_embd = config.nEmbed();
#ifdef __USE_GGML__
    bool isParam = false;
    // get tensors from llama_model (possibly mmapped)
    auto lmodel    = dolphin->GetRawModel();
    tok_embeddings = llama_get_model_tensor(lmodel, TN(LLM_TENSOR_TOKEN_EMBD));  // TN(LLM_TENSOR_TOKEN_EMBD)
    if (isParam)
        dolphin->nParams += tELEM(tok_embeddings);
    switch (opOut) {
        case ONLY_LOAD:
            _norm.w   = llama_get_model_tensor(lmodel, TN(LLM_TENSOR_OUTPUT_NORM));
            _output.w = llama_get_model_tensor(lmodel, TN(LLM_TENSOR_OUTPUT));
            break;
        case LOAD_GRAD_norm:  // bug@Optimizer::ggml_train
            _norm.w = llama_get_model_tensor(lmodel, TN(LLM_TENSOR_OUTPUT_NORM));
            assert(_norm.w->type == typNUMBER::F32);
            ggml_set_param(dolphin->GetGGCTX(), _norm.w);
            dolphin->nParams += tELEM(_norm.w);

            dolphin->InitGensor(dolphin->GetGGCTX(), _output.w, TN(LLM_TENSOR_OUTPUT), rnd);
            break;
        case LOAD_GRAD:  // bug!!!
            _norm.w = llama_get_model_tensor(lmodel, TN(LLM_TENSOR_OUTPUT_NORM));
            if (_norm.w->type != typNUMBER::F32)
                Gensor2float(dolphin->GetGGCTX(), _norm.w);
            ggml_set_param(dolphin->GetGGCTX(), _norm.w);
            dolphin->nParams += tELEM(_norm.w);
            _output.w = llama_get_model_tensor(lmodel, TN(LLM_TENSOR_OUTPUT));
            if (_output.w->type != typNUMBER::F32) {
                _output.w->data = Gensor2float(dolphin->GetGGCTX(), _output.w);
                _output.w->type = typNUMBER::F32;
            }
            ggml_set_param(dolphin->GetGGCTX(), _output.w);
            dolphin->nParams += tELEM(_output.w);
            break;
        case RND_GRAD:
            dolphin->InitGensor(dolphin->GetGGCTX(), _norm.w, TN(LLM_TENSOR_OUTPUT_NORM), rnd);
            dolphin->InitGensor(dolphin->GetGGCTX(), _output.w, TN(LLM_TENSOR_OUTPUT), rnd);
            break;

        default:
            assert(0);
    }
    assert(tok_embeddings != nullptr && _norm.w != nullptr && _output.w != nullptr);
    // ggml_tensor_dequant(ctx_build,gensor,typNUMBER::F32);
    if (0) {
        assert_shape_2d(tok_embeddings, config.nEmbed(), n_vocab);
        assert_shape_1d(_norm.w, config.nEmbed());
        assert_shape_2d(_output.w, config.nEmbed(), n_vocab);
    }
    int i = 0;
    for (auto map : MAEC) {
        std::string name = TN(LLM_DICT_DOWN, i);  //"dict.0.down.weight"
        dolphin->InitGensor(dolphin->GetGGCTX(), map->encode, TN(LLM_DICT_DOWN, i), rnd);
        if (map->decode != nullptr)
            dolphin->InitGensor(dolphin->GetGGCTX(), map->decode, TN(LLM_DICT_UP, i), rnd);
        i++;
    }

    assert(gensors.size() == 0);
#endif
}

void CDict_CHAR::LoadVocab(const char* model_path, int flag) {
    assert(strlen(model_path) == 0 || std::filesystem::exists(model_path));
    string word;
    /*enum llama_ftype ftype = LLAMA_FTYPE_MOSTLY_Q8_0;
    token_idx = -1;
    // n_vocab = len(chars);
    int nTT = n_vocab;
    score_idx = -1;
    if (score_idx == -1) {
        scores = nullptr;
    }else{
        scores = new float[nTT];
    }
    toktype_idx = -1;

    toktypes = new int[nTT];
    // memcpy(toktypes,gguf_get_arr_data(vctx, toktype_idx),sizeof(int)*nTT);
    tokenizer_name = "char_nano";
    vocab.resize(n_vocab);

    for (uint32_t i = 0; i < n_vocab; i++) {
        char a[2] = {(char)(i),'\0'};
        vocab[i] = strdup(a);
    }*/
}

void VariationaAE::save_gguf(struct gguf_context* fctx, int flag) {
#ifdef _TENSOR_G_
#else
    if (MAEC.size() == 0)
        return;
    int nLay = MAEC.size() + 1;
    assert(nLay >= 2);
    gguf_set_arr_data(fctx, kv(LLM_KV_DICT_VAE_LAYERS), GGUF_TYPE_INT32, dims.data(), nLay);
    for (auto coder : MAEC) {
        gguf_add_tensor(fctx, coder->encode);
        if (coder->decode != nullptr)
            gguf_add_tensor(fctx, coder->decode);
    }
#endif
}

//  llama.cpp/examples/tokenize/tokenize.cpp
CDict_LLAMA::CDict_LLAMA(Fish* nlp_, int flag) : DictVAE(nlp_, flag) {}

int Fish_token(CLI_params& config) {
    config.wiki_actor         = "copy";
    config.common.n_batch     = 1;
    config.model.preLogits_dB = 1;
    // config.isOnlyGPT          = true;
    arrHWIKI wikis = WIKI::MakeInstance("wikis", config, 0x0);

    hFISH fish = Fish::MakeInstance("Token_", config, wikis, Fish::ROLE_TYPE::COMMON, 0x110);

    hTokenizer hTok = std::make_shared<GTokenizer>(fish.get());  //
    // GTokenizer_WordPiece has some bug, need more time!
    // hTokenizer hTok = std::make_shared<GTokenizer_WordPiece>(fish.get());    //  [11233,1237,0,278,9100,3254]
    /*
        {12518,262,7523,318,1016,866,11}    llama.cpp
        {12514,1168,5793,2087,270,5143,2900}    GPT2
        {11228,35,1236,35,2760,2654,35,278,35,9095,35,3252}    Mistral-v0.2
    */
    std::string prompt = "when the smoke is going down";  //
    // prompt = "你相信这样的传说吗？";
    // prompt = "觉非所明，因明立所；所既妄立，生汝妄能";
    TOKENS ids                 = hTok->Encode(prompt);
    std::string decoded_prompt = hTok->Decode(ids);
    assert(prompt == decoded_prompt);

    if (0) {
        std::locale::global(std::locale());
        std::wifstream file("./tests/tokens_1.txt");
        file.imbue(std::locale(""));
        if (!file) {
            std::cerr << "Error: Unable to open input file." << std::endl;
            return 1;
        }

        // Read the entire file content into a single wide string
        std::wstringstream buffer;
        buffer << file.rdbuf();
        std::wstring input_text = buffer.str();
        std::wcout << input_text << std::endl;

        // Tokenize the input text
        auto r = hTok->Encode(input_text);
        std::wcout << "===== START=====" << std::endl;
        for (auto& x : r) {
            std::wcout << x << std::endl;
        }
        std::wcout << "===== END ======" << std::endl;
    }
    return 666;
}

GTokenizer_CHARset::GTokenizer_CHARset(Fish* nlp_, const std::vector<char>& charset_, int flag) : GTokenizer(nlp_, flag), charset(charset_) {
    if (charset.empty()) {
        charset = std::vector<char>({'_', '\n', ' ', '!', '$', '&', '\'', ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                                     'I', 'J',  'K', 'L', 'M', 'N', 'O',  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
                                     'e', 'f',  'g', 'h', 'i', 'j', 'k',  'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'});  // hack
    }
    assert(charset.size() > 0);
    assert(vocab.empty());
    int vocab_size = charset.size();
    for (auto c : charset) {
        string word{c};
        vocab[word] = (int)vocab.size();
        // vocab.push_back(word);
    }
    S.mask = 0;
    // scores = (float*)malloc(vocab_size * sizeof(float));
}

int GTokenizer_CHARset::STR2T(const char* txt, int txt_len, std::vector<TOKEN_ID>& btch, int flag) {
    int n_tokens = 0, nMost = btch.size();
    assert(txt_len <= nMost);
    unsigned char* a = (unsigned char*)(txt);
    for (int i = 0; i < txt_len; i++, a++) {
        TOKEN_ID t = (TOKEN_ID)(*a);
        // assert(t>=0 && t<n_vocab);
        btch[i] = t;
        n_tokens++;
    }
    return n_tokens;
}
std::string GTokenizer_CHARset::T2STR(TOKEN_ID tok, int flag) const {
    assert(tok >= 0 && tok < 256);
    string a = string(1, (char)tok);
    return a;
};