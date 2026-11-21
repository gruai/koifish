/**
 *  SPDX-FileCopyrightText: Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#include "Strata_Tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <queue>
#include <sstream>

#include "../CLI_params.hpp"
#include "../Shoal/Fish.hpp"
#include "../Shoal/gLLM.hpp"
#include "Dictionary.hpp"

namespace fs = std::filesystem;

GTokenizer_STRATA::GTokenizer_STRATA(Fish* dolphin, int flag) : HF_Tokenizer(dolphin, flag) {
    name = "TOKENIZER_STRATA";
    // config    = dolphin->config;
    // isDialect = config.dict.tpDialect == DIALECT_TYPE::DIALECT_on;

    // if (dolphin->config.model.isLoadCard()) {
    //     bool bRet = this->InitFromFiles(dolphin, flag);
    // }
    int nCurrentVocab = this->nVocab();
    if (nCurrentVocab != dolphin->config.model.pad_vocab_size) {
        _WARN("GTokenizer nVocab mismatch! \"vocab_size\"(in config.json)=%d.", dolphin->config.model.pad_vocab_size);
    }
}

TOKENS GTokenizer_STRATA::Encode(const std::string& sText, bool encode_bos, bool encode_eos) { return TOKENS({0}); }

/**
 * 1. "vocab" & "added_tokens"
 *  vocab Produced by the tokenizer algorithm during pretraining (BPE, SentencePiece, WordPiece, etc.).
 *  "added_tokens" is extra tokens for the training(for example: Chat tokens: <|im_start|>, <|im_end|>...),
 *      "added_tokens" has richer info than vocab(simple map str to id)
 * But in Rust/C++ tokenizer backend, Both sources are merged into a single trie or hashmap! Encoding and decoding don’t care where a token came from
 * 2. In qwen2/3
 *  The fist of "vocab" is "!", last="â½Ĺ": 151642
 *  Pad the vocab size to 151,936! 151936=187×128, the rest of the vocab is just empty rows in the embedding matrix, not missing tokens.
 * 3. "merges" is the “assembly instructions” for the BPE vocabulary.
 */
bool HF_Tokenizer::InitFromFiles(Fish* dolphin, int flag) {
    assert(dolphin != nullptr);
    fPath pathTokenJson = dolphin->config.model.sTokenJsonPath, sNull = "";
    assert(empty());
    if (!fs::exists(pathTokenJson)) {
        _WARN("[Token] InitFrom Invalid Files=\"%s\"!", pathTokenJson.c_str());
        return false;
    }
    try {
        JSON j = JSON::load(pathTokenJson);  //  /home/cys/rnd/lic/Models/Qwen3-0.6B/tokenizer.json
        if (j.empty())
            return false;
        JSON jVocab, jAddedTokens;
        int nV0 = 0;
        if (j.contains("model") && j["model"].contains("vocab")) {
            jVocab = j["model"]["vocab"];
            nV0    = jVocab.size();
        }

        if (dolphin->isModel({MD_QWEN, MD_AR_QWEN})) {
            JSON jNoise = JSON::parse(
                R"({
                "id": 151666,
                "content": "<noise>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
                })");
            bool isNeedNoise = true;
            if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
                for (const auto& item : j["added_tokens"]) {
                    std::string c    = item.value("content", "");
                    int id           = item.value("id", -1);
                    bool single_word = item.value(
                        "single_word", false);  //  false​ — It does not require word boundaries. Would match even if it sits adjacent to other  characters
                    bool special = item.value("special", false);  //  true​ — Treated as an indivisible control token:
                    bool lstrip  = item.value("lstrip", false);
                    bool rstrip  = item.value("rstrip", false);
                    bool normalized =
                        item.value("normalized", false);  // false​ — Match against the raw​ input, not the normalized version. So if the tokenizer has a
                                                          // Normalizer (lowercasing, unicode NFC, etc.), that step is bypassed for this token.
                    if (c.empty() || id == -1)
                        continue;
                    if (c == "<noise>") {
                        isNeedNoise = false;
                    }
                }
                if (isNeedNoise) {
                    j["added_tokens"].push_back(jNoise);
                }
            }
        }
        if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
            jAddedTokens = j["added_tokens"];
            nAddedToken  = jAddedTokens.size();
            // assert(nAdded==28);
        }
        std::ifstream fc(dolphin->config.model.sTokenConfigPath);  // path + "/tokenizer_config.JSON"
        bool clean_up_spaces = false;
        if (fc.is_open()) {
            std::stringstream ss_jc;
            ss_jc << fc.rdbuf();
            JSON jc = JSON::parse(ss_jc.str());
            if (jc.contains("chat_template"))
                set_chat_template(jc["chat_template"].get<std::string>());
            clean_up_spaces       = jc.value("clean_up_tokenization_spaces", false);
            j["config_overrides"] = jc;
        }
        if (!load_from_json_str(j.dump()))
            return false;

        set_clean_up_tokenization_spaces(clean_up_spaces);
        InitVocab(pathTokenJson);

        return true;
    } /*catch (JSON::parse_error& ex) {
        _INFO("[Tokenizer] Failed @%s! ERR=%s \n", pathTokenJson.c_str(), ex.what());
        return false;
        // std::cerr << "parse error at byte " << ex.byte << std::endl;
    }*/
    catch (...) {
        _ERROR("[Tokenizer] Failed @%s!\n", pathTokenJson.c_str());
        assert(0);
        return false;
    }
};

bool GTokenizer_STRATA::InitFromFiles(Fish* dolphin, int flag) {
    QWen38_strata* strata = dynamic_cast<QWen38_strata*>(dolphin);
    assert(strata != nullptr);
    string sNull                       = "", info;
    std::filesystem::path dTokenFolder = strata->dPackFolder / "tokenizer";
    int nV0 = 0, nMerge0 = 0;
    std::map<std::string, int> special_ids;
    assert(empty());
    try {
        JSON jTmp = JSON::load(dTokenFolder / "tokenizer.json");
        info      = jKV(jTmp, {"model"}, info);  //  a byte-level BPE in the GPT-2 style
        assert(info == "gpt2" && jTmp["pre"] == "qwen35");
        nV0     = jKV(jTmp, {"vocab_size"}, nV0);                             // 248320,
        nMerge0 = jKV(jTmp, {"n_merges"}, nMerge0);                           // 247587,
        for (const auto [txt, jItem] : jTmp["special_ids"].object_items()) {  //  "tokenizer.ggml.eos_token_id": 248046,
            int id = jItem.get<int>();
            assert(id > 0 && id < nV0);
            S.mapS2T[txt] = (TOKEN_ID)id;
        }
        assert(!jTmp.contains("added_tokens"));

        JSON jVocab = JSON::load(dTokenFolder / "vocab.json");  //  "!": 0, "\"": 1, "#": 2,
        if (nV0 != jVocab.size()) {
            _WARN("tokenizer.json says vocab_size=%d, but vocab.json has %ld", nV0, jVocab.size());
        }
        Json jHF = {
            {{"version", "1.0"},
             {"model",
              {
                  {"type", "BPE"},
                  {"vocab", jVocab},
              }}},
        };
        // jHF["added_tokens"] = jAddedTokens;
        // jHF["config_overrides"] = jc;

        /**
         * "merges.txt" Only for BPE-based models, but it is increasingly being replaced by the tokenizer.json format! This is a more comprehensive, single-file
         solution that bundles everything needed to run a tokenizer: the algorithm, the vocabulary, the merge rules, and the full pipeline configuration.
         * std::vector<std::string> merges;
         */

        bool clean_up_spaces = false;
        if (!load_from_json_str(jHF.dump()))
            return false;

        set_clean_up_tokenization_spaces(clean_up_spaces);
        InitVocab(dTokenFolder);
        if (!S.Init(dTokenFolder / "token_type.json", vocab))
            return false;
        return true;
    } catch (...) {
        _ERROR("[GTokenizer_STRATA] Failed @%s!\n", dTokenFolder.c_str());
        assert(0);
        return false;
    }
};

//  SpecialTokens are metadata that defines the enc/dec rules
bool TOKEN_Special::Init(CSTRING& jTypeFile, const MAP_VOCAB& vocab, int flag) {
    /**
         *  0: LLAMA_TOKEN_TYPE_UNDEFINED (Undefined)
            1: LLAMA_TOKEN_TYPE_NORMAL (A regular subword token)
            2: LLAMA_TOKEN_TYPE_UNKNOWN (The token used for unknown words)
            3: LLAMA_TOKEN_TYPE_CONTROL (A special control token, like <|im_start|> or <|eot_id|>)
            4: LLAMA_TOKEN_TYPE_USER_DEFINED (A token added by the user)
            5: LLAMA_TOKEN_TYPE_UNUSED (An unused token slot)
            6: LLAMA_TOKEN_TYPE_BYTE (A token representing a raw byte)*/
    std::vector<int> types;
    JSON jType = JSON::load(jTypeFile);
    types      = J_getArray<int>(jType);

    std::vector<std::pair<std::string, TOKEN_ID>> all_lits, always_lits;
    if (!types.empty()) {
        for (auto [txt, id] : vocab) {
            int ty = types[id];
            if (ty == 3 || ty == 4) {
                mapS2T[txt] = id;
                all_lits.push_back({txt, id});
                if (ty == 4)  // LLAMA_TOKEN_TYPE_USER_DEFINED
                    always_lits.push_back({txt, id});
                else  //  LLAMA_TOKEN_TYPE_CONTROL
                    control_tokens_.push_back(txt);
                max_special_len_ = std::max(max_special_len_, txt.size());
            }
        }
    }
    always_.Build(std::move(always_lits));
    special_.Build(std::move(all_lits));
    return true;
}

void LiteralSet::Build(std::vector<std::pair<std::string, TOKEN_ID>> lits) {
    // Longest literal first, or `<|im_end|>` could match a shorter prefix of itself.
    std::stable_sort(lits.begin(), lits.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    for (auto& v : by_first) v.clear();
    empty = true;
    for (auto& [s, id] : lits) {
        if (s.empty())
            continue;
        by_first[(uint8_t)s[0]].push_back({s, id});
        empty = false;
    }
}

bool LiteralSet::Find(std::string_view text, size_t from, size_t& start, size_t& len, TOKEN_ID& id) const {
    for (size_t p = from; p < text.size(); ++p) {
        for (const Lit& l : by_first[(uint8_t)text[p]]) {
            if (text.size() - p >= l.s.size() && std::memcmp(text.data() + p, l.s.data(), l.s.size()) == 0) {
                start = p;
                len   = l.s.size();
                id    = l.id;
                return true;
            }
        }
    }
    return false;
}

namespace strata {

// ===================================================================== Unicode classes
namespace {

struct UniRange {
    uint32_t lo, hi;
    uint8_t f;
};
#include "unicode_tables.inc"

enum : uint8_t { F_L = 1, F_M = 2, F_N = 4, F_P = 8, F_S = 16, F_SP = 32 };

// Unicode White_Space, which is what `\s` is for a str pattern in the `regex` module.
inline bool is_space_cp(uint32_t c) {
    return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

uint8_t cp_flags_slow(uint32_t c) {
    uint8_t f = 0;
    size_t lo = 0, hi = kUniRangeCount;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (kUniRanges[mid].hi < c)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < kUniRangeCount && kUniRanges[lo].lo <= c)
        f = kUniRanges[lo].f;
    if (is_space_cp(c))
        f |= F_SP;
    return f;
}

inline uint8_t cp_flags(uint32_t c) {
    static const std::array<uint8_t, 128> ascii = [] {
        std::array<uint8_t, 128> a{};
        for (uint32_t i = 0; i < 128; ++i) a[i] = cp_flags_slow(i);
        return a;
    }();
    return c < 128 ? ascii[c] : cp_flags_slow(c);
}

// --------------------------------------------------------------------- UTF-8
void put_utf8(std::string& s, uint32_t c) {
    if (c < 0x80)
        s += (char)c;
    else if (c < 0x800) {
        s += (char)(0xC0 | (c >> 6));
        s += (char)(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        s += (char)(0xE0 | (c >> 12));
        s += (char)(0x80 | ((c >> 6) & 0x3F));
        s += (char)(0x80 | (c & 0x3F));
    } else {
        s += (char)(0xF0 | (c >> 18));
        s += (char)(0x80 | ((c >> 12) & 0x3F));
        s += (char)(0x80 | ((c >> 6) & 0x3F));
        s += (char)(0x80 | (c & 0x3F));
    }
}

// Decodes `s` into code points and the byte offset of each (off has n+1 entries).  A byte that is not part of
// valid UTF-8 becomes U+FFFD of length 1: the pieces handed to BPE are sliced by byte offset, so the bytes
// themselves are never altered, only how they are classified.
void decode_utf8(std::string_view s, std::vector<uint32_t>& cp, std::vector<uint32_t>& off) {
    cp.clear();
    off.clear();
    cp.reserve(s.size());
    off.reserve(s.size() + 1);
    size_t i = 0, n = s.size();
    auto cont = [&](size_t k, uint8_t lo = 0x80, uint8_t hi = 0xBF) { return k < n && (uint8_t)s[k] >= lo && (uint8_t)s[k] <= hi; };
    while (i < n) {
        uint8_t b = (uint8_t)s[i];
        off.push_back((uint32_t)i);
        if (b < 0x80) {
            cp.push_back(b);
            ++i;
            continue;
        }
        uint32_t c = 0xFFFD;
        size_t len = 1;
        if (b >= 0xC2 && b <= 0xDF && cont(i + 1)) {
            c   = ((b & 0x1F) << 6) | ((uint8_t)s[i + 1] & 0x3F);
            len = 2;
        } else if (b >= 0xE0 && b <= 0xEF && cont(i + 1, b == 0xE0 ? 0xA0 : 0x80, b == 0xED ? 0x9F : 0xBF) && cont(i + 2)) {
            c   = ((b & 0x0F) << 12) | (((uint8_t)s[i + 1] & 0x3F) << 6) | ((uint8_t)s[i + 2] & 0x3F);
            len = 3;
        } else if (b >= 0xF0 && b <= 0xF4 && cont(i + 1, b == 0xF0 ? 0x90 : 0x80, b == 0xF4 ? 0x8F : 0xBF) && cont(i + 2) && cont(i + 3)) {
            c   = ((b & 0x07) << 18) | (((uint8_t)s[i + 1] & 0x3F) << 12) | (((uint8_t)s[i + 2] & 0x3F) << 6) | ((uint8_t)s[i + 3] & 0x3F);
            len = 4;
        }
        cp.push_back(c);
        i += len;
    }
    off.push_back((uint32_t)n);
}

// --------------------------------------------------------------------- the GPT-2 byte alphabet
struct ByteAlphabet {
    uint32_t to_cp[256];
    std::string to_utf8[256];
    std::unordered_map<uint32_t, uint8_t> from_cp;
    ByteAlphabet() {
        std::vector<int> bs, cs;
        for (int b = 0x21; b < 0x7F; ++b) bs.push_back(b);
        for (int b = 0xA1; b < 0xAD; ++b) bs.push_back(b);
        for (int b = 0xAE; b < 0x100; ++b) bs.push_back(b);
        cs    = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b)
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        for (size_t i = 0; i < bs.size(); ++i) {
            to_cp[bs[i]] = (uint32_t)cs[i];
            to_utf8[bs[i]].clear();
            put_utf8(to_utf8[bs[i]], (uint32_t)cs[i]);
            from_cp[(uint32_t)cs[i]] = (uint8_t)bs[i];
        }
    }
};
const ByteAlphabet& alphabet() {
    static const ByteAlphabet a;
    return a;
}

// --------------------------------------------------------------------- pre-tokenizer matchers
// Each returns the end of the match starting at p (limit lim), or p when no alternative matches.  They are the
// regex alternations written out, with the backtracking of each quantifier resolved by hand.
inline uint32_t lower_ascii(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }
inline bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }

// `\s*[\r\n]+` | `\s+(?!\S)` | `\s+`, the three whitespace alternatives common to both families.
size_t match_ws(const uint32_t* c, const uint8_t* fl, size_t p, size_t lim) {
    if (!(fl[p] & F_SP))
        return p;
    size_t e = p;
    while (e < lim && (fl[e] & F_SP)) ++e;
    for (size_t k = e; k > p; --k)  // \s* gives back until [\r\n]+ fits: the last newline
        if (is_nl(c[k - 1]))
            return k;
    if (e == lim)
        return e;  // \s+(?!\S) at the end of the text takes the whole run
    if (e - p >= 2)
        return e - 1;  // otherwise it leaves the last space for the next piece
    return e;          // \s+
}

size_t match_qwen35(const uint32_t* c, const uint8_t* fl, size_t p, size_t lim) {
    uint32_t ch = c[p];
    uint8_t f   = fl[p];
    // (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
    if (ch == '\'' && p + 1 < lim) {
        uint32_t a = lower_ascii(c[p + 1]);
        if (a == 's' || a == 't' || a == 'm' || a == 'd')
            return p + 2;
        if (p + 2 < lim) {
            uint32_t b = lower_ascii(c[p + 2]);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))
                return p + 3;
        }
    }
    // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    if (!(is_nl(ch) || (f & (F_L | F_N))) && p + 1 < lim && (fl[p + 1] & (F_L | F_M))) {
        size_t q = p + 2;
        while (q < lim && (fl[q] & (F_L | F_M))) ++q;
        return q;
    }
    if (f & (F_L | F_M)) {
        size_t q = p + 1;
        while (q < lim && (fl[q] & (F_L | F_M))) ++q;
        return q;
    }
    // \p{N}
    if (f & F_N)
        return p + 1;
    //  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
    {
        size_t q = (ch == ' ') ? p + 1 : p, s = q;
        while (q < lim && !(fl[q] & (F_SP | F_L | F_M | F_N))) ++q;
        if (q > s) {
            while (q < lim && is_nl(c[q])) ++q;
            return q;
        }
    }
    return match_ws(c, fl, p, lim);
}

inline bool ascii_punct(uint32_t c) { return c < 128 && c > 32 && std::ispunct((int)c); }
inline bool ascii_alpha(uint32_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }

// \p{N}{1,3}
size_t match_ds_digits(const uint32_t*, const uint8_t* fl, size_t p, size_t lim) {
    size_t q = p;
    while (q < lim && q < p + 3 && (fl[q] & F_N)) ++q;
    return q;
}

// [一-龥぀-ゟ゠-ヿ]+  (U+4E00-9FA5, U+3040-309F, U+30A0-30FF)
size_t match_ds_cjk(const uint32_t* c, const uint8_t*, size_t p, size_t lim) {
    auto in  = [](uint32_t x) { return (x >= 0x4E00 && x <= 0x9FA5) || (x >= 0x3040 && x <= 0x309F) || (x >= 0x30A0 && x <= 0x30FF); };
    size_t q = p;
    while (q < lim && in(c[q])) ++q;
    return q;
}

size_t match_ds_main(const uint32_t* c, const uint8_t* fl, size_t p, size_t lim) {
    uint32_t ch = c[p];
    uint8_t f   = fl[p];
    // [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
    if (ascii_punct(ch) && p + 1 < lim && ascii_alpha(c[p + 1])) {
        size_t q = p + 2;
        while (q < lim && ascii_alpha(c[q])) ++q;
        return q;
    }
    // [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
    if (!(is_nl(ch) || (f & (F_L | F_P | F_S))) && p + 1 < lim && (fl[p + 1] & (F_L | F_M))) {
        size_t q = p + 2;
        while (q < lim && (fl[q] & (F_L | F_M))) ++q;
        return q;
    }
    if (f & (F_L | F_M)) {
        size_t q = p + 1;
        while (q < lim && (fl[q] & (F_L | F_M))) ++q;
        return q;
    }
    //  ?[\p{P}\p{S}]+[\r\n]*
    {
        size_t q = (ch == ' ') ? p + 1 : p, s = q;
        while (q < lim && (fl[q] & (F_P | F_S))) ++q;
        if (q > s) {
            while (q < lim && is_nl(c[q])) ++q;
            return q;
        }
    }
    return match_ws(c, fl, p, lim);
}

// --------------------------------------------------------------------- JSON (just what the pack files use)
struct JsonReader {
    const std::string& s;
    size_t i = 0;
    explicit JsonReader(const std::string& str) : s(str) {}

    [[noreturn]] void fail(const char* what) const { throw std::runtime_error(std::string("tokenizer JSON: ") + what + " at byte " + std::to_string(i)); }
    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
    }
    char peek() {
        ws();
        if (i >= s.size())
            fail("unexpected end");
        return s[i];
    }
    void expect(char ch) {
        if (peek() != ch)
            fail("unexpected character");
        ++i;
    }
    bool accept(char ch) {
        if (peek() == ch) {
            ++i;
            return true;
        }
        return false;
    }
    bool literal(const char* w) {
        size_t n = std::strlen(w);
        if (s.compare(i, n, w) == 0) {
            i += n;
            return true;
        }
        return false;
    }
    uint32_t hex4() {
        if (i + 4 > s.size())
            fail("bad \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char ch = s[i++];
            v <<= 4;
            if (ch >= '0' && ch <= '9')
                v |= ch - '0';
            else if (ch >= 'a' && ch <= 'f')
                v |= ch - 'a' + 10;
            else if (ch >= 'A' && ch <= 'F')
                v |= ch - 'A' + 10;
            else
                fail("bad \\u escape");
        }
        return v;
    }
    std::string string() {
        expect('"');
        std::string out;
        while (true) {
            if (i >= s.size())
                fail("unterminated string");
            char ch = s[i++];
            if (ch == '"')
                break;
            if (ch != '\\') {
                out += ch;
                continue;
            }
            if (i >= s.size())
                fail("bad escape");
            char e = s[i++];
            switch (e) {
                case '"':
                    out += '"';
                    break;
                case '\\':
                    out += '\\';
                    break;
                case '/':
                    out += '/';
                    break;
                case 'b':
                    out += '\b';
                    break;
                case 'f':
                    out += '\f';
                    break;
                case 'n':
                    out += '\n';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case 't':
                    out += '\t';
                    break;
                case 'u': {
                    uint32_t u = hex4();
                    if (u >= 0xD800 && u < 0xDC00 && s.compare(i, 2, "\\u") == 0) {
                        size_t save = i;
                        i += 2;
                        uint32_t lo = hex4();
                        if (lo >= 0xDC00 && lo < 0xE000)
                            u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                        else
                            i = save;
                    }
                    put_utf8(out, u);
                    break;
                }
                default:
                    fail("bad escape");
            }
        }
        return out;
    }
    long long integer() {
        ws();
        size_t st = i;
        if (i < s.size() && s[i] == '-')
            ++i;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        if (i == st)
            fail("expected an integer");
        return std::stoll(s.substr(st, i - st));
    }
    void skip_value() {
        char ch = peek();
        if (ch == '"') {
            string();
            return;
        }
        if (ch == '{') {
            ++i;
            if (accept('}'))
                return;
            do {
                string();
                expect(':');
                skip_value();
            } while (accept(','));
            expect('}');
            return;
        }
        if (ch == '[') {
            ++i;
            if (accept(']'))
                return;
            do {
                skip_value();
            } while (accept(','));
            expect(']');
            return;
        }
        if (literal("true") || literal("false") || literal("null"))
            return;
        while (i < s.size() && std::strchr("+-.0123456789eE", s[i])) ++i;
    }
};

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool file_exists(const std::string& path) { return std::ifstream(path).good(); }

}  // namespace

// ===================================================================== free functions
bool pre_tokenizer_kind(const std::string& pre, PreKind& out) {
    const std::string p = pre.empty() ? "qwen35" : pre;
    if (p == "qwen35") {
        out = PreKind::Qwen35;
        return true;
    }
    if (p.rfind("deepseek", 0) == 0) {
        out = PreKind::DeepSeek3;
        return true;
    }
    return false;
}

std::string utf8_replace_invalid(const std::string& s) {
    static const char kRepl[] = "\xEF\xBF\xBD";
    std::string out;
    out.reserve(s.size());
    size_t i = 0, n = s.size();
    auto cont = [&](size_t k, uint8_t lo, uint8_t hi) { return k < n && (uint8_t)s[k] >= lo && (uint8_t)s[k] <= hi; };
    while (i < n) {
        uint8_t b = (uint8_t)s[i];
        if (b < 0x80) {
            out += s[i++];
            continue;
        }
        size_t need = 0;
        uint8_t lo = 0x80, hi = 0xBF;
        if (b >= 0xC2 && b <= 0xDF)
            need = 1;
        else if (b >= 0xE0 && b <= 0xEF) {
            need = 2;
            if (b == 0xE0)
                lo = 0xA0;
            if (b == 0xED)
                hi = 0x9F;
        } else if (b >= 0xF0 && b <= 0xF4) {
            need = 3;
            if (b == 0xF0)
                lo = 0x90;
            if (b == 0xF4)
                hi = 0x8F;
        } else {
            out += kRepl;
            ++i;
            continue;
        }
        size_t k = 1;
        bool ok  = true;
        for (; k <= need; ++k)
            if (!cont(i + k, k == 1 ? lo : 0x80, k == 1 ? hi : 0xBF)) {
                ok = false;
                break;
            }
        if (ok) {
            out.append(s, i, need + 1);
            i += need + 1;
        } else {
            out += kRepl;
            i += k;
        }  // the valid prefix and its lead byte become one U+FFFD
    }
    return out;
}

// ===================================================================== LiteralSet
void Tokenizer::LiteralSet::build(std::vector<std::pair<std::string, TokenId>> lits) {
    // Longest literal first, or `<|im_end|>` could match a shorter prefix of itself.
    std::stable_sort(lits.begin(), lits.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    for (auto& v : by_first) v.clear();
    empty = true;
    for (auto& [s, id] : lits) {
        if (s.empty())
            continue;
        by_first[(uint8_t)s[0]].push_back({s, id});
        empty = false;
    }
}

bool Tokenizer::LiteralSet::find(std::string_view text, size_t from, size_t& start, size_t& len, TokenId& id) const {
    for (size_t p = from; p < text.size(); ++p) {
        for (const Lit& l : by_first[(uint8_t)text[p]]) {
            if (text.size() - p >= l.s.size() && std::memcmp(text.data() + p, l.s.data(), l.s.size()) == 0) {
                start = p;
                len   = l.s.size();
                id    = l.id;
                return true;
            }
        }
    }
    return false;
}

// ===================================================================== Tokenizer
Tokenizer::Tokenizer(std::vector<std::string> tokens, const std::vector<std::string>& merges, std::vector<int> token_types, std::string pre,
                     std::map<std::string, int> special_ids)
    : tokens_(std::move(tokens)), pre_(std::move(pre)), token_types_(std::move(token_types)), special_ids_(std::move(special_ids)) {
    if (!pre_tokenizer_kind(pre_, kind_))
        throw UnknownPreTokenizer("This model splits text with a rule Strata doesn't have yet ('" + pre_ +
                                  "'), so it isn't loaded: running it with the wrong rule would change what it reads.");
    ids_.reserve(tokens_.size() * 2);
    for (size_t i = 0; i < tokens_.size(); ++i) ids_[tokens_[i]] = (TokenId)i;
    if (ids_.size() != tokens_.size())
        throw std::invalid_argument("vocabulary has duplicate tokens: " + std::to_string(tokens_.size()) + " entries, " + std::to_string(ids_.size()) +
                                    " unique");

    // Merge rules as (left, right) -> rank; a later duplicate pair overwrites an earlier one, as in the dict.
    ranks_.reserve(merges.size() * 2);
    merge_result_.assign(merges.size(), -1);
    for (size_t i = 0; i < merges.size(); ++i) {
        const std::string& m = merges[i];
        size_t sp            = m.find(' ');
        if (sp == std::string::npos || m.find(' ', sp + 1) != std::string::npos)
            throw std::invalid_argument("merge " + std::to_string(i) + " is not a pair: '" + m + "'");
        std::string a = m.substr(0, sp), b = m.substr(sp + 1);
        auto ia = ids_.find(a), ib = ids_.find(b);
        if (ia == ids_.end() || ib == ids_.end())
            throw std::invalid_argument("merge " + std::to_string(i) + " names a token outside the vocabulary: '" + m + "'");
        ranks_[((uint64_t)(uint32_t)ia->second << 32) | (uint32_t)ib->second] = (int32_t)i;
        auto ir                                                               = ids_.find(a + b);
        merge_result_[i]                                                      = ir == ids_.end() ? -1 : ir->second;
    }
    n_unique_merges_ = ranks_.size();

    const ByteAlphabet& al = alphabet();
    for (int b = 0; b < 256; ++b) {
        auto it      = ids_.find(al.to_utf8[b]);
        byte_sym_[b] = it == ids_.end() ? -1 : it->second;
    }

    // Literals matched directly instead of through BPE: type 4 (USER_DEFINED) always, type 3 (CONTROL) only
    // when parse_special is set.
    std::vector<std::pair<std::string, TokenId>> all_lits, always_lits;
    if (!token_types_.empty()) {
        for (size_t i = 0; i < token_types_.size() && i < tokens_.size(); ++i) {
            int ty = token_types_[i];
            if (ty == 3 || ty == 4) {
                special_tokens_[tokens_[i]] = (TokenId)i;
                all_lits.push_back({tokens_[i], (TokenId)i});
                if (ty == 4)
                    always_lits.push_back({tokens_[i], (TokenId)i});
                else
                    control_tokens_.push_back(tokens_[i]);
                max_special_len_ = std::max(max_special_len_, tokens_[i].size());
            }
        }
    }
    always_.build(std::move(always_lits));
    special_.build(std::move(all_lits));
}

std::unique_ptr<Tokenizer> Tokenizer::from_pack(const std::string& path) {
    std::string dir = path;
    if (file_exists(dir + "/tokenizer/tokenizer.json"))
        dir += "/tokenizer";

    std::string pre = "qwen35";
    std::map<std::string, int> special_ids;
    long long cfg_vocab = -1, cfg_merges = -1;
    {
        std::string txt = slurp(dir + "/tokenizer.json");
        JsonReader r(txt);
        r.expect('{');
        if (!r.accept('}')) {
            do {
                std::string key = r.string();
                r.expect(':');
                if (key == "pre")
                    pre = r.string();
                else if (key == "vocab_size")
                    cfg_vocab = r.integer();
                else if (key == "n_merges")
                    cfg_merges = r.integer();
                else if (key == "special_ids") {
                    r.expect('{');
                    if (!r.accept('}')) {
                        do {
                            std::string k = r.string();
                            r.expect(':');
                            special_ids[k] = (int)r.integer();
                        } while (r.accept(','));
                        r.expect('}');
                    }
                } else
                    r.skip_value();
            } while (r.accept(','));
            r.expect('}');
        }
    }

    std::vector<std::string> tokens;
    {
        std::string txt = slurp(dir + "/vocab.json");
        JsonReader r(txt);
        std::vector<std::pair<long long, std::string>> items;
        r.expect('{');
        if (!r.accept('}')) {
            do {
                std::string k = r.string();
                r.expect(':');
                items.push_back({r.integer(), std::move(k)});
            } while (r.accept(','));
            r.expect('}');
        }
        tokens.assign(items.size(), std::string());
        std::vector<char> seen(items.size(), 0);
        for (auto& [id, tok] : items) {
            if (id < 0 || (size_t)id >= items.size() || seen[id])
                throw std::runtime_error("vocab.json ids are not a permutation of 0..N-1");
            seen[id]   = 1;
            tokens[id] = std::move(tok);
        }
    }

    std::vector<std::string> merges;
    {
        std::string txt = slurp(dir + "/merges.txt");
        size_t pos      = 0;
        while (pos < txt.size()) {
            size_t nl = txt.find('\n', pos);
            if (nl == std::string::npos)
                nl = txt.size();
            size_t end = nl;
            if (end > pos && txt[end - 1] == '\r')
                --end;
            merges.emplace_back(txt, pos, end - pos);
            pos = nl + 1;
        }
    }

    std::vector<int> types;
    {
        std::string txt = slurp(dir + "/token_type.json");
        JsonReader r(txt);
        if (r.peek() == '[') {
            r.expect('[');
            if (!r.accept(']')) {
                do {
                    types.push_back((int)r.integer());
                } while (r.accept(','));
                r.expect(']');
            }
        } else if (!r.literal("null"))
            r.fail("token_type.json must be a list or null");
    }

    if (cfg_vocab >= 0 && (size_t)cfg_vocab != tokens.size())
        throw std::runtime_error("tokenizer.json says vocab_size " + std::to_string(cfg_vocab) + " but vocab.json has " + std::to_string(tokens.size()));
    if (cfg_merges >= 0 && (size_t)cfg_merges != merges.size())
        throw std::runtime_error("tokenizer.json says n_merges " + std::to_string(cfg_merges) + " but merges.txt has " + std::to_string(merges.size()));
    return std::make_unique<Tokenizer>(std::move(tokens), merges, std::move(types), pre, std::move(special_ids));
}

// --------------------------------------------------------------------- pre-tokenizer split
void Tokenizer::split(std::string_view seg, std::vector<std::string_view>& out) const {
    std::vector<uint32_t> cp, off;
    decode_utf8(seg, cp, off);
    const size_t n = cp.size();
    std::vector<uint8_t> fl(n);
    for (size_t i = 0; i < n; ++i) fl[i] = cp_flags(cp[i]);
    auto emit = [&](size_t a, size_t b) { out.push_back(seg.substr(off[a], off[b] - off[a])); };

    if (kind_ == PreKind::Qwen35) {
        // regex.findall: matches only; an unmatched character is skipped (none occur: every class is covered)
        size_t p = 0;
        while (p < n) {
            size_t e = match_qwen35(cp.data(), fl.data(), p, n);
            if (e > p) {
                emit(p, e);
                p = e;
            } else
                ++p;
        }
        return;
    }

    // deepseek3: a SEQUENCE.  Each pattern splits every piece the one before left into its matches and the
    // unmatched spans between them (llama.cpp's unicode_regex_split).
    using Matcher                 = size_t (*)(const uint32_t*, const uint8_t*, size_t, size_t);
    static const Matcher stages[] = {match_ds_digits, match_ds_cjk, match_ds_main};
    std::vector<std::pair<size_t, size_t>> pieces{{0, n}}, nxt;
    for (Matcher m : stages) {
        nxt.clear();
        for (auto [a, b] : pieces) {
            size_t pos = a, p = a;
            while (p < b) {
                size_t e = m(cp.data(), fl.data(), p, b);
                if (e > p) {
                    if (p > pos)
                        nxt.push_back({pos, p});
                    nxt.push_back({p, e});
                    pos = p = e;
                } else
                    ++p;
            }
            if (pos < b)
                nxt.push_back({pos, b});
        }
        pieces.swap(nxt);
    }
    for (auto [a, b] : pieces) emit(a, b);
}

std::vector<std::string> Tokenizer::pretokenize(std::string_view text) const {
    std::vector<std::string_view> views;
    split(text, views);
    return std::vector<std::string>(views.begin(), views.end());
}

// --------------------------------------------------------------------- BPE
// Merge by LOWEST RANK first, repeatedly - not left to right.  Symbols are vocabulary ids; -1 marks a merged
// string that is not a vocabulary token (it can never merge further, and is an error if it survives).
std::vector<int32_t> Tokenizer::bpe_scan(std::vector<int32_t> parts) const {
    while (parts.size() > 1) {
        long best         = -1;
        int32_t best_rank = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            if (parts[i] < 0 || parts[i + 1] < 0)
                continue;
            auto it = ranks_.find(((uint64_t)(uint32_t)parts[i] << 32) | (uint32_t)parts[i + 1]);
            if (it != ranks_.end() && (best < 0 || it->second < best_rank)) {
                best      = (long)i;
                best_rank = it->second;
            }
        }
        if (best < 0)
            break;
        parts[best] = merge_result_[best_rank];
        parts.erase(parts.begin() + best + 1);
    }
    return parts;
}

// The same result in O(n log n): symbols as a linked list, every adjacent pair with a rank in a heap keyed
// (rank, position); a popped pair whose two symbols are no longer the ones it was pushed with is stale.
std::vector<int32_t> Tokenizer::bpe_heap(std::vector<int32_t> parts) const {
    constexpr int32_t kGone = -2;
    const long n            = (long)parts.size();
    std::vector<long> nxt(n), prv(n);
    for (long i = 0; i < n; ++i) {
        nxt[i] = i + 1 < n ? i + 1 : -1;
        prv[i] = i - 1;
    }
    struct Item {
        int32_t rank;
        long pos;
        int32_t left, right;
    };
    auto later = [](const Item& a, const Item& b) { return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos; };
    std::priority_queue<Item, std::vector<Item>, decltype(later)> heap(later);
    auto push = [&](long i, long j) {
        if (parts[i] < 0 || parts[j] < 0)
            return;
        auto it = ranks_.find(((uint64_t)(uint32_t)parts[i] << 32) | (uint32_t)parts[j]);
        if (it != ranks_.end())
            heap.push({it->second, i, parts[i], parts[j]});
    };
    for (long i = 0; i + 1 < n; ++i) push(i, i + 1);
    while (!heap.empty()) {
        Item it = heap.top();
        heap.pop();
        long i = it.pos, j = nxt[i];
        if (parts[i] != it.left || j < 0 || parts[j] != it.right)
            continue;  // stale
        parts[i] = merge_result_[it.rank];
        parts[j] = kGone;
        long k   = nxt[j];
        nxt[i]   = k;
        if (k >= 0) {
            prv[k] = i;
            push(i, k);
        }
        long p = prv[i];
        if (p >= 0)
            push(p, i);
    }
    std::vector<int32_t> out;
    out.reserve(parts.size());
    for (int32_t s : parts)
        if (s != kGone)
            out.push_back(s);
    return out;
}

std::vector<TokenId> Tokenizer::bpe(std::string_view piece) const {
    std::vector<int32_t> parts(piece.size());
    for (size_t i = 0; i < piece.size(); ++i) parts[i] = byte_sym_[(uint8_t)piece[i]];
    parts = parts.size() > kHeapMin ? bpe_heap(std::move(parts)) : bpe_scan(std::move(parts));
    for (int32_t s : parts)
        if (s < 0)
            throw std::runtime_error("BPE produced a token outside the vocabulary");
    return parts;
}

void Tokenizer::encode_plain(std::string_view text, std::vector<TokenId>& out) const {
    if (text.empty())
        return;
    std::vector<std::string_view> pieces;
    split(text, pieces);
    // A piece's ids depend on the piece alone, so repeated pieces are looked up instead of merged again.
    for (std::string_view sv : pieces) {
        std::string key(sv);
        {
            std::lock_guard<std::mutex> g(cache_mu_);
            auto it = piece_ids_.find(key);
            if (it != piece_ids_.end()) {
                out.insert(out.end(), it->second.begin(), it->second.end());
                continue;
            }
        }
        std::vector<TokenId> got = bpe(sv);
        out.insert(out.end(), got.begin(), got.end());
        std::lock_guard<std::mutex> g(cache_mu_);
        if (piece_ids_.size() < kPieceCacheMax)
            piece_ids_.emplace(std::move(key), std::move(got));
    }
}

// Emit literal matches as single tokens and BPE the text between them.  The split is on the RAW text, before
// the byte mapping.  A match that starts inside a `plain` span is left to the text around it; like
// regex.finditer, the scan then resumes after that (skipped) match.
std::vector<TokenId> Tokenizer::encode_matching(std::string_view text, const LiteralSet& lits, const Spans& plain,
                                                std::vector<std::pair<size_t, size_t>>* marks) const {
    std::vector<TokenId> out;
    if (lits.empty) {
        encode_plain(text, out);
        return out;
    }
    Spans spans(plain);
    std::sort(spans.begin(), spans.end());
    size_t next_span = 0, plain_end = 0, pos = 0, scan = 0;
    size_t start, len;
    TokenId id;
    while (lits.find(text, scan, start, len, id)) {
        scan = start + len;
        while (next_span < spans.size() && spans[next_span].start <= start) {
            plain_end = std::max(plain_end, spans[next_span].end);
            ++next_span;
        }
        if (start < plain_end)
            continue;
        if (start > pos)
            encode_plain(text.substr(pos, start - pos), out);
        out.push_back(id);
        pos = start + len;
        if (marks)
            marks->push_back({pos, out.size()});
    }
    if (pos < text.size())
        encode_plain(text.substr(pos), out);
    return out;
}

std::vector<TokenId> Tokenizer::encode(std::string_view text, bool parse_special, const Spans& plain) const {
    return encode_matching(text, parse_special ? special_ : always_, plain, nullptr);
}

Tokenizer::Marked Tokenizer::encode_marked(std::string_view text, bool parse_special, const Spans& plain) const {
    Marked m;
    m.ids = encode_matching(text, parse_special ? special_ : always_, plain, &m.marks);
    return m;
}

std::string Tokenizer::token_bytes(TokenId i) const {
    if (i < 0 || (size_t)i >= tokens_.size())
        throw std::out_of_range("token id " + std::to_string(i) + " is outside the vocabulary (" + std::to_string(tokens_.size()) + ")");
    if ((size_t)i < token_types_.size() && (token_types_[i] == 3 || token_types_[i] == 4))
        return tokens_[i];  // a special token's string is the literal text
    const ByteAlphabet& al = alphabet();
    std::string raw;
    std::vector<uint32_t> cp, off;
    decode_utf8(tokens_[i], cp, off);
    for (uint32_t c : cp) {
        auto it = al.from_cp.find(c);
        if (it == al.from_cp.end())
            throw std::runtime_error("token " + std::to_string(i) + " contains a character outside the byte alphabet");
        raw += (char)it->second;
    }
    return raw;
}

std::string Tokenizer::decode_bytes(const std::vector<TokenId>& ids) const {
    std::string out;
    for (TokenId i : ids) out += token_bytes(i);
    return out;
}

std::string Tokenizer::decode(const std::vector<TokenId>& ids) const { return utf8_replace_invalid(decode_bytes(ids)); }

// ===================================================================== PromptEncoder
namespace {

size_t common_prefix_len(const std::string& a, std::string_view b) {
    size_t n = std::min(a.size(), b.size()), i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

// The first position below `limit` where the plain spans of two texts differ, or `limit`.
size_t plain_agree_until(const Spans& a, const Spans& b, size_t limit) {
    auto clip = [&](const Spans& s) {
        Spans r;
        for (const Span& x : s)
            if (x.start < limit && x.end > x.start)
                r.push_back({x.start, std::min(x.end, limit)});
        return r;
    };
    Spans ca = clip(a), cb = clip(b);
    size_t m = std::min(ca.size(), cb.size());
    for (size_t k = 0; k < m; ++k) {
        if (!(ca[k] == cb[k]))
            return ca[k].start != cb[k].start ? std::min(ca[k].start, cb[k].start) : std::min(ca[k].end, cb[k].end);
    }
    if (ca.size() != cb.size())
        return (ca.size() > cb.size() ? ca : cb)[m].start;
    return limit;
}

}  // namespace

std::vector<TokenId> PromptEncoder::encode(std::string_view text, const Spans& plain_in, bool store) {
    Spans plain(plain_in);
    std::sort(plain.begin(), plain.end());
    std::vector<std::shared_ptr<const Entry>> entries;
    {
        std::lock_guard<std::mutex> g(mu_);
        entries = entries_;
    }
    const long long margin = (long long)tok_.max_special_len() - 1;
    std::shared_ptr<const Entry> src;
    long cut_k = -1;
    for (const auto& e : entries) {
        size_t same     = plain_agree_until(e->plain, plain, common_prefix_len(e->text, text));
        long long bound = (long long)same - margin;
        // the last boundary c with c <= same - margin
        long k = (long)(std::upper_bound(e->ends.begin(), e->ends.end(), bound < 0 ? -1 : (size_t)bound) - e->ends.begin()) - 1;
        if (bound < 0)
            k = -1;
        if (k >= 0 && (!src || e->ends[k] > src->ends[cut_k])) {
            src   = e;
            cut_k = k;
        }
    }

    std::vector<TokenId> ids;
    std::vector<size_t> ends, counts;
    if (!src) {
        Tokenizer::Marked m = tok_.encode_marked(text, true, plain);
        ids                 = std::move(m.ids);
        for (auto& mk : m.marks) {
            ends.push_back(mk.first);
            counts.push_back(mk.second);
        }
        last_reused_ = 0;
    } else {
        size_t c = src->ends[cut_k], n = src->counts[cut_k];
        Spans rest;
        for (const Span& s : plain)
            if (s.end > c)
                rest.push_back({s.start > c ? s.start - c : 0, s.end - c});
        Tokenizer::Marked m = tok_.encode_marked(text.substr(c), true, rest);
        ids.assign(src->ids.begin(), src->ids.begin() + n);
        ids.insert(ids.end(), m.ids.begin(), m.ids.end());
        ends.assign(src->ends.begin(), src->ends.begin() + cut_k + 1);
        counts.assign(src->counts.begin(), src->counts.begin() + cut_k + 1);
        for (auto& mk : m.marks) {
            ends.push_back(c + mk.first);
            counts.push_back(n + mk.second);
        }
        last_reused_ = c;
    }
    if (!store)
        return ids;

    // every rendered prompt starts with the template's header, so sharing a special boundary does not make two
    // prompts one conversation: only the entry whose text is mostly reused is replaced
    std::shared_ptr<const Entry> old;
    if (src && 2 * src->ends[cut_k] > src->text.size())
        old = src;
    auto fresh = std::make_shared<Entry>();
    fresh->text.assign(text);
    fresh->plain  = plain;
    fresh->ids    = ids;
    fresh->ends   = std::move(ends);
    fresh->counts = std::move(counts);
    {
        std::lock_guard<std::mutex> g(mu_);
        std::vector<std::shared_ptr<const Entry>> next{fresh};
        for (const auto& e : entries_) {
            if (e == old)
                continue;
            if (next.size() >= keep_)
                break;
            next.push_back(e);
        }
        entries_ = std::move(next);
    }
    return ids;
}

void PromptEncoder::forget(const std::vector<TokenId>& ids) {
    std::lock_guard<std::mutex> g(mu_);
    std::vector<std::shared_ptr<const Entry>> next;
    for (const auto& e : entries_)
        if (e->ids != ids)
            next.push_back(e);
    entries_ = std::move(next);
}

}  // namespace strata
