// [todo] should add tokens in "added_tokens_decoder" in tokenizer_config.json

#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>  // for std::pair
#include <vector>

#include "../Utils/oniguruma/src/oniguruma.h"
#include "../Utils/utf8proc/utf8proc.h"
#include "Dictionary.hpp"
#include "jinja.hpp"
#include "ujson.hpp"

#ifdef _WIN32
#include <basetsd.h>
#include <windows.h>
typedef SSIZE_T ssize_t;
#endif

// using namespace tokenizer;
// namespace tokenizer {

using json = ujson::json;

// ==========================================
// C++11 Polyfills
// ==========================================
template <typename T, typename... Args>
std::unique_ptr<T> tokenizer_make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// ==========================================
// Internal Data Structures
// ==========================================

struct Encoding {
    std::vector<int> input_ids;
    std::vector<int> attention_mask;
};

struct PreTokenizedString {
    std::vector<std::string> splits;
};

// ==========================================
// Component Interfaces
// ==========================================

class Normalizer {
   public:
    virtual ~Normalizer()                                        = default;
    virtual std::string normalize(const std::string& text) const = 0;
};

class PreTokenizer {
   public:
    virtual ~PreTokenizer()                                  = default;
    virtual void pre_tokenize(PreTokenizedString& pts) const = 0;
};

class Model {
   public:
    virtual ~Model()                                                 = default;
    virtual std::vector<int> tokenize(const std::string& text) const = 0;
    virtual int token_to_id(const std::string& token) const          = 0;
    virtual std::string id_to_token(int id) const                    = 0;
    virtual size_t vocab_size() const                                = 0;
    virtual int GetVocabs(VOCAB_MAP& vocab, int flag = 0x0) const    = 0;
};

class PostProcessor {
   public:
    virtual ~PostProcessor()                       = default;
    virtual void process(Encoding& encoding) const = 0;
};

class Decoder {
   public:
    virtual ~Decoder()                                          = default;
    virtual void decode(std::vector<std::string>& tokens) const = 0;
    virtual void set_clean_up_tokenization_spaces(bool clean) {}
};

// ==========================================
// Utils
// ==========================================

static std::string OnigurumaRegexEscape(const std::string& pattern) {
    std::string escaped;
    for (char c : pattern) {
        if (c == '\\' || c == '.' || c == '+' || c == '*' || c == '?' || c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}' || c == '|' ||
            c == '^' || c == '$') {
            escaped += '\\';
        }
        escaped += c;
    }
    return escaped;
}

static std::string get_token_content(const json& j) {
    if (j.is_string())
        return j.get<std::string>();
    if (j.is_object() && j.contains("content"))
        return j["content"].get<std::string>();
    return "";
}

static std::vector<std::string> create_bytes_char_map() {
    auto u2u = [](int cp) -> std::string {
        std::string out;
        if (cp <= 0x7F)
            out += (char)cp;
        else if (cp <= 0x7FF) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else if (cp <= 0xFFFF) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
        return out;
    };
    std::vector<std::string> bs(256);
    std::unordered_map<unsigned char, std::string> temp_bs;
    for (int b = 33; b <= 126; ++b) temp_bs[(unsigned char)b] = u2u(b);
    for (int b = 161; b <= 172; ++b) temp_bs[(unsigned char)b] = u2u(b);
    for (int b = 174; b <= 255; ++b) temp_bs[(unsigned char)b] = u2u(b);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (temp_bs.find((unsigned char)b) == temp_bs.end())
            temp_bs[(unsigned char)b] = u2u(256 + n++);
        bs[b] = temp_bs[(unsigned char)b];
    }
    return bs;
}

// ==========================================
// Component Implementations
// ==========================================

class ClsOnigRegex {
   public:
    ClsOnigRegex(const std::string& pattern) : regex_(nullptr), valid_(false) {
        regex_t* reg;
        OnigErrorInfo einfo;
        onig_init();
        int r = onig_new(&reg, (uint8_t*)pattern.c_str(), (uint8_t*)(pattern.c_str() + pattern.length()), ONIG_OPTION_DEFAULT, ONIG_ENCODING_UTF8,
                         ONIG_SYNTAX_DEFAULT, &einfo);
        if (r == ONIG_NORMAL) {
            regex_ = (void*)reg;
            valid_ = true;
        } else {
            valid_ = false;
        }
    }
    ~ClsOnigRegex() {
        if (regex_)
            onig_free((regex_t*)regex_);
    }
    void* get() const { return regex_; }
    bool is_valid() const { return valid_; }

    bool search(const std::string& text, int start_offset, int end_offset, int& match_start, int& match_end) const {
        if (!valid_ || text.empty())
            return false;
        const uint8_t* str   = (const uint8_t*)text.c_str();
        const uint8_t* start = str + start_offset;
        const uint8_t* end   = str + end_offset;
        OnigRegion* region   = onig_region_new();
        int r                = onig_search((regex_t*)regex_, str, str + text.length(), start, end, region, ONIG_OPTION_NONE);
        if (r >= 0) {
            match_start = region->beg[0];
            match_end   = region->end[0];
            onig_region_free(region, 1);
            return true;
        }
        onig_region_free(region, 1);
        return false;
    }

   private:
    void* regex_;
    bool valid_;
};

class NFKCNormalizer : public Normalizer {
   public:
    std::string normalize(const std::string& text) const override {
        uint8_t* result = nullptr;
        ssize_t ret     = utf8proc_map((const uint8_t*)text.c_str(), 0, &result,
                                       (utf8proc_option_t)(UTF8PROC_NULLTERM | UTF8PROC_STABLE | UTF8PROC_COMPOSE | UTF8PROC_COMPAT));
        if (ret >= 0 && result) {
            std::string normalized_text = (const char*)result;
            free(result);
            return normalized_text;
        }
        return text;  // Return original if normalization fails
    }
};

class PrependNormalizer : public Normalizer {
    std::string prepend_;

   public:
    PrependNormalizer(const std::string& p) : prepend_(p) {}
    std::string normalize(const std::string& text) const override { return prepend_ + text; }
};

class ReplaceNormalizer : public Normalizer {
    std::string pattern_, content_;

   public:
    ReplaceNormalizer(const std::string& p, const std::string& c) : pattern_(p), content_(c) {}
    std::string normalize(const std::string& text) const override {
        if (pattern_.empty())
            return text;
        std::string out = text;
        size_t pos      = 0;
        while ((pos = out.find(pattern_, pos)) != std::string::npos) {
            out.replace(pos, pattern_.length(), content_);
            pos += content_.length();
        }
        return out;
    }
};

class SequenceNormalizer : public Normalizer {
    std::vector<std::shared_ptr<Normalizer>> normalizers_;

   public:
    SequenceNormalizer(const std::vector<std::shared_ptr<Normalizer>>& n) : normalizers_(n) {}
    std::string normalize(const std::string& text) const override {
        std::string out = text;
        for (const auto& n : normalizers_) out = n->normalize(out);
        return out;
    }
};

class BertNormalizer : public Normalizer {
    bool clean_text_, handle_chinese_chars_, strip_accents_, lowercase_;

   public:
    BertNormalizer(bool clean = true, bool chinese = true, bool accents = false, bool lower = true)
        : clean_text_(clean), handle_chinese_chars_(chinese), strip_accents_(accents), lowercase_(lower) {}

    std::string normalize(const std::string& text) const override {
        std::string out;
        const uint8_t* ptr = (const uint8_t*)text.c_str();
        size_t len = text.length(), i = 0;
        int32_t cp;
        while (i < len) {
            ssize_t r = utf8proc_iterate(ptr + i, len - i, &cp);
            if (r <= 0) {
                i++;
                continue;
            }
            std::string ch((const char*)ptr + i, r);

            // Clean text: remove control chars, replace whitespace
            if (clean_text_) {
                if (cp == '\t' || cp == '\n' || cp == '\r' || utf8proc_category(cp) == UTF8PROC_CATEGORY_ZS) {
                    out += ' ';
                    i += r;
                    continue;
                }
                if (cp == 0 || cp == 0xFFFD || utf8proc_category(cp) == UTF8PROC_CATEGORY_CC) {
                    i += r;
                    continue;
                }
            }

            // Handle Chinese chars: pad with spaces
            if (handle_chinese_chars_ && is_chinese_char(cp)) {
                out += ' ';
                out += ch;
                out += ' ';
                i += r;
                continue;
            }

            // Strip accents: decompose and skip combining marks
            if (strip_accents_) {
                uint8_t* decomposed = nullptr;
                ssize_t dlen        = utf8proc_map(ptr + i, r, &decomposed, (utf8proc_option_t)(UTF8PROC_DECOMPOSE));
                if (dlen > 0 && decomposed) {
                    for (ssize_t j = 0; decomposed[j] != 0;) {
                        int32_t dcp;
                        ssize_t dr = utf8proc_iterate(decomposed + j, -1, &dcp);
                        if (dr <= 0)
                            break;
                        if (utf8proc_category(dcp) != UTF8PROC_CATEGORY_MN) {
                            out.append((const char*)decomposed + j, dr);
                        }
                        j += dr;
                    }
                    free(decomposed);
                    i += r;
                    continue;
                }
            }

            out += ch;
            i += r;
        }
        // Lowercase if needed
        if (lowercase_) {
            std::string lower_out;
            ptr = (const uint8_t*)out.c_str();
            len = out.length();
            i   = 0;
            while (i < len) {
                ssize_t r = utf8proc_iterate(ptr + i, len - i, &cp);
                if (r <= 0) {
                    lower_out += out[i++];
                    continue;
                }
                int32_t lc = utf8proc_tolower(cp);
                char buf[8];
                int n = 0;
                if (lc <= 0x7F) {
                    buf[n++] = (char)lc;
                } else if (lc <= 0x7FF) {
                    buf[n++] = (char)(0xC0 | (lc >> 6));
                    buf[n++] = (char)(0x80 | (lc & 0x3F));
                } else if (lc <= 0xFFFF) {
                    buf[n++] = (char)(0xE0 | (lc >> 12));
                    buf[n++] = (char)(0x80 | ((lc >> 6) & 0x3F));
                    buf[n++] = (char)(0x80 | (lc & 0x3F));
                } else {
                    buf[n++] = (char)(0xF0 | (lc >> 18));
                    buf[n++] = (char)(0x80 | ((lc >> 12) & 0x3F));
                    buf[n++] = (char)(0x80 | ((lc >> 6) & 0x3F));
                    buf[n++] = (char)(0x80 | (lc & 0x3F));
                }
                lower_out.append(buf, n);
                i += r;
            }
            return lower_out;
        }
        return out;
    }

   private:
    static bool is_chinese_char(int32_t cp) {
        return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
               (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
    }
};

class SequencePreTokenizer : public PreTokenizer {
   public:
    std::vector<std::shared_ptr<PreTokenizer>> pts_;
    SequencePreTokenizer(const std::vector<std::shared_ptr<PreTokenizer>>& pts) : pts_(pts) {}
    void pre_tokenize(PreTokenizedString& pts) const override {
        for (const auto& pt : pts_) pt->pre_tokenize(pts);
    }
};

class ByteLevelPreTokenizer : public PreTokenizer {
    bool use_regex_ = false;
    mutable std::shared_ptr<ClsOnigRegex> regex_;

   public:
    ByteLevelPreTokenizer(bool use_regex = false) : use_regex_(use_regex) {
        if (use_regex_) {
            regex_ = std::make_shared<ClsOnigRegex>("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
        }
    }
    void pre_tokenize(PreTokenizedString& pts) const override {
        if (use_regex_ && regex_ && regex_->is_valid()) {
            std::vector<std::string> next_splits;
            for (const auto& s : pts.splits) {
                if (s.empty())
                    continue;
                int last_pos = 0;
                while (last_pos < (int)s.size()) {
                    int match_start = -1, match_end = -1;
                    if (regex_->search(s, last_pos, (int)s.size(), match_start, match_end)) {
                        if (match_start > last_pos) {
                            next_splits.push_back(s.substr(last_pos, match_start - last_pos));
                        }
                        if (match_end > match_start) {
                            next_splits.push_back(s.substr(match_start, match_end - match_start));
                        }
                        last_pos = match_end;
                        if (match_start == match_end)
                            last_pos++;
                    } else {
                        next_splits.push_back(s.substr(last_pos));
                        break;
                    }
                }
            }
            pts.splits = next_splits;
        }
        static auto byte_map = create_bytes_char_map();
        for (auto& s : pts.splits) {
            std::string out;
            for (unsigned char b : s) out += byte_map[b];
            s = out;
        }
    }
};

class DigitsPreTokenizer : public PreTokenizer {
    bool individual_digits_;

   public:
    DigitsPreTokenizer(bool id) : individual_digits_(id) {}
    void pre_tokenize(PreTokenizedString& pts) const override {
        std::vector<std::string> next_splits;
        for (const auto& s : pts.splits) {
            std::string current;
            for (size_t i = 0; i < s.length();) {
                int32_t cp;
                int len = utf8proc_iterate((const uint8_t*)s.data() + i, s.size() - i, &cp);
                if (len <= 0)
                    break;
                std::string c = s.substr(i, len);
                bool is_digit = (c.length() == 1 && c[0] >= '0' && c[0] <= '9');
                if (is_digit && individual_digits_) {
                    if (!current.empty()) {
                        next_splits.push_back(current);
                        current.clear();
                    }
                    next_splits.push_back(c);
                } else {
                    current += c;
                }
                i += len;
            }
            if (!current.empty())
                next_splits.push_back(current);
        }
        pts.splits = next_splits;
    }
};

class MetaspacePreTokenizer : public PreTokenizer {
   public:
    std::string replacement_;
    bool add_prefix_space_;
    MetaspacePreTokenizer(const std::string& rep, bool aps) : replacement_(rep), add_prefix_space_(aps) {}
    void pre_tokenize(PreTokenizedString& pts) const override {
        for (auto& s : pts.splits) {
            if (add_prefix_space_ && !s.empty() && s[0] != ' ') {
                s = " " + s;
            }
            std::string out;
            for (size_t i = 0; i < (int)s.size();) {
                int32_t cp;
                int len = utf8proc_iterate((const uint8_t*)s.data() + i, s.size() - i, &cp);
                if (len <= 0)
                    break;
                std::string c = s.substr(i, len);
                out += (c == " ") ? replacement_ : c;
                i += len;
            }
            s = out;
        }
    }
};

class SplitPreTokenizer : public PreTokenizer {
   public:
    std::unique_ptr<ClsOnigRegex> regex_;
    bool invert_;
    std::string behavior_;
    SplitPreTokenizer(const std::string& pattern, bool invert, const std::string& behavior = "Isolated")
        : regex_(tokenizer_make_unique<ClsOnigRegex>(pattern)), invert_(invert), behavior_(behavior) {}
    void pre_tokenize(PreTokenizedString& pts) const override {
        if (!regex_ || !regex_->is_valid())
            return;
        std::vector<std::string> new_splits;
        for (const auto& s : pts.splits) {
            int current_pos = 0;
            while (current_pos < (int)s.size()) {
                int match_start = -1, match_end = -1;
                if (regex_->search(s, current_pos, (int)s.size(), match_start, match_end)) {
                    // Found a match
                    if (invert_) {
                        // Invert means we keep the matched parts
                        if (match_end > match_start) {
                            new_splits.push_back(s.substr(match_start, match_end - match_start));
                        }
                    } else {
                        // Not inverted means we split by the matched parts
                        if (match_start > current_pos) {
                            new_splits.push_back(s.substr(current_pos, match_start - current_pos));
                        }
                        if (behavior_ == "Isolated" && match_end > match_start) {
                            new_splits.push_back(s.substr(match_start, match_end - match_start));
                        }
                        // If behavior_ == "Removed", we just don't add the matched part
                    }
                    current_pos = match_end;
                    if (match_start == match_end) {  // Handle zero-width matches to avoid infinite loops
                        current_pos++;
                    }
                } else {
                    // No more matches, add the rest of the string
                    if (current_pos < (int)s.size()) {
                        new_splits.push_back(s.substr(current_pos));
                    }
                    break;
                }
            }
        }
        pts.splits = new_splits;
    }
};

class BertPreTokenizer : public PreTokenizer {
   public:
    void pre_tokenize(PreTokenizedString& pts) const override {
        std::vector<std::string> new_splits;
        for (const auto& s : pts.splits) {
            std::string current;
            const uint8_t* ptr = (const uint8_t*)s.c_str();
            size_t len = s.length(), i = 0;
            int32_t cp;
            while (i < len) {
                ssize_t r = utf8proc_iterate(ptr + i, len - i, &cp);
                if (r <= 0) {
                    i++;
                    continue;
                }
                std::string ch((const char*)ptr + i, r);
                bool is_whitespace  = (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || utf8proc_category(cp) == UTF8PROC_CATEGORY_ZS);
                bool is_punctuation = utf8proc_category(cp) == UTF8PROC_CATEGORY_PD || utf8proc_category(cp) == UTF8PROC_CATEGORY_PS ||
                                      utf8proc_category(cp) == UTF8PROC_CATEGORY_PE || utf8proc_category(cp) == UTF8PROC_CATEGORY_PC ||
                                      utf8proc_category(cp) == UTF8PROC_CATEGORY_PO || utf8proc_category(cp) == UTF8PROC_CATEGORY_PI ||
                                      utf8proc_category(cp) == UTF8PROC_CATEGORY_PF || (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
                                      (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126);
                if (is_whitespace) {
                    if (!current.empty()) {
                        new_splits.push_back(current);
                        current.clear();
                    }
                } else if (is_punctuation) {
                    if (!current.empty()) {
                        new_splits.push_back(current);
                        current.clear();
                    }
                    new_splits.push_back(ch);
                } else {
                    current += ch;
                }
                i += r;
            }
            if (!current.empty())
                new_splits.push_back(current);
        }
        pts.splits = new_splits;
    }
};

// Moved create_bytes_char_map up

struct PairHash {
    inline size_t operator()(const std::pair<int, int>& v) const {
        // IntPairHash from boost (simplified)
        size_t seed = 0;
        seed ^= std::hash<int>{}(v.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(v.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

class BPEModel : public Model {
   public:
    bool use_byte_level_;
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<std::pair<int, int>, int, PairHash> merges_;
    mutable std::mutex cache_mutex_;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;

    BPEModel(const std::map<std::string, int>& vocab, const std::map<std::pair<int, int>, int>& merges, const std::map<std::string, int>& added_tokens,
             bool use_byte_level, bool byte_fallback)
        : use_byte_level_(use_byte_level) {
        for (auto const& x : vocab) {
            vocab_[x.first]        = x.second;
            id_to_token_[x.second] = x.first;
        }
        for (auto const& x : merges) merges_[x.first] = x.second;
    }

    int token_to_id(const std::string& token) const override {
        auto it = vocab_.find(token);
        return (it != vocab_.end()) ? it->second : -1;
    }
    std::string id_to_token(int id) const override {
        auto it = id_to_token_.find(id);
        return (it != id_to_token_.end()) ? it->second : "";
    }
    size_t vocab_size() const override { return vocab_.size(); }
    int GetVocabs(VOCAB_MAP& vocab, int flag = 0x0) const override {
        vocab = vocab_;
        return vocab.size();
    }

    std::vector<int> tokenize(const std::string& text) const override {
        if (text.empty())
            return {};
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto cit = cache_.find(text);
            if (cit != cache_.end())
                return cit->second;
        }

        std::vector<int> out;
        if (use_byte_level_) {
            static auto byte_map = create_bytes_char_map();
            for (unsigned char b : text) {
                int id = token_to_id(byte_map[b]);
                if (id != -1)
                    out.push_back(id);
            }
        } else {
            const uint8_t* ptr = (const uint8_t*)text.c_str();
            size_t len = text.length(), off = 0;
            int32_t cp;
            while (off < len) {
                ssize_t ret = utf8proc_iterate(ptr + off, len - off, &cp);
                if (ret <= 0) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned char)ptr[off]);
                    int id = token_to_id(buf);
                    if (id != -1)
                        out.push_back(id);
                    off++;
                    continue;
                }
                std::string s((const char*)ptr + off, ret);
                int id = token_to_id(s);
                if (id != -1)
                    out.push_back(id);
                else {
                    for (size_t i = 0; i < (size_t)ret; ++i) {
                        char buf[16];
                        snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned char)ptr[off + i]);
                        int bid = token_to_id(buf);
                        if (bid != -1)
                            out.push_back(bid);
                    }
                }
                off += ret;
            }
        }
        while (out.size() > 1) {
            int best = -1, min_r = 1e9;
            for (size_t i = 0; i < out.size() - 1; ++i) {
                auto it = merges_.find({out[i], out[i + 1]});
                if (it != merges_.end() && it->second < min_r) {
                    min_r = it->second;
                    best  = i;
                }
            }
            if (best == -1)
                break;
            std::string m = id_to_token(out[best]) + id_to_token(out[best + 1]);
            int nid       = token_to_id(m);
            if (nid == -1)
                break;
            out[best] = nid;
            out.erase(out.begin() + best + 1);
        }
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            cache_[text] = out;
        }
        return out;
    }

    void load(const json& v, const json& m) {
        for (auto it = v.begin(); it != v.end(); ++it) {
            vocab_[it.key()]                    = it.value().get<int>();
            id_to_token_[it.value().get<int>()] = it.key();
        }
        int rank = 0;
        for (const auto& item : m) {
            std::string s1, s2;
            if (item.is_string()) {
                std::string line = item.get<std::string>();
                size_t p         = line.find(' ');
                if (p != std::string::npos) {
                    s1 = line.substr(0, p);
                    s2 = line.substr(p + 1);
                }
            } else if (item.is_array() && item.size() >= 2) {
                s1 = item[0].get<std::string>();
                s2 = item[1].get<std::string>();
            }
            if (!s1.empty() && !s2.empty())
                merges_[{token_to_id(s1), token_to_id(s2)}] = rank++;
        }
    }
};

class WordPieceModel : public Model {
    std::string unk_token_;
    std::string continuing_subword_prefix_;
    int max_input_chars_per_word_;
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_token_;
    int unk_token_id_;

   public:
    WordPieceModel(const std::string& unk = "[UNK]", const std::string& prefix = "##", int max_chars = 100)
        : unk_token_(unk), continuing_subword_prefix_(prefix), max_input_chars_per_word_(max_chars), unk_token_id_(-1) {}

    void load(const json& v) {
        for (auto it = v.begin(); it != v.end(); ++it) {
            vocab_[it.key()]                    = it.value().get<int>();
            id_to_token_[it.value().get<int>()] = it.key();
        }
        auto it = vocab_.find(unk_token_);
        if (it != vocab_.end())
            unk_token_id_ = it->second;
    }

    int token_to_id(const std::string& token) const override {
        auto it = vocab_.find(token);
        return (it != vocab_.end()) ? it->second : unk_token_id_;
    }

    std::string id_to_token(int id) const override {
        auto it = id_to_token_.find(id);
        return (it != id_to_token_.end()) ? it->second : unk_token_;
    }

    size_t vocab_size() const override { return vocab_.size(); }
    int GetVocabs(VOCAB_MAP& vocab, int flag = 0x0) const override {
        vocab = vocab_;
        return vocab.size();
    }

    std::vector<int> tokenize(const std::string& text) const override {
        if (text.empty())
            return {};
        // If word is too long, return unk
        if ((int)text.length() > max_input_chars_per_word_) {
            return unk_token_id_ != -1 ? std::vector<int>{unk_token_id_} : std::vector<int>{};
        }
        std::vector<int> out;
        size_t start = 0;
        bool is_bad  = false;

        while (start < text.length()) {
            size_t end = text.length();
            int cur_id = -1;

            // Greedy match
            while (end > start) {
                std::string substr = text.substr(start, end - start);
                if (start > 0)
                    substr = continuing_subword_prefix_ + substr;
                auto it = vocab_.find(substr);
                if (it != vocab_.end()) {
                    cur_id = it->second;
                    break;
                }
                end--;
            }

            if (cur_id == -1) {
                is_bad = true;
                break;
            } else {
                out.push_back(cur_id);
                start = end;
            }
        }

        if (is_bad)
            return {unk_token_id_};
        return out;
    }
};

class UnigramModel : public Model {
    std::string unk_token_;
    int unk_token_id_;
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_token_;
    std::vector<double> scores_;
    bool byte_fallback_;
    size_t max_token_len_ = 0;

   public:
    UnigramModel(int unk_id = 0, bool byte_fallback = false) : unk_token_id_(unk_id), byte_fallback_(byte_fallback) {}

    void load(const json& v) {
        int idx = 0;
        for (const auto& item : v) {
            if (item.is_array() && item.size() >= 2) {
                std::string token = item[0].get<std::string>();
                double score      = item[1].get<double>();
                vocab_[token]     = idx;
                id_to_token_[idx] = token;
                scores_.push_back(score);
                if (token.length() > max_token_len_)
                    max_token_len_ = token.length();
                if (idx == unk_token_id_)
                    unk_token_ = token;
                idx++;
            }
        }
    }

    int token_to_id(const std::string& token) const override {
        auto it = vocab_.find(token);
        return (it != vocab_.end()) ? it->second : unk_token_id_;
    }

    std::string id_to_token(int id) const override {
        auto it = id_to_token_.find(id);
        return (it != id_to_token_.end()) ? it->second : unk_token_;
    }

    size_t vocab_size() const override { return vocab_.size(); }
    int GetVocabs(VOCAB_MAP& vocab, int flag = 0x0) const override {
        vocab = vocab_;
        return vocab.size();
    }

    std::vector<int> tokenize(const std::string& text) const override {
        if (text.empty())
            return {};

        size_t n = text.length();
        std::vector<double> best_scores(n + 1, -1e18);
        std::vector<int> best_ids(n + 1, -1);
        std::vector<size_t> best_prev_pos(n + 1, 0);

        best_scores[0] = 0.0;

        for (size_t i = 1; i <= n; ++i) {
            size_t start_len = (i > max_token_len_) ? (i - max_token_len_) : 0;
            for (size_t j = i - 1; j != (size_t)-1 && j >= start_len; --j) {  // Iterate backwards from i-1
                if (best_scores[j] <= -1e17)
                    continue;

                std::string sub = text.substr(j, i - j);
                auto it         = vocab_.find(sub);

                int token_id = -1;
                double score = -1e18;

                if (it != vocab_.end()) {
                    token_id = it->second;
                    score    = scores_[token_id];
                } else if (byte_fallback_ && (i - j) == 1) {
                    unsigned char b = (unsigned char)text[j];
                    char buf[16];
                    snprintf(buf, sizeof(buf), "<0x%02X>", b);
                    auto bf_it = vocab_.find(buf);
                    if (bf_it != vocab_.end()) {
                        token_id = bf_it->second;
                        score    = scores_[token_id];
                    } else {
                        token_id = unk_token_id_;
                        score    = (unk_token_id_ < (int)scores_.size()) ? scores_[unk_token_id_] : -10.0;
                    }
                } else {
                    continue;
                }

                double new_score = best_scores[j] + score;
                if (new_score > best_scores[i] || best_scores[i] <= -1e17) {
                    best_scores[i]   = new_score;
                    best_prev_pos[i] = j;
                    best_ids[i]      = token_id;
                }
            }

            // If unreachable, force greedy step with UNK as fallback if everything failed
            // (Only if not byte fallback, or byte fallback failed to match)
            if (best_scores[i] <= -1e17) {
                // Try to find the start of the current character (UTF-8)
                int char_len = 1;
                for (int k = 1; k <= 4 && (int)i - k >= 0; ++k) {
                    unsigned char c = (unsigned char)text[i - k];
                    if ((c & 0xC0) != 0x80) {  // Start byte or ASCII
                        int expected = 1;
                        if (c >= 0xF0)
                            expected = 4;
                        else if (c >= 0xE0)
                            expected = 3;
                        else if (c >= 0xC0)
                            expected = 2;

                        if (expected == k)
                            char_len = k;
                        break;
                    }
                }

                double prev_score = best_scores[i - char_len];
                if (prev_score > -1e17) {
                    double unk_score = (unk_token_id_ < (int)scores_.size()) ? scores_[unk_token_id_] : -10.0;
                    best_scores[i]   = prev_score + unk_score;
                    best_prev_pos[i] = i - char_len;
                    best_ids[i]      = unk_token_id_;
                }
            }
        }

        std::vector<int> out;
        if (best_scores[n] <= -1e17)
            return {};

        size_t cur = n;
        while (cur > 0) {
            int id = best_ids[cur];
            // Merge contiguous UNKs
            if (out.empty() || id != unk_token_id_ || out.back() != unk_token_id_) {
                out.push_back(id);
            }
            cur = best_prev_pos[cur];
        }
        std::reverse(out.begin(), out.end());
        return out;
    }
};

class TemplateProcessing : public PostProcessor {
   public:
    struct Step {
        bool is_token;
        int id;
    };
    std::vector<Step> steps_;
    TemplateProcessing(const std::vector<Step>& s) : steps_(s) {}
    void process(Encoding& enc) const override {
        std::vector<int> out;
        for (const auto& s : steps_) {
            if (s.is_token) {
                if (s.id != -1)
                    out.push_back(s.id);
            } else
                out.insert(out.end(), enc.input_ids.begin(), enc.input_ids.end());
        }
        enc.input_ids = out;
        enc.attention_mask.assign(out.size(), 1);
    }
};

class ReplaceDecoder : public Decoder {
    std::string pattern_, content_;

   public:
    ReplaceDecoder(const std::string& p, const std::string& c) : pattern_(p), content_(c) {}
    void decode(std::vector<std::string>& tokens) const override {
        for (auto& t : tokens) {
            size_t pos = 0;
            while ((pos = t.find(pattern_, pos)) != std::string::npos) {
                t.replace(pos, pattern_.length(), content_);
                pos += content_.length();
            }
        }
    }
};

class StripDecoder : public Decoder {
    std::string content_;
    int start_, stop_;

   public:
    StripDecoder(const std::string& c, int start, int stop) : content_(c), start_(start), stop_(stop) {}
    void decode(std::vector<std::string>& tokens) const override {
        if (tokens.empty())
            return;
        if (start_ > 0 && !tokens[0].empty() && tokens[0].find(content_) == 0) {
            tokens[0] = tokens[0].substr(content_.length());
        }
        if (stop_ > 0 && !tokens.back().empty()) {
            size_t pos = tokens.back().rfind(content_);
            if (pos != std::string::npos && pos + content_.length() == tokens.back().length()) {
                tokens.back() = tokens.back().substr(0, pos);
            }
        }
    }
};

class FuseDecoder : public Decoder {
   public:
    void decode(std::vector<std::string>& tokens) const override {
        if (tokens.size() <= 1)
            return;
        std::string fused;
        for (const auto& t : tokens) fused += t;
        tokens = {fused};
    }
};

class ByteFallbackDecoder : public Decoder {
   public:
    void decode(std::vector<std::string>& tokens) const override {
        for (auto& t : tokens) {
            if (t.length() >= 3 && t.substr(0, 3) == "<0x") {
                int b;
                if (sscanf(t.c_str(), "<0x%02X>", &b) == 1)
                    t = std::string(1, (char)b);
            }
        }
    }
};

class ByteLevelDecoder : public Decoder {
   public:
    void decode(std::vector<std::string>& tokens) const override {
        static auto bm = []() {
            std::unordered_map<std::string, unsigned char> m;
            auto byte_vec = create_bytes_char_map();
            for (int i = 0; i < 256; ++i) {
                if (!byte_vec[i].empty())
                    m[byte_vec[i]] = (unsigned char)i;
            }
            return m;
        }();
        for (auto& t : tokens) {
            std::string out;
            for (size_t i = 0; i < t.length();) {
                const uint8_t* tp = (const uint8_t*)t.c_str();
                int32_t cp;
                ssize_t r = utf8proc_iterate(tp + i, t.length() - i, &cp);
                if (r > 0) {
                    std::string ch(t.substr(i, r));
                    auto it = bm.find(ch);
                    if (it != bm.end())
                        out += (char)it->second;
                    else
                        out += ch;
                    i += r;
                } else
                    out += t[i++];
            }
            t = out;
        }
    }
};

class WordPieceDecoder : public Decoder {
    std::string prefix_;
    bool cleanup_;

   public:
    WordPieceDecoder(const std::string& prefix = "##", bool cleanup = true) : prefix_(prefix), cleanup_(cleanup) {}

    void set_clean_up_tokenization_spaces(bool clean) override { cleanup_ = clean; }

    void decode(std::vector<std::string>& tokens) const override {
        std::string out;
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::string token = tokens[i];
            bool is_suffix    = (token.rfind(prefix_, 0) == 0);
            if (is_suffix)
                token = token.substr(prefix_.length());

            if (i == 0) {
                out += token;
            } else {
                bool add_space = true;
                if (is_suffix)
                    add_space = false;
                else {
                    char first_char = token.empty() ? 0 : token[0];
                    // Always clean basic punctuation
                    if (strchr(".,!?", first_char))
                        add_space = false;

                    if (cleanup_) {
                        // Clean extended punctuation if enabled
                        if (first_char == '\'')
                            add_space = false;

                        // No space after single quote
                        if (add_space && !out.empty()) {
                            char last_char = out.back();
                            if (last_char == '\'')
                                add_space = false;
                        }
                    }
                }

                if (add_space)
                    out += " ";
                out += token;
            }
        }
        tokens.clear();
        tokens.push_back(out);
    }
};

class MetaspaceDecoder : public Decoder {
    std::string replacement_;
    bool add_prefix_space_;

   public:
    MetaspaceDecoder(const std::string& rep = "▁", bool aps = true) : replacement_(rep), add_prefix_space_(aps) {}
    void decode(std::vector<std::string>& tokens) const override {
        for (auto& t : tokens) {
            std::string out;
            size_t i = 0;
            while (i < t.length()) {
                if (t.substr(i, replacement_.length()) == replacement_) {
                    out += " ";
                    i += replacement_.length();
                } else {
                    out += t[i++];
                }
            }
            t = out;
        }
        if (add_prefix_space_ && !tokens.empty()) {
            if (!tokens[0].empty() && tokens[0][0] == ' ') {
                tokens[0] = tokens[0].substr(1);
            }
        }
    }
};

class SequenceDecoder : public Decoder {
    std::vector<std::shared_ptr<Decoder>> decoders_;

   public:
    SequenceDecoder(const std::vector<std::shared_ptr<Decoder>>& d) : decoders_(d) {}
    void decode(std::vector<std::string>& tokens) const override {
        for (const auto& d : decoders_) d->decode(tokens);
    }
    void set_clean_up_tokenization_spaces(bool clean) override {
        for (const auto& d : decoders_) d->set_clean_up_tokenization_spaces(clean);
    }
};

class CoreDecoder : public Decoder {
   public:
    std::shared_ptr<Model> model_;
    CoreDecoder(std::shared_ptr<Model> m) : model_(m) {}
    void decode(std::vector<std::string>& tokens) const override { /* Not used in this design */ }
};

// ==========================================
// HF_Tokenizer::Impl
// ==========================================

struct HF_Tokenizer::Impl {
    struct AddedToken {
        int id;
        std::string content;
        bool special;
        bool lstrip;
        bool rstrip;
        bool normalized;
    };
    std::shared_ptr<Normalizer> normalizer_;
    std::shared_ptr<PreTokenizer> pre_tokenizer_;
    std::shared_ptr<Model> model_;
    std::shared_ptr<PostProcessor> post_processor_;
    std::shared_ptr<Decoder> decoder_;
    TOKEN_Special special_tokens_;
    std::shared_ptr<ClsOnigRegex> added_tokens_regex_;
    std::vector<AddedToken> added_tokens_;
    std::string chat_template_;
    std::shared_ptr<jinja::Template> jinja_template_;

    std::vector<int> encode(const HF_Tokenizer* public_api, const std::string& text, bool add_special_tokens) const {
        if (text.empty())
            return {};
        std::vector<int> input_ids;

        // 1. Identify added tokens in original text (assuming normalized: false for most)
        std::vector<std::pair<std::string, bool>> units;
        size_t last = 0;
        while (last < text.length()) {
            int match_start = -1, match_end = -1;
            if (added_tokens_regex_ && added_tokens_regex_->search(text, (int)last, (int)text.length(), match_start, match_end)) {
                std::string match_token = text.substr(match_start, match_end - match_start);
                const AddedToken* at    = nullptr;
                for (const auto& t : added_tokens_) {
                    if (t.content == match_token) {
                        at = &t;
                        break;
                    }
                }

                size_t prefix_start = last;
                size_t prefix_end   = match_start;
                size_t next_start   = match_end;

                if (at) {
                    if (at->lstrip) {
                        while (prefix_end > prefix_start && isspace((unsigned char)text[prefix_end - 1])) prefix_end--;
                    }
                    if (at->rstrip) {
                        while (next_start < text.length() && isspace((unsigned char)text[next_start])) next_start++;
                    }
                }

                if (prefix_end > prefix_start)
                    units.push_back({text.substr(prefix_start, prefix_end - prefix_start), false});
                units.push_back({match_token, true});
                last = next_start;
            } else {
                units.push_back({text.substr(last), false});
                break;
            }
        }

        if (add_special_tokens && special_tokens_.bos != -1)
            input_ids.push_back(special_tokens_.bos);

        for (const auto& unit : units) {
            if (unit.second) {
                int id = public_api->token_to_id(unit.first);
                if (id != -1)
                    input_ids.push_back(id);
            } else {
                // 2. Normalize only non-special units
                std::string normalized = normalizer_ ? normalizer_->normalize(unit.first) : unit.first;
                if (normalized.empty())
                    continue;

                // 3. Pre-tokenize and model tokenize
                PreTokenizedString pts;
                pts.splits.push_back(normalized);

                if (pre_tokenizer_)
                    pre_tokenizer_->pre_tokenize(pts);

                for (const auto& s : pts.splits) {
                    auto ids = model_->tokenize(s);
                    input_ids.insert(input_ids.end(), ids.begin(), ids.end());
                }
            }
        }
        return input_ids;
    }

    void set_clean_up_tokenization_spaces(bool clean) {
        if (decoder_) {
            decoder_->set_clean_up_tokenization_spaces(clean);
        }
    }

    bool load_from_json(HF_Tokenizer* public_api, const json& j) {
        if (j.contains("model") && j["model"].is_object()) {
            std::string model_type = j["model"].value("type", "");
            // Auto-detect model type if not specified
            if (model_type.empty()) {
                if (j["model"].contains("vocab") && j["model"]["vocab"].is_array()) {
                    model_type = "Unigram";
                } else if (j["model"].contains("continuing_subword_prefix") ||
                           (j["model"].contains("vocab") && j["model"]["vocab"].is_object() && !j["model"].contains("merges"))) {
                    model_type = "WordPiece";
                } else {
                    model_type = "BPE";
                }
            }

            if (model_type == "WordPiece") {
                // WordPiece model
                std::string unk_token = j["model"].value("unk_token", "[UNK]");
                std::string prefix    = j["model"].value("continuing_subword_prefix", "##");
                int max_chars         = j["model"].value("max_input_chars_per_word", 100);
                auto wp               = std::make_shared<WordPieceModel>(unk_token, prefix, max_chars);
                if (j["model"].contains("vocab")) {
                    wp->load(j["model"]["vocab"]);
                }
                this->model_ = wp;
            } else if (model_type == "Unigram") {
                // Unigram model
                int unk_id         = j["model"].value("unk_id", 0);
                bool byte_fallback = j["model"].value("byte_fallback", false);
                auto ug            = std::make_shared<UnigramModel>(unk_id, byte_fallback);
                if (j["model"].contains("vocab") && j["model"]["vocab"].is_array()) {
                    ug->load(j["model"]["vocab"]);
                }
                this->model_ = ug;
            } else {
                // BPE model (default)
                auto vocab = j["model"]["vocab"].get<std::map<std::string, int>>();
                std::map<std::pair<int, int>, int> merges;
                if (j["model"].contains("merges")) {
                    int rank = 0;
                    for (const auto& item : j["model"]["merges"]) {
                        std::string s1, s2;
                        if (item.is_string()) {
                            std::string line = item.get<std::string>();
                            size_t pos       = line.find(' ');
                            if (pos != std::string::npos) {
                                s1 = line.substr(0, pos);
                                s2 = line.substr(pos + 1);
                            }
                        } else if (item.is_array() && item.size() >= 2) {
                            s1 = item[0].get<std::string>();
                            s2 = item[1].get<std::string>();
                        }
                        if (!s1.empty() && !s2.empty() && vocab.count(s1) && vocab.count(s2))
                            merges[{vocab[s1], vocab[s2]}] = rank++;
                    }
                }
                std::map<std::string, int> added_tokens;
                bool byte_fallback = false;
                if (j["model"].contains("byte_fallback"))
                    byte_fallback = j["model"]["byte_fallback"].get<bool>();

                bool use_byte_level = false;
                auto check_bl       = [](const json& c) -> bool {
                    if (!c.is_object())
                        return false;
                    if (c.value("type", "") == "ByteLevel")
                        return true;
                    if (c.contains("pretokenizers")) {
                        for (const auto& s : c["pretokenizers"])
                            if (s.is_object() && s.value("type", "") == "ByteLevel")
                                return true;
                    }
                    if (c.contains("processors")) {
                        for (const auto& s : c["processors"])
                            if (s.is_object() && s.value("type", "") == "ByteLevel")
                                return true;
                    }
                    if (c.contains("decoders")) {
                        for (const auto& s : c["decoders"])
                            if (s.is_object() && s.value("type", "") == "ByteLevel")
                                return true;
                    }
                    return false;
                };
                if (check_bl(j.value("pre_tokenizer", json())))
                    use_byte_level = true;
                if (check_bl(j.value("post_processor", json())))
                    use_byte_level = true;
                if (check_bl(j.value("decoder", json())))
                    use_byte_level = true;

                // If we have a ByteLevelPreTokenizer in the sequence, BPEModel should not do the mapping itself
                bool pt_has_byte_level = false;
                if (j.contains("pre_tokenizer") && j["pre_tokenizer"].is_object()) {
                    auto pt = j["pre_tokenizer"];
                    if (pt.value("type", "") == "ByteLevel")
                        pt_has_byte_level = true;
                    else if (pt.value("type", "") == "Sequence" && pt.contains("pretokenizers")) {
                        for (const auto& s : pt["pretokenizers"])
                            if (s.is_object() && s.value("type", "") == "ByteLevel")
                                pt_has_byte_level = true;
                    }
                }

                auto bpe     = std::make_shared<BPEModel>(vocab, merges, added_tokens, use_byte_level && !pt_has_byte_level, byte_fallback);
                this->model_ = bpe;
            }
        }
        if (j.contains("normalizer") && !j["normalizer"].is_null()) {
            auto create_norm = [&](const json& s) -> std::shared_ptr<Normalizer> {
                std::string type = s.value("type", "");
                if (type == "NFKC")
                    return std::make_shared<NFKCNormalizer>();
                if (type == "Precompiled") {
                    // Precompiled usually implies NFKC + some char map.
                    // For GTE/XLM-R, it seems to map ZWJ (\u200d) to space, causing split.
                    std::vector<std::shared_ptr<Normalizer>> norms;
                    norms.push_back(std::make_shared<NFKCNormalizer>());
                    norms.push_back(std::make_shared<ReplaceNormalizer>("\xE2\x80\x8D", " "));  // ZWJ -> Space
                    return std::make_shared<SequenceNormalizer>(norms);
                }
                if (type == "Prepend")
                    return std::make_shared<PrependNormalizer>(s.value("prepend", ""));
                if (type == "Lowercase")
                    return std::make_shared<BertNormalizer>(false, false, false, true);
                if (type == "StripAccents")
                    return std::make_shared<BertNormalizer>(false, false, true, false);
                if (type == "NFKD")
                    return std::make_shared<NFKCNormalizer>();  // Approximate with NFKC for now
                if (type == "Replace") {
                    std::string p;
                    if (s.contains("pattern") && s["pattern"].is_object())
                        p = s["pattern"].value("String", "");
                    else if (s.contains("pattern") && s["pattern"].is_string())
                        p = s["pattern"].get<std::string>();
                    return std::make_shared<ReplaceNormalizer>(p, s.value("content", ""));
                }
                if (type == "BertNormalizer") {
                    bool lowercase     = s.value("lowercase", true);
                    bool strip_accents = lowercase;
                    if (s.contains("strip_accents") && !s["strip_accents"].is_null()) {
                        strip_accents = s["strip_accents"].get<bool>();
                    }
                    return std::make_shared<BertNormalizer>(s.value("clean_text", true), s.value("handle_chinese_chars", true), strip_accents, lowercase);
                }
                return nullptr;
            };
            if (j["normalizer"].value("type", "") == "Sequence") {
                std::vector<std::shared_ptr<Normalizer>> norms;
                for (const auto& s : j["normalizer"]["normalizers"]) {
                    auto n = create_norm(s);
                    if (n)
                        norms.push_back(n);
                }
                this->normalizer_ = std::make_shared<SequenceNormalizer>(norms);
            } else {
                this->normalizer_ = create_norm(j["normalizer"]);
            }
        }
        if (j.contains("decoder") && !j["decoder"].is_null()) {
            auto create_dec = [&](const json& s) -> std::shared_ptr<Decoder> {
                std::string type = s.value("type", "");
                if (type == "Replace") {
                    std::string p;
                    if (s["pattern"].is_object())
                        p = s["pattern"].value("String", "");
                    else if (s["pattern"].is_string())
                        p = s["pattern"].get<std::string>();
                    return std::make_shared<ReplaceDecoder>(p, s.value("content", ""));
                }
                if (type == "ByteFallback")
                    return std::make_shared<ByteFallbackDecoder>();
                if (type == "ByteLevel")
                    return std::make_shared<ByteLevelDecoder>();
                if (type == "Fuse")
                    return std::make_shared<FuseDecoder>();
                if (type == "Strip")
                    return std::make_shared<StripDecoder>(s.value("content", ""), s.value("start", 0), s.value("stop", 0));
                if (type == "WordPiece")
                    return std::make_shared<WordPieceDecoder>(s.value("prefix", "##"), s.value("cleanup", true));
                if (type == "Metaspace")
                    return std::make_shared<MetaspaceDecoder>(s.value("replacement", "\u2581"), s.value("add_prefix_space", true));
                return nullptr;
            };
            if (j["decoder"].value("type", "") == "Sequence") {
                std::vector<std::shared_ptr<Decoder>> decs;
                for (const auto& s : j["decoder"]["decoders"]) {
                    auto d = create_dec(s);
                    if (d)
                        decs.push_back(d);
                }
                this->decoder_ = std::make_shared<SequenceDecoder>(decs);
            } else {
                this->decoder_ = create_dec(j["decoder"]);
            }
        }
        if (!this->decoder_) {
            // Default decoder if none specified
            this->decoder_ = std::make_shared<ByteLevelDecoder>();
        }
        if (j.contains("pre_tokenizer") && !j["pre_tokenizer"].is_null()) {
            auto pt        = j["pre_tokenizer"];
            auto create_pt = [&](const json& s) -> std::shared_ptr<PreTokenizer> {
                std::string type = s.value("type", "");
                if (type == "Split") {
                    std::string p;
                    if (s.contains("pattern")) {
                        if (s["pattern"].is_object())
                            p = s["pattern"].value("Regex", "");
                        else if (s["pattern"].is_string())
                            p = s["pattern"].get<std::string>();
                    }
                    if (!p.empty())
                        return std::make_shared<SplitPreTokenizer>(p, s.value("invert", false), s.value("behavior", "Isolated"));
                } else if (type == "Metaspace") {
                    return std::make_shared<MetaspacePreTokenizer>(s.value("str_rep", "▁"), s.value("add_prefix_space", true));
                } else if (type == "ByteLevel") {
                    return std::make_shared<ByteLevelPreTokenizer>(s.value("use_regex", true));
                } else if (type == "Digits") {
                    return std::make_shared<DigitsPreTokenizer>(s.value("individual_digits", false));
                } else if (type == "BertPreTokenizer") {
                    return std::make_shared<BertPreTokenizer>();
                } else if (type == "WhitespaceSplit") {
                    // T5/Albert use WhitespaceSplit but it seems to break Unigram's ability to see spaces.
                    // Ignoring it allows Metaspace to run on the full string, producing "_Hello_World",
                    // which Unigram then likely segments correctly.
                    return std::make_shared<SplitPreTokenizer>("\\s+", false, "Removed");
                }
                return nullptr;
            };
            if (pt.value("type", "") == "Sequence" && pt.contains("pretokenizers")) {
                std::vector<std::shared_ptr<PreTokenizer>> pts;
                for (const auto& s : pt["pretokenizers"]) {
                    auto p = create_pt(s);
                    if (p)
                        pts.push_back(p);
                }
                this->pre_tokenizer_ = std::make_shared<SequencePreTokenizer>(pts);
            } else {
                this->pre_tokenizer_ = create_pt(pt);
            }
        }
        if (j.contains("post_processor") && !j["post_processor"].is_null()) {
            auto pp  = j["post_processor"];
            auto ptl = [&](const json& s) {
                std::vector<TemplateProcessing::Step> steps;
                if (s.contains("single")) {
                    for (const auto& i : s["single"]) {
                        if (i.contains("SpecialToken"))
                            steps.push_back({true, public_api->token_to_id(i["SpecialToken"]["id"].get<std::string>())});
                        else if (i.contains("Sequence"))
                            steps.push_back({false, 0});
                    }
                    this->post_processor_ = std::make_shared<TemplateProcessing>(steps);
                }
            };
            if (pp.value("type", "") == "TemplateProcessing")
                ptl(pp);
            else if (pp.value("type", "") == "Sequence" && pp.contains("processors")) {
                for (const auto& s : pp["processors"])
                    if (s.value("type", "") == "TemplateProcessing") {
                        ptl(s);
                        break;
                    }
            }
        }
        if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
            std::vector<std::string> cs;
            for (const auto& item : j["added_tokens"]) {
                std::string c   = item.value("content", "");
                int id          = item.value("id", -1);
                bool special    = item.value("special", false);
                bool lstrip     = item.value("lstrip", false);
                bool rstrip     = item.value("rstrip", false);
                bool normalized = item.value("normalized", false);
                if (c.empty() || id == -1)
                    continue;
                cs.push_back(c);
                this->added_tokens_.push_back({id, c, special, lstrip, rstrip, normalized});  // Store added token info
                if (c == "[PAD]" || c == "<pad>")
                    this->special_tokens_.pad = id;
                if (c == "[BOS]" || c == "<s>" || c == "<bos>")
                    this->special_tokens_.bos = id;
                if (c == "[EOS]" || c == "</s>" || c == "<eos>")
                    this->special_tokens_.eos = id;
                if (c == "[UNK]" || c == "<unk>")
                    this->special_tokens_.unk = id;
                if (c == "[M]" || c == "<M>")
                    this->special_tokens_.mask = id;
                if (c == "[NOISE]" || c == "<noise>")
                    this->special_tokens_.noise = id;

                auto bpe = std::dynamic_pointer_cast<BPEModel>(this->model_);
                if (bpe) {
                    bpe->vocab_[c]        = id;
                    bpe->id_to_token_[id] = c;
                }
            }
            if (!cs.empty()) {
                std::sort(cs.begin(), cs.end(), [](const std::string& a, const std::string& b) { return a.length() > b.length(); });
                std::string p;
                for (size_t i = 0; i < cs.size(); ++i) {
                    if (i > 0)
                        p += "|";
                    p += OnigurumaRegexEscape(cs[i]);
                }
                this->added_tokens_regex_ = std::make_shared<ClsOnigRegex>(p);
            }
        }
        if (j.contains("config_overrides")) {  //  from "/tokenizer_config.json"
            auto co = j["config_overrides"];
            if (co.contains("bos_token"))
                this->special_tokens_.bos = public_api->token_to_id(get_token_content(co["bos_token"]));
            if (co.contains("eos_token"))
                this->special_tokens_.eos = public_api->token_to_id(get_token_content(co["eos_token"]));
            if (co.contains("pad_token"))
                this->special_tokens_.pad = public_api->token_to_id(get_token_content(co["pad_token"]));
            if (co.contains("unk_token"))
                this->special_tokens_.unk = public_api->token_to_id(get_token_content(co["unk_token"]));
        }
        return true;
    }
};

// ==========================================
// HF_Tokenizer Public API Implementation
// ==========================================

HF_Tokenizer::HF_Tokenizer() : impl_(std::unique_ptr<Impl>(new Impl())) {}
HF_Tokenizer::~HF_Tokenizer() = default;

HF_Tokenizer::HF_Tokenizer(Fish* dolphin, int flag) : impl_(std::make_unique<Impl>()) {
    config      = dolphin->config;
    name        = "HF_Tokenizer";
    config      = dolphin->config;
    bool isInit = false;
    if (config.model.isLoadCard()) {
    }
    string sTokenJsonPath = dolphin->config.model.sTokenJsonPath;
    isInit                = InitHF(dolphin, flag);
    if (!isInit) {
        _WARN("[HF_Tokenizer] failed to load \"%s\"!\n", sTokenJsonPath.c_str());
        vocab.clear();
    }
    if (dolphin->isModel({NLP_SCORE_})) {
    }
}

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
bool HF_Tokenizer::InitHF(Fish* dolphin, int flag) {
    assert(dolphin != nullptr);
    string sTokenJsonPath = dolphin->config.model.sTokenJsonPath, sNull = "";
    assert(nVocab() == 0);
    try {
        std::ifstream f(sTokenJsonPath);
        if (!f.is_open())
            return false;
        std::stringstream ss_j;
        ss_j << f.rdbuf();
        json j = json::parse(ss_j.str());
        if (j.is_null())
            return false;
        if (dolphin->isModel({NLP_SCORE_})) {
            int _noise_id = 151666;
            json jNoise   = json::parse(
                R"({
                "id": 151666,
                "content": "<noise>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
                })");
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
                    assert(id != _noise_id);
                }
                j["added_tokens"].push_back(jNoise);
            }
        }

        std::ifstream fc(dolphin->config.model.sTokenConfigPath);  // path + "/tokenizer_config.json"
        bool clean_up_spaces = false;
        if (fc.is_open()) {
            std::stringstream ss_jc;
            ss_jc << fc.rdbuf();
            json jc = json::parse(ss_jc.str());
            if (jc.contains("chat_template"))
                set_chat_template(jc["chat_template"].get<std::string>());
            clean_up_spaces       = jc.value("clean_up_tokenization_spaces", false);
            j["config_overrides"] = jc;
        }
        if (!load_from_json_str(j.dump()))
            return false;

        set_clean_up_tokenization_spaces(clean_up_spaces);
        int nV = impl_->model_->GetVocabs(vocab, 0x0);
        if (nV == 0) {  // training stage may has no vocab tables
            _WARN("[HF_Tokenizer] vocab is empty! @\"%s\"\n", sTokenJsonPath.c_str());
            return false;
        }
        S.bos = bos_token_id();
        S.eos = eos_token_id();
        S.unk = unk_token_id();
        // sep_id  = jKV(jVocab, {sep_token}, sep_id);
        S.pad = pad_token_id();
        // cls_id  = jKV(jVocab, {cls_token}, cls_id);
        // mask_id = jKV(jVocab, {mask_token}, mask_id);

        _INFO("[HF_Tokenizer] nVocab=%d %s special=%ld @\"%s\"\n", nV, S.Dump(0x0).c_str(), special_tokens.size(), sTokenJsonPath.c_str());

        return true;
    } catch (JSON::parse_error& ex) {
        _INFO("[Tokenizer] Failed @%s! ERR=%s \n", sTokenJsonPath.c_str(), ex.what());
        return false;
        // std::cerr << "parse error at byte " << ex.byte << std::endl;
    } catch (...) {
        _INFO("[Tokenizer] Failed @%s!\n", sTokenJsonPath.c_str());
        assert(0);
        return false;
    }
};

// int HF_Tokenizer::nVocab(int flag) const {
//     if (impl_ == nullptr)
//         return 0x0; //config.model.pad_vocab_size;
//     if (impl_->model_ == nullptr)
//         return 0x0; //config.model.pad_vocab_size;

//     auto nV = impl_->model_->vocab_size();
//     return nV;
// }

std::string HF_Tokenizer::T2STR(TOKEN_ID tok, int flag) const {
    /*string sA = id_to_token({(int)tok});
    if (impl_->decoder_) {
        std::vector<std::string> tokens({sA});
        impl_->decoder_->decode(tokens);
        sA = tokens[0];
    }*/

    string sB = GTokenizer::T2STR(tok);  // utf8proc_iterate
    return sB;
}

std::vector<int> HF_Tokenizer::encode(const std::string& text, bool add_special_tokens) const { return impl_->encode(this, text, add_special_tokens); }

std::string HF_Tokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    if (impl_ == nullptr || impl_->model_ == nullptr)
        return "";

    std::vector<std::string> tokens;
    for (int id : ids) {
        if (skip_special_tokens) {
            // Check if special token
            bool special = false;
            for (const auto& at : impl_->added_tokens_) {
                if (at.id == id && at.special) {
                    special = true;
                    break;
                }
            }
            if (special)
                continue;
        }
        std::string t = impl_->model_->id_to_token(id);
        if (!t.empty())
            tokens.push_back(t);
    }
    if (impl_->decoder_)
        impl_->decoder_->decode(tokens);
    std::string out;
    for (const auto& t : tokens) out += t;
    return out;
}

int HF_Tokenizer::token_to_id(const std::string& t) const { return impl_->model_ ? impl_->model_->token_to_id(t) : -1; }
std::string HF_Tokenizer::id_to_token(int id) const { return impl_->model_ ? impl_->model_->id_to_token(id) : ""; }
int HF_Tokenizer::pad_token_id() const { return impl_->special_tokens_.pad; }
int HF_Tokenizer::bos_token_id() const { return impl_->special_tokens_.bos; }
int HF_Tokenizer::eos_token_id() const { return impl_->special_tokens_.eos; }
int HF_Tokenizer::unk_token_id() const { return impl_->special_tokens_.unk; }

void HF_Tokenizer::set_chat_template(const std::string& t) {
    impl_->chat_template_  = t;
    impl_->jinja_template_ = std::make_shared<jinja::Template>(t);
}
std::string HF_Tokenizer::apply_chat_template(const ChatMessages& msgs, bool add_gen) const {
    if (!impl_->jinja_template_)
        return "";
    json j_msgs = json::array();
    for (const auto& m : msgs) j_msgs.push_back({{"role", m.first}, {"content", m.second}});
    json extra         = json::object();
    extra["bos_token"] = id_to_token(impl_->special_tokens_.bos);
    extra["eos_token"] = id_to_token(impl_->special_tokens_.eos);
    return impl_->jinja_template_->apply_chat_template(j_msgs, add_gen, json::array(), extra);
}

std::string HF_Tokenizer::apply_chat_template(const std::string& json_str, bool add_generation_prompt) const {
    if (!impl_->jinja_template_)
        return "";
    auto j_msgs = json::parse(json_str);
    if (!j_msgs.is_array())
        return "";
    json extra         = json::object();
    extra["bos_token"] = id_to_token(impl_->special_tokens_.bos);
    extra["eos_token"] = id_to_token(impl_->special_tokens_.eos);
    return impl_->jinja_template_->apply_chat_template(j_msgs, add_generation_prompt, json::array(), extra);
}

bool HF_Tokenizer::load_from_json_str(const std::string& json_str) {
    auto j = json::parse(json_str);
    if (j.is_null())
        return false;
    return impl_->load_from_json(this, j);
}

void HF_Tokenizer::set_clean_up_tokenization_spaces(bool clean) { impl_->set_clean_up_tokenization_spaces(clean); }

// ==========================================
// AutoTokenizer Implementation
// ==========================================

std::shared_ptr<HF_Tokenizer> AutoTokenizer::from_pretrained(const std::string& path) {
    auto tok = std::make_shared<HF_Tokenizer>();
    std::ifstream f(path + "/tokenizer.json");
    if (!f.is_open())
        return nullptr;
    std::stringstream ss_j;
    ss_j << f.rdbuf();
    json j = json::parse(ss_j.str());
    if (j.is_null())
        return nullptr;

    std::ifstream fc(path + "/tokenizer_config.json");
    bool clean_up_spaces = false;
    if (fc.is_open()) {
        std::stringstream ss_jc;
        ss_jc << fc.rdbuf();
        json jc = json::parse(ss_jc.str());
        if (jc.contains("chat_template"))
            tok->set_chat_template(jc["chat_template"].get<std::string>());
        clean_up_spaces       = jc.value("clean_up_tokenization_spaces", false);
        j["config_overrides"] = jc;
    }
    if (!tok->load_from_json_str(j.dump()))
        return nullptr;
    tok->set_clean_up_tokenization_spaces(clean_up_spaces);
    return tok;
}

//}  namespace tokenizer