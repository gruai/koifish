// strata_tokenizer.h - C++ port of tools/strata_tokenizer.py (the reference implementation of record).
//
// Byte-level BPE (GPT-2 byte alphabet) with the `qwen35` and `deepseek3` pre-tokenizers, type-3/type-4 special
// token matching, `plain` spans, and the incremental PromptEncoder.  It loads the pack's `tokenizer/`
// directory (vocab.json, merges.txt, token_type.json, tokenizer.json) that the Python `extract()` writes; it
// does not read GGUF files.
//
// Offsets: Python indexes `plain` spans and match marks in characters; here they are UTF-8 BYTE offsets into
// the text.  Every comparison the algorithm makes is order-based, so the results are the same ids.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata {

using TokenId = int32_t;

struct Span {
    size_t start = 0, end = 0;          // [start, end) in UTF-8 bytes
    bool operator==(const Span& o) const { return start == o.start && end == o.end; }
    bool operator<(const Span& o) const { return start != o.start ? start < o.start : end < o.end; }
};
using Spans = std::vector<Span>;

// The model names a pre-tokenizer type this build has no splitting rule for.
struct UnknownPreTokenizer : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

enum class PreKind { Qwen35, DeepSeek3 };

// "" -> qwen35, "qwen35" -> Qwen35, "deepseek*" -> DeepSeek3; anything else returns false.
bool pre_tokenizer_kind(const std::string& pre, PreKind& out);

// Python's bytes.decode("utf-8", errors="replace"): each maximal invalid subsequence becomes U+FFFD.
std::string utf8_replace_invalid(const std::string& bytes);

class Tokenizer {
public:
    // token_types may be empty (no special tokens).  Throws std::invalid_argument on a bad vocabulary or
    // merge list, UnknownPreTokenizer on an unknown `pre`.
    Tokenizer(std::vector<std::string> tokens, const std::vector<std::string>& merges,
              std::vector<int> token_types = {}, std::string pre = "qwen35",
              std::map<std::string, int> special_ids = {});

    // `path` is a pack directory containing tokenizer/, or the tokenizer/ directory itself.
    static std::unique_ptr<Tokenizer> from_pack(const std::string& path);

    // parse_special controls only the type-3 CONTROL literals; type-4 USER_DEFINED are matched either way.
    // `plain`: spans that are ordinary text even where they spell a literal.
    std::vector<TokenId> encode(std::string_view text, bool parse_special = false, const Spans& plain = {}) const;

    struct Marked {
        std::vector<TokenId> ids;
        std::vector<std::pair<size_t, size_t>> marks;   // (end offset in text, ids so far) after each special match
    };
    Marked encode_marked(std::string_view text, bool parse_special = false, const Spans& plain = {}) const;

    // The pre-tokenizer alone: the pieces BPE runs on (Python: Tokenizer._split).  Special tokens are not
    // looked for here.
    std::vector<std::string> pretokenize(std::string_view text) const;

    std::string token_bytes(TokenId id) const;                       // raw bytes of one token
    std::string decode_bytes(const std::vector<TokenId>& ids) const; // concatenated raw bytes
    std::string decode(const std::vector<TokenId>& ids) const;       // valid UTF-8, invalid bytes -> U+FFFD

    size_t vocab_size() const { return tokens_.size(); }
    size_t n_merges() const { return n_unique_merges_; }
    size_t max_special_len() const { return max_special_len_; }      // in bytes
    const std::string& pre() const { return pre_; }
    const std::map<std::string, int>& special_ids() const { return special_ids_; }
    const std::unordered_map<std::string, TokenId>& special_tokens() const { return special_tokens_; }
    const std::vector<std::string>& tokens() const { return tokens_; }

    static constexpr size_t kHeapMin = 64;           // words up to this many symbols use the plain scan
    static constexpr size_t kPieceCacheMax = 200000;

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

private:
    struct LiteralSet {
        struct Lit { std::string s; TokenId id; };
        std::array<std::vector<Lit>, 256> by_first;   // longest literal first within a bucket
        bool empty = true;
        void build(std::vector<std::pair<std::string, TokenId>> lits);
        bool find(std::string_view text, size_t from, size_t& start, size_t& len, TokenId& id) const;
    };

    void split(std::string_view segment, std::vector<std::string_view>& out) const;
    std::vector<TokenId> bpe(std::string_view piece) const;
    std::vector<int32_t> bpe_scan(std::vector<int32_t> parts) const;
    std::vector<int32_t> bpe_heap(std::vector<int32_t> parts) const;
    void encode_plain(std::string_view text, std::vector<TokenId>& out) const;
    std::vector<TokenId> encode_matching(std::string_view text, const LiteralSet& lits, const Spans& plain,
                                         std::vector<std::pair<size_t, size_t>>* marks) const;

    std::vector<std::string> tokens_;
    std::string pre_;
    PreKind kind_;
    std::vector<int> token_types_;
    std::map<std::string, int> special_ids_;
    std::unordered_map<std::string, TokenId> ids_;
    std::unordered_map<uint64_t, int32_t> ranks_;        // (left id << 32 | right id) -> rank
    std::vector<int32_t> merge_result_;                  // rank -> id of left+right, or -1 if not a vocab token
    size_t n_unique_merges_ = 0;
    std::array<int32_t, 256> byte_sym_;                  // byte -> id of its byte-alphabet character, or -1
    std::unordered_map<std::string, TokenId> special_tokens_;
    std::vector<std::string> control_tokens_;
    LiteralSet always_, special_;
    size_t max_special_len_ = 1;

    mutable std::mutex cache_mu_;
    mutable std::unordered_map<std::string, std::vector<TokenId>> piece_ids_;
};

// tok.encode(text, parse_special=True, plain=...) for chat prompts, reusing the ids of an earlier prompt up to a
// special-token boundary both share.  See the Python docstring for why that is exact.  Thread-safe.
class PromptEncoder {
public:
    explicit PromptEncoder(const Tokenizer& tok, size_t keep = 4) : tok_(tok), keep_(keep) {}

    std::vector<TokenId> encode(std::string_view text, const Spans& plain = {}, bool store = true);
    void forget(const std::vector<TokenId>& ids);       // the prompt with these ids was refused
    size_t last_reused() const { return last_reused_; } // bytes taken from an earlier prompt

private:
    struct Entry {
        std::string text;
        Spans plain;
        std::vector<TokenId> ids;
        std::vector<size_t> ends;     // mark end offsets
        std::vector<size_t> counts;   // ids so far at each mark
    };
    const Tokenizer& tok_;
    size_t keep_;
    std::mutex mu_;
    std::vector<std::shared_ptr<const Entry>> entries_;   // replaced, never changed in place
    std::atomic<size_t> last_reused_{0};
};

}  // namespace strata
