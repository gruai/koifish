/* Copyright (c) 2013 Dropbox, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

#include "../GST_log.hpp"
#include "json12.hpp"
using JSON = json11::Json;
namespace json11 {

static const int max_depth = 200;

using std::initializer_list;
using std::make_shared;
// using std::map;
using std::move;
using std::string;
using std::vector;

/* Helper for representing null - just a do-nothing struct, plus comparison
 * operators so the helpers in JsonValue work. We can't use nullptr_t because
 * it may not be orderable.
 */
struct NullStruct {
    bool operator==(NullStruct) const { return true; }
    bool operator<(NullStruct) const { return false; }
};

/* * * * * * * * * * * * * * * * * * * *
 * Serialization
 */

static void dump(NullStruct, string& out) { out += "null"; }

static void dump(double value, string& out) {
    if (std::isfinite(value)) {
        char buf[32];
        snprintf(buf, sizeof buf, "%.17g", value);
        out += buf;
    } else {
        out += "null";
    }
}

static void dump(int value, string& out) {
    char buf[32];
    snprintf(buf, sizeof buf, "%d", value);
    out += buf;
}

static void dump(int64_t value, string& out) {
    char buf[32];
    snprintf(buf, sizeof buf, "%ld", value);
    out += buf;
}

static void dump(size_t value, string& out) {
    char buf[32];
    snprintf(buf, sizeof buf, "%ld", value);
    out += buf;
}

static void dump(bool value, string& out) { out += value ? "true" : "false"; }

static void dump(const string& value, string& out) {
    out += '"';
    for (size_t i = 0; i < value.length(); i++) {
        const char ch = value[i];
        if (ch == '\\') {
            out += "\\\\";
        } else if (ch == '"') {
            out += "\\\"";
        } else if (ch == '\b') {
            out += "\\b";
        } else if (ch == '\f') {
            out += "\\f";
        } else if (ch == '\n') {
            out += "\\n";
        } else if (ch == '\r') {
            out += "\\r";
        } else if (ch == '\t') {
            out += "\\t";
        } else if (static_cast<uint8_t>(ch) <= 0x1f) {
            char buf[8];
            snprintf(buf, sizeof buf, "\\u%04x", ch);
            out += buf;
        } else if (static_cast<uint8_t>(ch) == 0xe2 && static_cast<uint8_t>(value[i + 1]) == 0x80 && static_cast<uint8_t>(value[i + 2]) == 0xa8) {
            out += "\\u2028";
            i += 2;
        } else if (static_cast<uint8_t>(ch) == 0xe2 && static_cast<uint8_t>(value[i + 1]) == 0x80 && static_cast<uint8_t>(value[i + 2]) == 0xa9) {
            out += "\\u2029";
            i += 2;
        } else {
            out += ch;
        }
    }
    out += '"';
}

static void dump(const Json::array& values, string& out) {
    bool first = true;
    out += "[";
    for (const auto& value : values) {
        if (!first)
            out += ", ";
        value.dump(out);
        first = false;
    }
    out += "]";
}

static void dump(const Json::object& values, string& out) {
    bool first = true;
    out += "{";
    for (const auto& kv : values) {
        if (!first)
            out += ", ";
        dump(kv.first, out);
        out += ": ";
        kv.second.dump(out);
        first = false;
    }
    out += "}";
}

void Json::dump(string& out) const { m_ptr->dump(out); }

/* * * * * * * * * * * * * * * * * * * *
 * Value wrappers
 */

template <Json::Type tag, typename T>
class Value : public JsonValue {
   protected:
    // Constructors
    explicit Value(const T& value) : m_value(value) {}
    explicit Value(T&& value) : m_value(move(value)) {}

    // Get type tag
    Json::Type type() const override { return tag; }

    // Comparisons
    bool equals(const JsonValue* other) const override { return m_value == static_cast<const Value<tag, T>*>(other)->m_value; }
    bool less(const JsonValue* other) const override { return m_value < static_cast<const Value<tag, T>*>(other)->m_value; }

    T m_value;  // was: const T m_value;
    void dump(string& out) const override { json11::dump(m_value, out); }
};

class JsonDouble final : public Value<Json::NUMBER, double> {
    double number_value() const override { return m_value; }
    int int_value() const override { return static_cast<int>(m_value); }
    bool equals(const JsonValue* other) const override { return m_value == other->number_value(); }
    bool less(const JsonValue* other) const override { return m_value < other->number_value(); }

   public:
    explicit JsonDouble(double value) : Value(value) {}
};

class JsonInt final : public Value<Json::NUMBER, int> {
    double number_value() const override { return m_value; }
    int int_value() const override { return m_value; }
    bool equals(const JsonValue* other) const override { return m_value == other->number_value(); }
    bool less(const JsonValue* other) const override { return m_value < other->number_value(); }

   public:
    explicit JsonInt(int value) : Value(value) {}
};

class JsonSize final : public Value<Json::NUMBER, size_t> {
    size_t size_value() const override { return m_value; }
    int int_value() const override { return m_value; }
    bool equals(const JsonValue* other) const override { return m_value == other->size_value(); }
    bool less(const JsonValue* other) const override { return m_value < other->size_value(); }

   public:
    explicit JsonSize(size_t value) : Value(value) {}
};

class JsonInt64 final : public Value<Json::NUMBER, int64_t> {
    int64_t int64_value() const override { return m_value; }
    int int_value() const override { return m_value; }
    bool equals(const JsonValue* other) const override { return m_value == other->int64_value(); }
    bool less(const JsonValue* other) const override { return m_value < other->int64_value(); }

   public:
    explicit JsonInt64(int64_t value) : Value(value) {}
};

class JsonBoolean final : public Value<Json::BOOL, bool> {
    bool bool_value() const override { return m_value; }

   public:
    explicit JsonBoolean(bool value) : Value(value) {}
};

class JsonString final : public Value<Json::STRING, string> {
    const string& string_value() const override { return m_value; }

   public:
    explicit JsonString(const string& value) : Value(value) {}
    explicit JsonString(string&& value) : Value(move(value)) {}
};

class JsonArray final : public Value<Json::ARRAY, Json::array> {
    const Json::array& array_items() const override { return m_value; }
    const Json& operator[](size_t i) const override;

   public:
    explicit JsonArray(const Json::array& value) : Value(value) {}
    explicit JsonArray(Json::array&& value) : Value(move(value)) {}

    Json::array* mutable_array() override { return &m_value; }
};

class JsonObject final : public Value<Json::OBJECT, Json::object> {
    const Json::object& object_items() const override { return m_value; }
    const Json& operator[](const string& key) const override;

   public:
    explicit JsonObject(const Json::object& value) : Value(value) {}
    explicit JsonObject(Json::object&& value) : Value(std::move(value)) {}
    Json::object* mutable_object() override { return &m_value; }
};

class JsonNull final : public Value<Json::NUL, NullStruct> {
   public:
    JsonNull() : Value({}) {}
};

/* * * * * * * * * * * * * * * * * * * *
 * Static globals - static-init-safe
 */
struct Statics {
    const std::shared_ptr<JsonValue> null = make_shared<JsonNull>();
    const std::shared_ptr<JsonValue> t    = make_shared<JsonBoolean>(true);
    const std::shared_ptr<JsonValue> f    = make_shared<JsonBoolean>(false);
    const string empty_string;
    const vector<Json> empty_vector;
    const MY_MAP<string, Json> empty_map;
    Statics() {}
};

static const Statics& statics() {
    static const Statics s{};
    return s;
}

static const Json& static_null() {
    // This has to be separate, not in Statics, because Json() accesses statics().null.
    static const Json json_null;
    return json_null;
}

/* * * * * * * * * * * * * * * * * * * *
 * Constructors
 */

Json::Json() noexcept : m_ptr(statics().null) {}
Json::Json(std::nullptr_t) noexcept : m_ptr(statics().null) {}
Json::Json(double value) : m_ptr(make_shared<JsonDouble>(value)) {}
Json::Json(int value) : m_ptr(make_shared<JsonInt>(value)) {}
Json::Json(uint32_t value) : m_ptr(make_shared<JsonInt>(value)) {}
Json::Json(int64_t value) : m_ptr(make_shared<JsonInt64>(value)) {}
Json::Json(size_t value) : m_ptr(make_shared<JsonSize>(value)) {}
Json::Json(bool value) : m_ptr(value ? statics().t : statics().f) {}
Json::Json(const string& value) : m_ptr(make_shared<JsonString>(value)) {}
Json::Json(string&& value) : m_ptr(make_shared<JsonString>(move(value))) {}
Json::Json(const char* value) : m_ptr(make_shared<JsonString>(value)) {}
Json::Json(const Json::array& values) : m_ptr(make_shared<JsonArray>(values)) {}
Json::Json(Json::array&& values) : m_ptr(make_shared<JsonArray>(move(values))) {}
Json::Json(const Json::object& values) : m_ptr(make_shared<JsonObject>(values)) {}
Json::Json(Json::object&& values) : m_ptr(make_shared<JsonObject>(std::move(values))) {}

/* * * * * * * * * * * * * * * * * * * *
 * Accessors
 */

Json::Type Json::type() const { return m_ptr->type(); }
double Json::number_value() const { return m_ptr->number_value(); }
int Json::int_value() const { return m_ptr->int_value(); }
bool Json::bool_value() const { return m_ptr->bool_value(); }
const string& Json::string_value() const { return m_ptr->string_value(); }
const vector<Json>& Json::array_items() const { return m_ptr->array_items(); }
const MY_MAP<string, Json>& Json::object_items() const { return m_ptr->object_items(); }
const Json& Json::operator[](size_t i) const { return (*m_ptr)[i]; }
const Json& Json::operator[](const string& key) const { return (*m_ptr)[key]; }

double JsonValue::number_value() const { return 0; }
size_t JsonValue::size_value() const { return 0; }
int64_t JsonValue::int64_value() const { return 0; }
int JsonValue::int_value() const { return 0; }
bool JsonValue::bool_value() const { return false; }
const string& JsonValue::string_value() const { return statics().empty_string; }
const vector<Json>& JsonValue::array_items() const { return statics().empty_vector; }
const MY_MAP<string, Json>& JsonValue::object_items() const { return statics().empty_map; }
const Json& JsonValue::operator[](size_t) const { return static_null(); }
const Json& JsonValue::operator[](const string&) const { return static_null(); }

const Json& JsonObject::operator[](const string& key) const {
    auto iter = m_value.find(key);
    return (iter == m_value.end()) ? static_null() : iter->second;
}
const Json& JsonArray::operator[](size_t i) const {
    if (i >= m_value.size())
        return static_null();
    else
        return m_value[i];
}

/* * * * * * * * * * * * * * * * * * * *
 * Comparison
 */

bool Json::operator==(const Json& other) const {
    if (m_ptr == other.m_ptr)
        return true;
    if (m_ptr->type() != other.m_ptr->type())
        return false;

    return m_ptr->equals(other.m_ptr.get());
}

bool Json::operator<(const Json& other) const {
    if (m_ptr == other.m_ptr)
        return false;
    if (m_ptr->type() != other.m_ptr->type())
        return m_ptr->type() < other.m_ptr->type();

    return m_ptr->less(other.m_ptr.get());
}

/* * * * * * * * * * * * * * * * * * * *
 * Parsing
 */

/* esc(c)
 *
 * Format char c suitable for printing in an error message.
 */
static inline string esc(char c) {
    char buf[12];
    if (static_cast<uint8_t>(c) >= 0x20 && static_cast<uint8_t>(c) <= 0x7f) {
        snprintf(buf, sizeof buf, "'%c' (%d)", c, c);
    } else {
        snprintf(buf, sizeof buf, "(%d)", c);
    }
    return string(buf);
}

static inline bool in_range(long x, long lower, long upper) { return (x >= lower && x <= upper); }

namespace {
/* JsonParser
 *
 * Object that tracks all state of an in-progress parse.
 */
struct JsonParser final {
    /* State
     */
    const string& str;
    size_t i;
    string& err;
    bool failed;
    const JsonParse strategy;

    /* fail(msg, err_ret = Json())
     *
     * Mark this parse as failed.
     */
    Json fail(string&& msg) { return fail(move(msg), Json()); }

    template <typename T>
    T fail(string&& msg, const T err_ret) {
        if (!failed)
            err = std::move(msg);
        failed = true;
        return err_ret;
    }

    /* consume_whitespace()
     *
     * Advance until the current character is non-whitespace.
     */
    void consume_whitespace() {
        while (str[i] == ' ' || str[i] == '\r' || str[i] == '\n' || str[i] == '\t') i++;
    }

    /* consume_comment()
     *
     * Advance comments (c-style inline and multiline).
     */
    bool consume_comment() {
        bool comment_found = false;
        if (str[i] == '/') {
            i++;
            if (i == str.size())
                return fail("unexpected end of input after start of comment", false);
            if (str[i] == '/') {  // inline comment
                i++;
                // advance until next line, or end of input
                while (i < str.size() && str[i] != '\n') {
                    i++;
                }
                comment_found = true;
            } else if (str[i] == '*') {  // multiline comment
                i++;
                if (i > str.size() - 2)
                    return fail("unexpected end of input inside multi-line comment", false);
                // advance until closing tokens
                while (!(str[i] == '*' && str[i + 1] == '/')) {
                    i++;
                    if (i > str.size() - 2)
                        return fail("unexpected end of input inside multi-line comment", false);
                }
                i += 2;
                comment_found = true;
            } else
                return fail("malformed comment", false);
        }
        return comment_found;
    }

    /* consume_garbage()
     *
     * Advance until the current character is non-whitespace and non-comment.
     */
    void consume_garbage() {
        consume_whitespace();
        if (strategy == JsonParse::COMMENTS) {
            bool comment_found = false;
            do {
                comment_found = consume_comment();
                if (failed)
                    return;
                consume_whitespace();
            } while (comment_found);
        }
    }

    /* get_next_token()
     *
     * Return the next non-whitespace character. If the end of the input is reached,
     * flag an error and return 0.
     */
    char get_next_token() {
        consume_garbage();
        if (failed)
            return static_cast<char>(0);
        if (i == str.size())
            return fail("unexpected end of input", static_cast<char>(0));

        return str[i++];
    }

    /* encode_utf8(pt, out)
     *
     * Encode pt as UTF-8 and add it to out.
     */
    void encode_utf8(long pt, string& out) {
        if (pt < 0)
            return;

        if (pt < 0x80) {
            out += static_cast<char>(pt);
        } else if (pt < 0x800) {
            out += static_cast<char>((pt >> 6) | 0xC0);
            out += static_cast<char>((pt & 0x3F) | 0x80);
        } else if (pt < 0x10000) {
            out += static_cast<char>((pt >> 12) | 0xE0);
            out += static_cast<char>(((pt >> 6) & 0x3F) | 0x80);
            out += static_cast<char>((pt & 0x3F) | 0x80);
        } else {
            out += static_cast<char>((pt >> 18) | 0xF0);
            out += static_cast<char>(((pt >> 12) & 0x3F) | 0x80);
            out += static_cast<char>(((pt >> 6) & 0x3F) | 0x80);
            out += static_cast<char>((pt & 0x3F) | 0x80);
        }
    }

    /* parse_string()
     *
     * Parse a string, starting at the current position.
     */
    string parse_string() {
        string out;
        long last_escaped_codepoint = -1;
        while (true) {
            if (i == str.size())
                return fail("unexpected end of input in string", "");

            char ch = str[i++];

            if (ch == '"') {
                encode_utf8(last_escaped_codepoint, out);
                return out;
            }

            if (in_range(ch, 0, 0x1f))
                return fail("unescaped " + esc(ch) + " in string", "");

            // The usual case: non-escaped characters
            if (ch != '\\') {
                encode_utf8(last_escaped_codepoint, out);
                last_escaped_codepoint = -1;
                out += ch;
                continue;
            }

            // Handle escapes
            if (i == str.size())
                return fail("unexpected end of input in string", "");

            ch = str[i++];

            if (ch == 'u') {
                // Extract 4-byte escape sequence
                string esc = str.substr(i, 4);
                // Explicitly check length of the substring. The following loop
                // relies on std::string returning the terminating NUL when
                // accessing str[length]. Checking here reduces brittleness.
                if (esc.length() < 4) {
                    return fail("bad \\u escape: " + esc, "");
                }
                for (size_t j = 0; j < 4; j++) {
                    if (!in_range(esc[j], 'a', 'f') && !in_range(esc[j], 'A', 'F') && !in_range(esc[j], '0', '9'))
                        return fail("bad \\u escape: " + esc, "");
                }

                long codepoint = strtol(esc.data(), nullptr, 16);

                // JSON specifies that characters outside the BMP shall be encoded as a pair
                // of 4-hex-digit \u escapes encoding their surrogate pair components. Check
                // whether we're in the middle of such a beast: the previous codepoint was an
                // escaped lead (high) surrogate, and this is a trail (low) surrogate.
                if (in_range(last_escaped_codepoint, 0xD800, 0xDBFF) && in_range(codepoint, 0xDC00, 0xDFFF)) {
                    // Reassemble the two surrogate pairs into one astral-plane character, per
                    // the UTF-16 algorithm.
                    encode_utf8((((last_escaped_codepoint - 0xD800) << 10) | (codepoint - 0xDC00)) + 0x10000, out);
                    last_escaped_codepoint = -1;
                } else {
                    encode_utf8(last_escaped_codepoint, out);
                    last_escaped_codepoint = codepoint;
                }

                i += 4;
                continue;
            }

            encode_utf8(last_escaped_codepoint, out);
            last_escaped_codepoint = -1;

            if (ch == 'b') {
                out += '\b';
            } else if (ch == 'f') {
                out += '\f';
            } else if (ch == 'n') {
                out += '\n';
            } else if (ch == 'r') {
                out += '\r';
            } else if (ch == 't') {
                out += '\t';
            } else if (ch == '"' || ch == '\\' || ch == '/') {
                out += ch;
            } else {
                return fail("invalid escape character " + esc(ch), "");
            }
        }
    }

    /* parse_number()
     *
     * Parse a double.
     */
    Json parse_number() {
        size_t start_pos = i;

        if (str[i] == '-')
            i++;

        // Integer part
        if (str[i] == '0') {
            i++;
            if (in_range(str[i], '0', '9'))
                return fail("leading 0s not permitted in numbers");
        } else if (in_range(str[i], '1', '9')) {
            i++;
            while (in_range(str[i], '0', '9')) i++;
        } else {
            return fail("invalid " + esc(str[i]) + " in number");
        }

        if (str[i] != '.' && str[i] != 'e' && str[i] != 'E' && (i - start_pos) <= static_cast<size_t>(std::numeric_limits<int>::digits10)) {
            return std::atoi(str.c_str() + start_pos);
        }

        // Decimal part
        if (str[i] == '.') {
            i++;
            if (!in_range(str[i], '0', '9'))
                return fail("at least one digit required in fractional part");

            while (in_range(str[i], '0', '9')) i++;
        }

        // Exponent part
        if (str[i] == 'e' || str[i] == 'E') {
            i++;

            if (str[i] == '+' || str[i] == '-')
                i++;

            if (!in_range(str[i], '0', '9'))
                return fail("at least one digit required in exponent");

            while (in_range(str[i], '0', '9')) i++;
        }

        return std::strtod(str.c_str() + start_pos, nullptr);
    }

    /* expect(str, res)
     *
     * Expect that 'str' starts at the character that was just read. If it does, advance
     * the input and return res. If not, flag an error.
     */
    Json expect(const string& expected, Json res) {
        assert(i != 0);
        i--;
        if (str.compare(i, expected.length(), expected) == 0) {
            i += expected.length();
            return res;
        } else {
            return fail("parse error: expected " + expected + ", got " + str.substr(i, expected.length()));
        }
    }

    /* parse_json()
     *
     * Parse a JSON object.
     */
    Json parse_json(int depth) {
        if (depth > max_depth) {
            return fail("exceeded maximum nesting depth");
        }

        char ch = get_next_token();
        if (failed)
            return Json();

        if (ch == '-' || (ch >= '0' && ch <= '9')) {
            i--;
            return parse_number();
        }

        if (ch == 't')
            return expect("true", true);

        if (ch == 'f')
            return expect("false", false);

        if (ch == 'n')
            return expect("null", Json());

        if (ch == '"')
            return parse_string();

        if (ch == '{') {
            MY_MAP<string, Json> data;
            ch = get_next_token();
            if (ch == '}')
                return data;

            while (1) {
                if (ch != '"')
                    return fail("expected '\"' in object, got " + esc(ch));

                string key = parse_string();
                if (failed)
                    return Json();

                ch = get_next_token();
                if (ch != ':')
                    return fail("expected ':' in object, got " + esc(ch));

                data[std::move(key)] = parse_json(depth + 1);
                if (failed)
                    return Json();

                ch = get_next_token();
                if (ch == '}')
                    break;
                if (ch != ',')
                    return fail("expected ',' in object, got " + esc(ch));

                ch = get_next_token();
            }
            return data;
        }

        if (ch == '[') {
            vector<Json> data;
            ch = get_next_token();
            if (ch == ']')
                return data;

            while (1) {
                i--;
                data.push_back(parse_json(depth + 1));
                if (failed)
                    return Json();

                ch = get_next_token();
                if (ch == ']')
                    break;
                if (ch != ',')
                    return fail("expected ',' in list, got " + esc(ch));

                ch = get_next_token();
                (void)ch;
            }
            return data;
        }

        return fail("expected value, got " + esc(ch));
    }
};
}  // namespace

Json Json::parse(const string& in, string& err, JsonParse strategy) {
    JsonParser parser{in, 0, err, false, strategy};
    Json result = parser.parse_json(0);

    // Check for any trailing garbage
    parser.consume_garbage();
    if (parser.failed) {
        _ERROR("[JSON] failed to parse \"%s\"", in.c_str());
        return Json();
    }

    if (parser.i != in.size()) {
        return parser.fail("unexpected trailing " + esc(in[parser.i]));
    }

    return result;
}

Json Json::load(const std::string& path, int flag) {
    std::string err;
    try {
        std::ifstream f(path, std::ios::in | std::ios::binary);
        if (!f.is_open()) {
            _ERROR("[JSON] cannot open file: @%s", path.c_str());
            return Json();
        }
        // Optional: check size limit? Probably not
        std::stringstream ss_j;
        ss_j << f.rdbuf();
        if (f.bad()) {
            _ERROR("[JSON] I/O error reading file: @%s", path.c_str());
            return Json();
        }
        if (f.fail() && !f.eof()) {
            _ERROR("[JSON] failed to read file: @%s", path.c_str());
            return Json();
        }
        std::string parse_err;
        Json j = Json::parse(ss_j.str(), parse_err);
        if (!parse_err.empty()) {
            _ERROR("[JSON] failed to parse file: @%s. err=%s", path.c_str(), parse_err.c_str());
            return Json();
        }
        // Note: literal `null` is valid JSON - is_null() may be legit
        // Only warn if you consider null an error
        if (j.is_null()) {
            _WARN("[JSON] is null @%s", path.c_str());
        }
        return j;
    } catch (const std::exception& e) {
        _ERROR("[JSON] exception while loading %s. exception = %s", path.c_str(), e.what());
        return Json();
    } catch (...) {
        _ERROR("[JSON] unknown exception while loading %s", path.c_str());
        return Json();
    }
}

// Documented in json12.hpp
vector<Json> Json::parse_multi(const string& in, std::string::size_type& parser_stop_pos, string& err, JsonParse strategy) {
    JsonParser parser{in, 0, err, false, strategy};
    parser_stop_pos = 0;
    vector<Json> json_vec;
    while (parser.i != in.size() && !parser.failed) {
        json_vec.push_back(parser.parse_json(0));
        if (parser.failed)
            break;

        // Check for another object
        parser.consume_garbage();
        if (parser.failed)
            break;
        parser_stop_pos = parser.i;
    }
    return json_vec;
}

/* * * * * * * * * * * * * * * * * * * *
 * Shape-checking
 */

bool Json::has_shape(const shape& types, string& err) const {
    if (!is_object()) {
        err = "expected JSON object, got " + dump();
        return false;
    }

    const auto& obj_items = object_items();
    for (auto& item : types) {
        const auto it = obj_items.find(item.first);
        if (it == obj_items.cend() || it->second.type() != item.second) {
            err = "bad type for " + item.first + " in " + dump();
            return false;
        }
    }

    return true;
}

Json& Json::operator[](const std::string& key) {
    if (type() == NUL) {
        *this = Json(Json::object{});  // null -> empty object
    } else if (type() == OBJECT && m_ptr.use_count() > 1) {
        *this = Json(Json::object(object_items()));  // copy-on-write
    }
    Json::object* m = m_ptr->mutable_object();
    if (!m) {
        throw std::logic_error("json11: operator[](string) on a non-object Json");
    }
    return (*m)[key];
}

std::size_t Json::erase(const std::string& key) {
    if (type() != OBJECT) {
        return 0;  // nothing to erase
    }
    if (object_items().find(key) == object_items().end()) {
        return 0;  // key absent, avoid a needless copy
    }
    if (m_ptr.use_count() > 1) {
        *this = Json(Json::object(object_items()));  // copy-on-write
    }
    Json::object* m = m_ptr->mutable_object();
    return m ? m->erase(key) : 0;
}

void Json::push_back(const Json& value) {
    if (type() == NUL) {
        *this = Json(Json::array{});  // null -> empty array
    } else if (type() == ARRAY && m_ptr.use_count() > 1) {
        *this = Json(Json::array(array_items()));  // copy-on-write
    }
    Json::array* a = m_ptr->mutable_array();
    if (!a) {
        throw std::logic_error("json11: push_back on a non-array Json");
    }
    a->push_back(value);
}

void Json::push_back(Json&& value) {
    Json tmp(std::move(value));  // keeps the const& path simple and safe for self-push
    push_back(static_cast<const Json&>(tmp));
}

void Json::clear() {
    switch (type()) {
        case OBJECT:
            if (m_ptr.use_count() > 1) {
                *this = Json(Json::object{});  // shared: just point at a fresh empty object
            } else {
                m_ptr->mutable_object()->clear();  // sole owner: clear in place
            }
            break;
        case ARRAY:
            if (m_ptr.use_count() > 1) {
                *this = Json(Json::array{});
            } else {
                m_ptr->mutable_array()->clear();
            }
            break;
        default:
            *this = Json();  // number/string/bool -> null
            break;
    }
}

}  // namespace json11

template <>
std::string J_get(const json11::Json& jObj) {
    std::string val;
    if (jObj.is_string()) {
        std::string s = jObj.string_value();
        val           = s;
    } else {
        NOT_IMPLEMENTED;
    }
    return val;
}
template <>
int J_get(const json11::Json& jObj) {
    int val;
    if (jObj.is_number()) {
        val = jObj.int_value();
    } else {
        NOT_IMPLEMENTED;
    }
    return val;
}
template <>
float J_get(const json11::Json& jObj) {
    float val = 0.0f;
    if (jObj.is_number()) {
        val = jObj.number_value();
    } else {
        NOT_IMPLEMENTED;
    }
    return val;
}
template <>
double J_get(const json11::Json& jObj) {
    double val = 0.0f;
    if (jObj.is_number()) {
        val = jObj.number_value();
    } else {
        NOT_IMPLEMENTED;
    }
    return val;
}
template <>
size_t J_get(const json11::Json& jObj) {
    size_t val = 0;
    if (jObj.is_number()) {
        val = jObj.number_value();
    } else {
        NOT_IMPLEMENTED;
    }
    return val;
}
template <>
bool J_get(const json11::Json& jObj) {
    if (jObj.is_bool()) {
        return jObj.bool_value();
    } else {
        NOT_IMPLEMENTED;
    }
    return false;
}

namespace msgpack_detail {

using Buf = std::vector<uint8_t>;

// ---------- encoding (big-endian, smallest representation) ----------
inline void put8(Buf& b, uint8_t v) { b.push_back(v); }
inline void put16(Buf& b, uint16_t v) {
    put8(b, uint8_t(v >> 8));
    put8(b, uint8_t(v));
}
inline void put32(Buf& b, uint32_t v) {
    put16(b, uint16_t(v >> 16));
    put16(b, uint16_t(v));
}
inline void put64(Buf& b, uint64_t v) {
    put32(b, uint32_t(v >> 32));
    put32(b, uint32_t(v));
}

inline void encode_int(Buf& b, int64_t v) {
    if (v >= 0) {
        uint64_t u = uint64_t(v);
        if (u <= 0x7f)
            put8(b, uint8_t(u));
        else if (u <= 0xff) {
            put8(b, 0xcc);
            put8(b, uint8_t(u));
        } else if (u <= 0xffff) {
            put8(b, 0xcd);
            put16(b, uint16_t(u));
        } else if (u <= 0xffffffff) {
            put8(b, 0xce);
            put32(b, uint32_t(u));
        } else {
            put8(b, 0xcf);
            put64(b, u);
        }
    } else {
        if (v >= -32)
            put8(b, uint8_t(v));  // negative fixint
        else if (v >= INT8_MIN) {
            put8(b, 0xd0);
            put8(b, uint8_t(v));
        } else if (v >= INT16_MIN) {
            put8(b, 0xd1);
            put16(b, uint16_t(v));
        } else if (v >= INT32_MIN) {
            put8(b, 0xd2);
            put32(b, uint32_t(v));
        } else {
            put8(b, 0xd3);
            put64(b, uint64_t(v));
        }
    }
}

inline void encode_len(Buf& b, size_t n, uint8_t fix_base, uint8_t fix_max, int8_t op16, int8_t op32, int8_t op8 /* -1 = none */) {
    if (n <= fix_max)
        put8(b, uint8_t(fix_base | n));
    else if (op8 >= 0 && n <= 0xff) {
        put8(b, uint8_t(op8));
        put8(b, uint8_t(n));
    } else if (n <= 0xffff) {
        put8(b, uint8_t(op16));
        put16(b, uint16_t(n));
    } else if (n <= 0xffffffffu) {
        put8(b, uint8_t(op32));
        put32(b, uint32_t(n));
    } else
        throw std::length_error("msgpack: container/string too large");
}

inline void encode(Buf& b, const JSON& j) {
    switch (j.type()) {
        case JSON::NUL:
            put8(b, 0xc0);
            break;
        case JSON::BOOL:
            put8(b, j.bool_value() ? 0xc3 : 0xc2);
            break;
        case JSON::NUMBER: {
            double d = j.number_value();
            // json11 does not distinguish int from double on the type level,
            // so integral values within int64 range are written as integers.
            if (d == std::floor(d) && !(d == 0 && std::signbit(d)) && d >= -9223372036854775808.0 && d < 9223372036854775808.0) {
                encode_int(b, int64_t(d));
            } else {
                uint64_t bits;
                std::memcpy(&bits, &d, 8);
                put8(b, 0xcb);
                put64(b, bits);
            }
            break;
        }
        case JSON::STRING: {
            const std::string& s = j.string_value();
            encode_len(b, s.size(), 0xa0, 31, 0xda, 0xdb, 0xd9);
            b.insert(b.end(), s.begin(), s.end());
            break;
        }
        case JSON::ARRAY: {
            const auto& a = j.array_items();
            encode_len(b, a.size(), 0x90, 15, 0xdc, 0xdd, -1);
            for (const auto& e : a) encode(b, e);
            break;
        }
        case JSON::OBJECT: {
            const auto& o = j.object_items();
            encode_len(b, o.size(), 0x80, 15, 0xde, 0xdf, -1);
            for (const auto& kv : o) {
                encode_len(b, kv.first.size(), 0xa0, 31, 0xda, 0xdb, 0xd9);
                b.insert(b.end(), kv.first.begin(), kv.first.end());
                encode(b, kv.second);
            }
            break;
        }
    }
}

// ---------- decoding (bounds-checked) ----------
struct Reader {
    const uint8_t* p;
    size_t n;
    size_t pos = 0;
    size_t remaining() const { return n - pos; }
    void need(size_t k) const {
        if (remaining() < k)
            throw std::runtime_error("msgpack: unexpected end of input");
    }
    uint8_t u8() {
        need(1);
        return p[pos++];
    }
    uint16_t u16() {
        uint16_t v = u8();
        return uint16_t(v << 8 | u8());
    }
    uint32_t u32() {
        uint32_t v = u16();
        return v << 16 | u16();
    }
    uint64_t u64() {
        uint64_t v = u32();
        return v << 32 | u32();
    }
    std::string bytes(size_t len) {
        need(len);
        std::string s(reinterpret_cast<const char*>(p + pos), len);
        pos += len;
        return s;
    }
};

inline JSON from_i64(int64_t v) {
    if (v >= std::numeric_limits<int>::min() && v <= std::numeric_limits<int>::max())
        return JSON(int(v));
    return JSON(double(v));  // json11 has no 64-bit integer type
}
inline JSON from_u64(uint64_t v) {
    if (v <= uint64_t(std::numeric_limits<int>::max()))
        return JSON(int(v));
    return JSON(double(v));
}

inline JSON decode(Reader& r, int depth);

inline JSON decode_array(Reader& r, size_t count, int depth) {
    if (count > r.remaining())
        throw std::runtime_error("msgpack: bad array length");
    JSON::array a;
    a.reserve(count);
    for (size_t i = 0; i < count; ++i) a.push_back(decode(r, depth + 1));
    return JSON(std::move(a));
}
inline JSON decode_map(Reader& r, size_t count, int depth) {
    if (count > r.remaining() / 2)
        throw std::runtime_error("msgpack: bad map length");
    JSON::object o;
    for (size_t i = 0; i < count; ++i) {
        JSON k = decode(r, depth + 1);
        if (!k.is_string())
            throw std::runtime_error("msgpack: map key is not a string");
        o[k.string_value()] = decode(r, depth + 1);  // last duplicate wins
    }
    return JSON(std::move(o));
}

inline JSON decode(Reader& r, int depth) {
    if (depth > 512)
        throw std::runtime_error("msgpack: nesting too deep");
    uint8_t t = r.u8();
    if (t <= 0x7f)
        return JSON(int(t));  // positive fixint
    if (t >= 0xe0)
        return JSON(int(int8_t(t)));  // negative fixint
    if ((t & 0xe0) == 0xa0)
        return JSON(r.bytes(t & 0x1f));  // fixstr
    if ((t & 0xf0) == 0x90)
        return decode_array(r, t & 0x0f, depth);
    if ((t & 0xf0) == 0x80)
        return decode_map(r, t & 0x0f, depth);
    switch (t) {
        case 0xc0:
            return JSON(nullptr);
        case 0xc2:
            return JSON(false);
        case 0xc3:
            return JSON(true);
        case 0xc4:
            return JSON(r.bytes(r.u8()));  // bin -> string (raw bytes)
        case 0xc5:
            return JSON(r.bytes(r.u16()));
        case 0xc6:
            return JSON(r.bytes(r.u32()));
        case 0xca: {
            uint32_t b = r.u32();
            float f;
            std::memcpy(&f, &b, 4);
            return JSON(double(f));
        }
        case 0xcb: {
            uint64_t b = r.u64();
            double d;
            std::memcpy(&d, &b, 8);
            return JSON(d);
        }
        case 0xcc:
            return from_u64(r.u8());
        case 0xcd:
            return from_u64(r.u16());
        case 0xce:
            return from_u64(r.u32());
        case 0xcf:
            return from_u64(r.u64());
        case 0xd0:
            return from_i64(int8_t(r.u8()));
        case 0xd1:
            return from_i64(int16_t(r.u16()));
        case 0xd2:
            return from_i64(int32_t(r.u32()));
        case 0xd3:
            return from_i64(int64_t(r.u64()));
        case 0xd9:
            return JSON(r.bytes(r.u8()));
        case 0xda:
            return JSON(r.bytes(r.u16()));
        case 0xdb:
            return JSON(r.bytes(r.u32()));
        case 0xdc:
            return decode_array(r, r.u16(), depth);
        case 0xdd:
            return decode_array(r, r.u32(), depth);
        case 0xde:
            return decode_map(r, r.u16(), depth);
        case 0xdf:
            return decode_map(r, r.u32(), depth);
        default:
            throw std::runtime_error("msgpack: unsupported type byte (ext/reserved)");
    }
}

}  // namespace msgpack_detail

std::vector<uint8_t> J_to_msgpack(const JSON& js) {
    msgpack_detail::Buf b;
    msgpack_detail::encode(b, js);
    return b;
}

JSON J_from_msgpack(const std::vector<uint8_t>& msgpack) {
    msgpack_detail::Reader r{msgpack.data(), msgpack.size()};
    JSON j = msgpack_detail::decode(r, 0);
    if (r.remaining() != 0)
        throw std::runtime_error("msgpack: trailing bytes");
    // if (!jPath.empty()) {  //  save json file with more info
    //     std::ofstream o(jPath);
    //     o << std::setw(4) << jsConfig << std::endl;
    // }
    return j;
}