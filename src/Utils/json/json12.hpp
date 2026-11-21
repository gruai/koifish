/**
 * json12
 * 1. lose the “any Json can be shared freely across threads” guarantee to support
 *      a. "Json& operator[](const std::string& key)" function b. erase c. push_back
 * 2. std::map/unordered_map does not preserve insertion order!
 */
/* json11
 *
 * json11 is a tiny JSON library for C++11, providing JSON parsing and serialization.
 *
 * The core object provided by the library is json11::Json. A Json object represents any JSON
 * value: null, bool, number (int or double), string (std::string), array (std::vector), or
 * object (map_).
 *
 * Json objects act like values: they can be assigned, copied, moved, compared for equality or
 * order, etc. There are also helper methods Json::dump, to serialize a Json to a string, and
 * Json::parse (static) to parse a std::string as a Json object.
 *
 * Internally, the various types of Json object are represented by the JsonValue class
 * hierarchy.
 *
 * A note on numbers - JSON specifies the syntax of number formatting but not its semantics,
 * so some JSON implementations distinguish between integers and floating-point numbers, while
 * some don't. In json11, we choose the latter. Because some JSON implementations (namely
 * Javascript itself) treat all numbers as the same type, distinguishing the two leads
 * to JSON that will be *silently* changed by a round-trip through those implementations.
 * Dangerous! To avoid that risk, json11 stores all numbers as double internally, but also
 * provides integer helpers.
 *
 * Fortunately, double-precision IEEE754 ('double') can precisely store any integer in the
 * range +/-2^53, which includes every 'int' on most systems. (Timestamps often use int64
 * or long long to avoid the Y2038K problem; a double storing microseconds since some epoch
 * will be exact for +/- 275 years.)
 */

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

#pragma once

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
// #include <unordered_map>
#include "../GST_hashmap.hpp"
#define MY_MAP vector_map
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../../g_def_x.hpp"

#ifdef _MSC_VER
#if _MSC_VER <= 1800  // VS 2013
#ifndef noexcept
#define noexcept throw()
#endif

#ifndef snprintf
#define snprintf _snprintf_s
#endif
#endif
#endif

namespace json11 {
template <typename T>
struct is_std_map : std::false_type {};
template <typename K, typename V, typename C, typename A>
struct is_std_map<std::map<K, V, C, A>> : std::true_type {};

enum JsonParse { STANDARD, COMMENTS };

class JsonValue;

class Json final {
   public:
    // Types
    enum Type { NUL, NUMBER, BOOL, STRING, ARRAY, OBJECT };

    // Array and object typedefs
    typedef std::vector<Json> array;
    typedef MY_MAP<std::string, Json> object;

    // Constructors for the various types of JSON value.
    Json() noexcept;                 // NUL
    Json(std::nullptr_t) noexcept;   // NUL
    Json(double value);              // douoble NUMBER
    Json(int value);                 // int NUMBER
    Json(uint32_t value);            // uint32_t NUMBER
    Json(int64_t value);             // int64_t NUMBER
    Json(size_t value);              // size_t NUMBER
    Json(bool value);                // BOOL
    Json(const std::string& value);  // STRING
    Json(std::string&& value);       // STRING
    Json(const char* value);         // STRING
    Json(const array& values);       // ARRAY
    Json(array&& values);            // ARRAY
    Json(const object& values);      // OBJECT
    Json(object&& values);           // OBJECT
    Json(std::initializer_list<std::pair<const std::string, Json>> values) : Json(make_object(values)) {}

    // for Json j = {1, 2};
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
    Json& operator=(std::initializer_list<T> list) {
        array a;
        for (const T& v : list) a.push_back(Json(v));
        return *this = Json(std::move(a));
    }

    static Json parse(const std::string& in) {
        std::string err;
        return parse(in, err);
    }

    static Json load(const std::string& path, int flag = 0x0);

    // Implicit constructor: anything with a to_json() function.
    template <class T, class = decltype(&T::to_json)>
    Json(const T& t) : Json(t.to_json()) {}

    // Implicit constructor: map_-like objects
    template <class M, typename std::enable_if<std::is_constructible<std::string, decltype(std::declval<M>().begin()->first)>::value &&
                                                   std::is_constructible<Json, decltype(std::declval<M>().begin()->second)>::value,
                                               int>::type = 0>
    Json(const M& m) : Json(object(m.begin(), m.end())) {}

    // Implicit constructor: vector-like objects (std::list, std::vector, std::set, etc)
    template <class V, typename std::enable_if<std::is_constructible<Json, decltype(*std::declval<V>().begin())>::value, int>::type = 0>
    Json(const V& v) : Json(array(v.begin(), v.end())) {}

    // This prevents Json(some_pointer) from accidentally producing a bool. Use
    // Json(bool(some_pointer)) if that behavior is desired.
    Json(void*) = delete;

    // Accessors
    Type type() const;

    bool is_null() const { return type() == NUL; }
    bool is_number() const { return type() == NUMBER; }
    bool is_number_integer() const {
        if (!is_number())
            return false;
        double d = number_value();
        return d == std::floor(d);
    }
    bool is_number_float() const { return is_number() && !is_number_integer(); }
    bool is_bool() const { return type() == BOOL; }
    bool is_boolean() const { return is_bool(); }  // to support rapid_json like function
    bool is_string() const { return type() == STRING; }
    bool is_array() const { return type() == ARRAY; }
    bool is_object() const { return type() == OBJECT; }

    // The only place that maps C++ types to JSON types.
    // Returns false (and leaves `out` untouched) if the JSON type doesn't match.
    template <typename T>
    bool try_get(T& out) const {
        using U = std::remove_cv_t<std::remove_reference_t<T>>;

        if constexpr (std::is_same_v<U, Json>) {
            out = *this;
            return true;
        } else if constexpr (std::is_same_v<U, bool>) {  // must be before arithmetic
            if (!is_bool())
                return false;
            out = bool_value();
        } else if constexpr (std::is_arithmetic_v<U>) {  // int, long, float, double, ...
            if (!is_number())
                return false;
            out = static_cast<U>(number_value());
        } else if constexpr (std::is_same_v<U, std::string>) {
            if (!is_string())
                return false;
            out = string_value();
        } else if constexpr (std::is_same_v<U, array>) {
            if (!is_array())
                return false;
            out = array_items();
        } else if constexpr (std::is_same_v<U, object>) {
            if (!is_object())
                return false;
            out = object_items();
        } else if constexpr (is_std_map<U>::value && std::is_same<typename U::key_type, std::string>::value) {
            if (!is_object())
                return false;
            U tmp;
            for (const auto& kv : object_items()) {
                typename U::mapped_type x{};
                if (!kv.second.try_get(x))
                    return false;  // a value has the wrong type -> fail
                tmp.emplace(kv.first, std::move(x));
            }
            out = std::move(tmp);
        } else {
            static_assert(sizeof(U) == 0, "Json::try_get<T>: unsupported type");
        }
        return true;
    }

    // Value if the type matches, otherwise T{} (0, false, "", empty container), as before.
    template <typename T>
    T get() const {
        T result{};
        try_get(result);
        return result;
    }

    // Value of `key` if present, non-null and of the right type, otherwise default_value.
    template <typename T>
    T value(const std::string& key, const T& default_value) const {
        T result = default_value;
        if (is_object()) {
            const object& o = object_items();
            auto it         = o.find(key);
            if (it != o.end() && !it->second.is_null())
                it->second.try_get(result);
        }
        return result;
    }

    // So that item.value("name", "none") works with a string literal.
    std::string value(const std::string& key, const char* default_value) const { return value<std::string>(key, std::string(default_value)); }

    // Return the enclosed value if this is a number, 0 otherwise. Note that json11 does not
    // distinguish between integer and non-integer numbers - number_value() and int_value()
    // can both be applied to a NUMBER-typed object.
    double number_value() const;
    int int_value() const;

    // Return the enclosed value if this is a boolean, false otherwise.
    bool bool_value() const;
    // Return the enclosed string if this is a string, "" otherwise.
    const std::string& string_value() const;
    // Return the enclosed std::vector if this is an array, or an empty vector otherwise.
    const array& array_items() const;
    // Return the enclosed map_ if this is an object, or an empty map_ otherwise.
    const object& object_items() const;

    // ---- Iteration: yields Json values (not pairs) for both objects and arrays ----
    class const_iterator {
       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = Json;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const Json*;
        using reference         = const Json&;

        const_iterator() = default;
        explicit const_iterator(array::const_iterator it);
        explicit const_iterator(object::const_iterator it);

        const Json& operator*() const;
        const Json* operator->() const { return &**this; }
        const_iterator& operator++();
        const_iterator operator++(int);
        bool operator==(const const_iterator& o) const;
        bool operator!=(const const_iterator& o) const { return !(*this == o); }

        // Object key (only valid when iterating an object)
        const std::string& key() const;
        // The current element's value (works for objects and arrays)
        const Json& value() const { return **this; }

       private:
        array::const_iterator m_arr{};
        object::const_iterator m_obj{};
        bool m_isObj = false;
    };

    const_iterator begin() const;
    const_iterator end() const;

    const_iterator find(const std::string& key) const;

    // Key + value iteration over an object: for (const auto& [k, v] : j.items())
    struct items_range {
        object::const_iterator b, e;
        object::const_iterator begin() const { return b; }
        object::const_iterator end() const { return e; }
    };
    items_range items() const {
        const object& o = object_items();
        return items_range{o.begin(), o.end()};
    }

    inline size_t size() const {
        switch (type()) {
            case ARRAY:
                return array_items().size();
            case OBJECT:
                return object_items().size();
            default:
                return 0;
        }
    }

    // object::const_iterator find(const std::string& key) const { return object_items().find(key); }
    bool contains(const std::string& key) const {
        const object& items = object_items();
        return items.find(key) != items.end();
    }

    // Return a reference to arr[i] if this is an array, Json() otherwise.
    const Json& operator[](size_t i) const;
    // Return a reference to obj[key] if this is an object, Json() otherwise.
    const Json& operator[](const std::string& key) const;
    Json& operator[](const std::string& key);

    const Json& at(const std::string& key) const {
        if (!is_object()) {
            throw std::out_of_range("Json::at(\"" + key + "\"): not an object");
        }
        const object& o = object_items();
        auto it         = o.find(key);
        if (it == o.end()) {
            throw std::out_of_range("Json::at: key not found: " + key);
        }
        return it->second;
    }
    Json& at(const std::string& key) {
        if (!contains(key)) {
            throw std::out_of_range("Json::at: key not found: " + key);
        }
        return (*this)[key];
    }
    const Json& at(size_t i) const {
        if (!is_array()) {
            throw std::out_of_range("Json::at(index): not an array");
        }
        const array& a = array_items();
        if (i >= a.size()) {
            throw std::out_of_range("Json::at: index out of range: " + std::to_string(i));
        }
        return a[i];
    }

    // Serialize.
    void dump(std::string& out) const;
    std::string dump() const {
        std::string out;
        dump(out);
        return out;
    }
    std::string dump(int tab) const { return dump(); }

    // Parse. If parse fails, return Json() and assign an error message to err.
    static Json parse(const std::string& in, std::string& err, JsonParse strategy = JsonParse::STANDARD);
    static Json parse(const char* in, std::string& err, JsonParse strategy = JsonParse::STANDARD) {
        if (in) {
            return parse(std::string(in), err, strategy);
        } else {
            err = "null input";
            return nullptr;
        }
    }
    // Parse multiple objects, concatenated or separated by whitespace
    static std::vector<Json> parse_multi(const std::string& in, std::string::size_type& parser_stop_pos, std::string& err,
                                         JsonParse strategy = JsonParse::STANDARD);

    static inline std::vector<Json> parse_multi(const std::string& in, std::string& err, JsonParse strategy = JsonParse::STANDARD) {
        std::string::size_type parser_stop_pos;
        return parse_multi(in, parser_stop_pos, err, strategy);
    }

    bool operator==(const Json& rhs) const;
    bool operator<(const Json& rhs) const;
    bool operator!=(const Json& rhs) const { return !(*this == rhs); }
    bool operator<=(const Json& rhs) const { return !(rhs < *this); }
    bool operator>(const Json& rhs) const { return (rhs < *this); }
    bool operator>=(const Json& rhs) const { return !(*this < rhs); }

    // more functions add by cys
    std::ostream& operator>>(std::ostream& o) const {
        o << dump();
        return o;
    }
    friend std::ostream& operator<<(std::ostream& o, const Json& j) { return o << j.dump(); }
    friend std::istream& operator>>(std::istream& in, Json& json) {
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::string err;
        Json parsed = Json::parse(text, err);
        if (err.empty()) {
            json = std::move(parsed);
        } else {
            in.setstate(std::ios::failbit);
        }
        return in;
    }

    std::istream& read(std::istream& i) {
        std::string text((std::istreambuf_iterator<char>(i)), std::istreambuf_iterator<char>());
        std::string err;
        Json parsed = Json::parse(text, err);
        if (!err.empty()) {
            i.setstate(std::ios::failbit);
            return i;
        }
        *this = parsed;
        return i;
    }

    // operator form: j << std::cin;
    std::istream& operator<<(std::istream& i) { return read(i); }

    bool empty() const {
        switch (type()) {
            case NUL:
                return true;
            case ARRAY:
                return array_items().empty();
            case OBJECT:
                return object_items().empty();
            case STRING:
                return string_value().empty();
            default:
                return false;  // NUMBER, BOOL,
        }
    }

    /* has_shape(types, err)
     *
     * Return true if this is a JSON object and, for each item in types, has a field of
     * the given type. If not, return false and set err to a descriptive message.
     */
    typedef std::initializer_list<std::pair<std::string, Type>> shape;
    bool has_shape(const shape& types, std::string& err) const;

    void push_back(const Json& value);
    void push_back(Json&& value);

    void clear();
    std::size_t erase(const std::string& key);
    template <typename It>
    void update(It first, It last, bool merge_objects = false) {
        if (is_null()) {
            *this = Json(Json::object{});  // null -> empty object
        }
        for (; first != last; ++first) {
            const std::string& k = first->first;
            const Json& v        = first->second;
            if (merge_objects && contains(k) && v.is_object()) {
                Json& mine = (*this)[k];
                if (mine.is_object()) {
                    mine.update(v, true);  // recursive merge
                    continue;
                }
            }
            (*this)[k] = v;  // insert or overwrite
        }
    }

    // Whole-object version
    void update(const Json& j, bool merge_objects = false) {
        Json other      = j;  // cheap shared copy: safe for j.update(j)
        const object& o = other.object_items();
        update(o.begin(), o.end(), merge_objects);
    }

   private:
    std::shared_ptr<JsonValue> m_ptr;

    static object make_object(std::initializer_list<std::pair<const std::string, Json>> values) {
        object o;
        for (const auto& kv : values) o.emplace(kv.first, kv.second);
        return o;
    }
};

inline Json::const_iterator Json::find(const std::string& key) const {
    if (!is_object())
        return end();                                 // non-object: never found
    return const_iterator(object_items().find(key));  // object iterator, same kind as end()
}

inline Json::const_iterator::const_iterator(array::const_iterator it) : m_arr(it), m_isObj(false) {}

inline Json::const_iterator::const_iterator(object::const_iterator it) : m_obj(it), m_isObj(true) {}

inline const Json& Json::const_iterator::operator*() const { return m_isObj ? m_obj->second : *m_arr; }

inline Json::const_iterator& Json::const_iterator::operator++() {
    if (m_isObj)
        ++m_obj;
    else
        ++m_arr;
    return *this;
}

inline Json::const_iterator Json::const_iterator::operator++(int) {
    const_iterator t = *this;
    ++*this;
    return t;
}

inline bool Json::const_iterator::operator==(const const_iterator& o) const { return m_isObj ? m_obj == o.m_obj : m_arr == o.m_arr; }

inline const std::string& Json::const_iterator::key() const { return m_obj->first; }

inline Json::const_iterator Json::begin() const { return is_object() ? const_iterator(object_items().begin()) : const_iterator(array_items().begin()); }

inline Json::const_iterator Json::end() const { return is_object() ? const_iterator(object_items().end()) : const_iterator(array_items().end()); }

// Internal class hierarchy - JsonValue objects are not exposed to users of this API.
class JsonValue {
   protected:
    friend class Json;
    friend class JsonInt;
    friend class JsonDouble;
    friend class JsonSize;
    friend class JsonInt64;
    virtual Json::Type type() const                   = 0;
    virtual bool equals(const JsonValue* other) const = 0;
    virtual bool less(const JsonValue* other) const   = 0;
    virtual void dump(std::string& out) const         = 0;
    virtual double number_value() const;
    virtual size_t size_value() const;
    virtual int64_t int64_value() const;
    virtual int int_value() const;
    virtual bool bool_value() const;
    virtual const std::string& string_value() const;
    virtual const Json::array& array_items() const;
    virtual const Json& operator[](size_t i) const;
    virtual const Json::object& object_items() const;
    virtual const Json& operator[](const std::string& key) const;
    virtual ~JsonValue() {}

    virtual Json::object* mutable_object() { return nullptr; }
    virtual Json::array* mutable_array() { return nullptr; }
};

}  // namespace json11

json11::Json jKEY(const json11::Json& jConfig, const std::vector<std::string>& keys, int flag = 0x0);
inline std::vector<json11::Json> jChilds(const json11::Json& jConfig, const std::vector<std::string>& keys = {}, int flag = 0x0) {
    const json11::Json& jObj = keys.empty() ? jConfig : jKEY(jConfig, keys, flag);
    std::vector<json11::Json> childs;
    for (const auto& [key, child] : jObj.object_items()) {
        childs.push_back(child);
    }
    return childs;
}

bool inline J_has(const json11::Json& jObj, const std::string& key, int flag = 0x0) {
    const json11::Json::object& items = jObj.object_items();
    return items.find(key) != items.end();
}

bool inline J_isEmpty(const json11::Json& jObj) {
    if (jObj.is_null())
        return true;
    if ((jObj.is_array() && jObj.array_items().empty()) || (jObj.is_object() && jObj.object_items().empty()))
        return true;

    return false;
}

template <typename T>
T J_get(const json11::Json& jObj) {
    T val;
    NOT_IMPLEMENTED;
    return val;
}

template <typename T>
std::vector<T> J_getArray(const json11::Json& jObj) {
    std::vector<T> val;
    if (jObj.is_array()) {
        const auto& items = jObj.array_items();
        val.reserve(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
            T x{};
            if (items[i].try_get(x)) {
                val.push_back(std::move(x));
            } else {
                throw std::runtime_error("J_getArray: element " + std::to_string(i) + " has the wrong type: " + items[i].dump());
            }
        }
    } else {
        NOT_IMPLEMENTED;
    }
    return val;
}

inline std::string J_it2key(const json11::Json::const_iterator& it) { return it.key(); }
inline json11::Json J_it2value(const json11::Json::const_iterator& it) { return *it; }
