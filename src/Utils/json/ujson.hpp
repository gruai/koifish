#ifndef UJSON_HPP
#define UJSON_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "json.hpp"
using nlohmann_json = nlohmann::json;
namespace ujson {
class json {
   public:
    // Constructors
    json() : m_doc(std::make_shared<nlohmann_json>()), m_val(m_doc.get()) {}
    json(const nlohmann_json& j) : m_doc(std::make_shared<nlohmann_json>(j)), m_val(m_doc.get()) {}
    json(nlohmann_json&& j) : m_doc(std::make_shared<nlohmann_json>(std::move(j))), m_val(m_doc.get()) {}

    json(const json& other) : m_doc(other.m_doc), m_val(other.m_val) {}
    json(json&& other) : m_doc(std::move(other.m_doc)), m_val(other.m_val) {}

    json(std::nullptr_t) : json() { *m_val = nullptr; }
    json(bool b) : json() { *m_val = b; }

    template <typename T, typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value, int>::type = 0>
    json(T i) : json() {
        if (std::is_signed<T>::value)
            *m_val = static_cast<int64_t>(i);
        else
            *m_val = static_cast<uint64_t>(i);
    }

    json(float f) : json() { *m_val = f; }
    json(double d) : json() { *m_val = d; }
    json(const std::string& s) : json() { *m_val = s; }
    json(const char* s) : json() { *m_val = s; }

    json(std::initializer_list<json> init) : json() {
        bool is_obj = true;
        for (const auto& v : init) {
            if (!v.is_array() || v.size() != 2 || !v[0].is_string()) {
                is_obj = false;
                break;
            }
        }
        if (is_obj && init.size() > 0) {
            *m_val = nlohmann_json::object();
            for (const auto& v : init) {
                (*m_val)[v[0].get<std::string>()] = *(v[1].m_val);
            }
        } else {
            *m_val = nlohmann_json::array();
            for (const auto& v : init) {
                m_val->push_back(*(v.m_val));
            }
        }
    }

    // Assignment updates the pointed-to value
    json& operator=(const json& other) {
        if (this == &other)
            return *this;
        if (m_val == other.m_val)
            return *this;
        *m_val = *(other.m_val);
        return *this;
    }

    // Type checks
    bool is_null() const { return m_val->is_null(); }
    bool is_boolean() const { return m_val->is_boolean(); }
    bool is_number() const { return m_val->is_number(); }
    bool is_number_integer() const { return m_val->is_number_integer(); }
    bool is_number_float() const { return m_val->is_number_float(); }
    bool is_number_unsigned() const { return m_val->is_number_unsigned(); }
    bool is_string() const { return m_val->is_string(); }
    bool is_array() const { return m_val->is_array(); }
    bool is_object() const { return m_val->is_object(); }

    template <typename T>
    T get() const {
        return m_val->get<T>();
    }

    operator std::string() const { return m_val->get<std::string>(); }

    json operator[](const std::string& key) const {
        if (is_object() && m_val->contains(key)) {
            return json(m_doc, const_cast<nlohmann_json*>(&(m_val->at(key))));
        }
        return json(nullptr, nullptr);
    }

    json operator[](const std::string& key) {
        if (!is_object())
            *m_val = nlohmann_json::object();
        return json(m_doc, &((*m_val)[key]));
    }

    json operator[](size_t idx) const {
        if (is_array() && idx < m_val->size()) {
            return json(m_doc, const_cast<nlohmann_json*>(&(m_val->at(idx))));
        }
        return json(nullptr, nullptr);
    }

    json operator[](size_t idx) {
        if (!is_array())
            *m_val = nlohmann_json::array();
        return json(m_doc, &((*m_val)[idx]));
    }

    json at(const std::string& key) const { return json(m_doc, const_cast<nlohmann_json*>(&(m_val->at(key)))); }
    json at(const std::string& key) { return json(m_doc, &(m_val->at(key))); }

    template <typename T>
    T value(const std::string& key, const T& default_value) const {
        auto it = m_val->find(key);
        if (it != m_val->end()) {
            return json(m_doc, const_cast<nlohmann_json*>(&(*it))).get<T>();
        }
        return default_value;
    }

    json value(const std::string& key, const json& default_value) const {
        auto it = m_val->find(key);
        if (it != m_val->end()) {
            return json(m_doc, const_cast<nlohmann_json*>(&(*it)));
        }
        return default_value;
    }

    std::string value(const std::string& key, const char* default_value) const { return m_val->value(key, std::string(default_value)); }

    bool contains(const std::string& key) const { return m_val->contains(key); }
    size_t count(const std::string& key) const { return m_val->count(key); }
    size_t size() const { return m_val->size(); }
    bool empty() const { return m_val->empty(); }

    class iterator {
        friend class json;
        friend class const_iterator;
        std::shared_ptr<nlohmann_json> m_doc;
        nlohmann_json::iterator m_it;

       public:
        iterator(std::shared_ptr<nlohmann_json> doc, nlohmann_json::iterator it) : m_doc(doc), m_it(it) {}
        bool operator!=(const iterator& other) const { return m_it != other.m_it; }
        bool operator==(const iterator& other) const { return m_it == other.m_it; }
        iterator& operator++() {
            ++m_it;
            return *this;
        }
        std::string key() const { return m_it.key(); }
        json value() const { return json(m_doc, &(*m_it)); }
        json operator*() { return json(m_doc, &(*m_it)); }
    };

    class const_iterator {
        friend class json;
        std::shared_ptr<nlohmann_json> m_doc;
        nlohmann_json::const_iterator m_it;

       public:
        const_iterator(std::shared_ptr<nlohmann_json> doc, nlohmann_json::const_iterator it) : m_doc(doc), m_it(it) {}
        const_iterator(const iterator& other) : m_doc(other.m_doc), m_it(other.m_it) {}
        bool operator!=(const const_iterator& other) const { return m_it != other.m_it; }
        bool operator==(const const_iterator& other) const { return m_it == other.m_it; }
        bool operator!=(const iterator& other) const { return m_it != other.m_it; }
        bool operator==(const iterator& other) const { return m_it == other.m_it; }
        const_iterator& operator++() {
            ++m_it;
            return *this;
        }
        std::string key() const { return m_it.key(); }
        json value() const { return json(m_doc, const_cast<nlohmann_json*>(&(*m_it))); }
        const json operator*() const { return json(m_doc, const_cast<nlohmann_json*>(&(*m_it))); }
    };

    iterator begin() { return iterator(m_doc, m_val->begin()); }
    iterator end() { return iterator(m_doc, m_val->end()); }
    const_iterator begin() const { return const_iterator(m_doc, m_val->begin()); }
    const_iterator end() const { return const_iterator(m_doc, m_val->end()); }

    iterator find(const std::string& key) { return iterator(m_doc, m_val->find(key)); }
    const_iterator find(const std::string& key) const { return const_iterator(m_doc, m_val->find(key)); }

    void push_back(const json& val) { m_val->push_back(*(val.m_val)); }
    void erase(const std::string& key) { m_val->erase(key); }
    void clear() { m_val->clear(); }

    bool operator==(const json& other) const { return *m_val == *(other.m_val); }
    bool operator!=(const json& other) const { return *m_val != *(other.m_val); }
    bool operator<(const json& other) const { return *m_val < *(other.m_val); }
    bool operator>(const json& other) const { return *m_val > *(other.m_val); }
    bool operator<=(const json& other) const { return *m_val <= *(other.m_val); }
    bool operator>=(const json& other) const { return *m_val >= *(other.m_val); }

    static json parse(const std::string& s) {
        try {
            auto doc = std::make_shared<nlohmann_json>(nlohmann_json::parse(s));
            return json(doc, doc.get());
        } catch (...) {
            return json();
        }
    }
    static json object() {
        json j;
        *(j.m_val) = nlohmann_json::object();
        return j;
    }
    static json array() {
        json j;
        *(j.m_val) = nlohmann_json::array();
        return j;
    }

    std::string dump(int indent = -1) const { return m_val->dump(indent); }

    const nlohmann_json& raw() const { return *m_val; }
    nlohmann_json& raw() { return *m_val; }

   private:
    // Internal constructor
    json(std::shared_ptr<nlohmann_json> doc, nlohmann_json* val) : m_doc(doc), m_val(val) {
        if (!m_val) {
            static nlohmann_json null_val = nullptr;
            m_val                         = &null_val;
            m_doc                         = std::make_shared<nlohmann_json>(nullptr);
        }
    }

    std::shared_ptr<nlohmann_json> m_doc;
    nlohmann_json* m_val;
};
}  // namespace ujson

#endif  // UJSON_HPP