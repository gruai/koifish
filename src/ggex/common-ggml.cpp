// #include "common-ggml.h"
#include <math.h>

#include <map>
#include <regex>

#define UNUSED GGML_UNUSED

static const std::map<std::string, enum ggml_ftype> GGML_FTYPE_MAP = {
    {"q4_0", GGML_FTYPE_MOSTLY_Q4_0}, {"q4_1", GGML_FTYPE_MOSTLY_Q4_1}, {"q5_0", GGML_FTYPE_MOSTLY_Q5_0}, {"q5_1", GGML_FTYPE_MOSTLY_Q5_1},
    {"q8_0", GGML_FTYPE_MOSTLY_Q8_0}, {"q2_k", GGML_FTYPE_MOSTLY_Q2_K}, {"q3_k", GGML_FTYPE_MOSTLY_Q3_K}, {"q4_k", GGML_FTYPE_MOSTLY_Q4_K},
    {"q5_k", GGML_FTYPE_MOSTLY_Q5_K}, {"q6_k", GGML_FTYPE_MOSTLY_Q6_K},
};

void ggml_print_ftypes(FILE *fp) {
    for (auto it = GGML_FTYPE_MAP.begin(); it != GGML_FTYPE_MAP.end(); it++) {
        fprintf(fp, "  type = \"%s\" or %d\n", it->first.c_str(), it->second);
    }
}

enum ggml_ftype ggml_parse_ftype(const char *str) {
    enum ggml_ftype ftype;
    if (str[0] == 'q') {
        const auto it = GGML_FTYPE_MAP.find(str);
        if (it == GGML_FTYPE_MAP.end()) {
            fprintf(stderr, "%s: unknown ftype '%s'\n", __func__, str);
            return GGML_FTYPE_UNKNOWN;
        }
        ftype = it->second;
    } else {
        ftype = (enum ggml_ftype)atoi(str);
    }

    return ftype;
}
