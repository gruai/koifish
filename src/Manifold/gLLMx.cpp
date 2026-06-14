/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  More models derived from NLP_AutoRegressive
 *
 *  \brief General Language model
 *  \author Yingshi Chen
 */

#include "gLLM.hpp"

LLM_MAMBA::LLM_MAMBA(const std::string& nam_, struct CLI_params params, ROLE_TYPE role, int flag) : NLP_AutoRegressive(nam_, params, role, flag) {
    assert(arch == MODEL_ARCH::NLP_MAMBA);
    bool worst_case = true;
    // isLoadTokenEmbed = true;
    // config.common.adam.alpha = 0.0001;     //
}

hGensor LLM_MAMBA::BuildTarget(void* ctx, hGensor cur, int flag) { return nullptr; }

Guppy::Guppy(const std::string& nam_, struct CLI_params params, ROLE_TYPE role, int flag) : NLP_AutoRegressive(nam_, params, role, flag) {
    assert(arch == MODEL_ARCH::NLP_GUPPY);
    // config.model.isFFNShareParam = true;
    // config.model.isEmbedWeightTying = false;
    // isBias = config.model.isBias;    //   if true, converge much slower
}

string Guppy::__repr__(string& suffix, string& prefix, int flag) {
    char buf[5012]  = "\0";
    const char* tab = prefix.c_str();
    string sBasic   = NLP_AutoRegressive::__repr__(suffix, prefix, flag);
    sprintf(buf + strlen(buf), "%s", sBasic.c_str());
    _INFO("Guppy:    Bias=(normal=%d,slp=%d) AttOnBC=%d\n========\n", config.model.isNormalBias, config.model.isSLPBias, isAttOnBC);
    return buf;
}

// static uint64_t rng_seed = 42;

bool Guppy::BeforeNextStep(int iter, int flag) {
    int nLayer = config.nLayer(), l;
    for (l = 0; l < nLayer; l++) {
        FFN* ffn = GetNeuron<FFN>("FFN", l);
        ffn->UpdateSamps(iter * nLayer + l);
        // SelfAttention *qkv = GetNeuron<SelfAttention>("FFN",l);
    }
    return true;
}

string Guppy::DebugInfo(int type, int flag) {
    char buf[5012] = "\0";
    sprintf(buf + strlen(buf), "|gw|=(%.2f,%.2f)", hEmbed->w->gnorm, hEmbed->wInv->gnorm);
    return buf;
}

GPT2::GPT2(const std::string& nam_, struct CLI_params params, ROLE_TYPE role, int flag) : NLP_AutoRegressive(nam_, params, role, flag) {
    assert(arch == MODEL_ARCH::NLP_GPT2 || arch == MODEL_ARCH::NLP_GPT2_char);
    // isBias = config.model.isBias;    //   if true, converge much slower
}

/*
GPT(
  (transformer): ModuleDict(
    (wte): Embedding(65, 768)
    (wpe): Embedding(256, 768)
    (drop): Dropout(p=0, inplace=False)
    (h): ModuleList(
      (0): Block(
        (ln_1): LayerNorm()
        (attn): CausalSelfAttention(
          (c_attn): Linear(in_features=768, out_features=2304, bias=False)
          (c_proj): Linear(in_features=768, out_features=768, bias=False)
          (attn_dropout): Dropout(p=0, inplace=False)
          (resid_dropout): Dropout(p=0, inplace=False)
        )
        (ln_2): LayerNorm()
        (mlp): MLP(
          (c_fc): Linear(in_features=768, out_features=3072, bias=False)
          (gelu): GELU(approximate='none')
          (c_proj): Linear(in_features=3072, out_features=768, bias=False)
          (dropout): Dropout(p=0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm()
  )
  (lm_head): Linear(in_features=768, out_features=65, bias=False)
)

*/
/*
int DTS_GPT2::STR2T(void *hLLM,const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag)    {
    // bool gpt2_model_load(const std::string & fname, gpt2_model & model, gpt_vocab & vocab, const gpt_params & params) {
    gpt_vocab vocab;
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);
    return 0x0;
}*/

CDict_GPT2::CDict_GPT2(Fish* nlp_, int flag) : DictVAE(nlp_, flag) {
    int n_batch = config.n_batch(), n_ctx = config.n_ctx(), n_ctx_train = config.n_ctx_train, n_embd = config.nEmbed();
}

CDict_CHAR::CDict_CHAR(Fish* nlp_, int flag) : DictVAE(nlp_, flag) {
    int n_batch = config.n_batch(), n_ctx = config.n_ctx(), n_ctx_train = config.n_ctx_train, n_embd = config.nEmbed();
    // n_vocab=256;
}
int CDict_CHAR::InitMAEC(void* ctx_build, const std::vector<int>& dims_, int flag) {
    int n_batch = config.n_batch(), n_ctx = config.n_ctx(), n_ctx_train = config.n_ctx_train, n_embd = config.nEmbed();

    assert(gensors.size() == 0);
    return 0x0;
}
int CDict_CHAR::STR2T(const char* txt, int txt_len, std::vector<TOKEN_ID>& btch, int flag) {
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
std::string CDict_CHAR::T2STR(TOKEN_ID tok, int flag) {
    assert(tok >= 0 && tok < 256);
    string a = string(1, (char)tok);
    return a;
};

int CDict_GPT2::InitMAEC(void* ctx_build, const std::vector<int>& dims_, int flag) {
    int n_batch = config.n_batch(), n_ctx = config.n_ctx(), n_ctx_train = config.n_ctx_train, n_embd = config.nEmbed();

    // tok_embeddings = dolphin->AddTensor(ctx_build,_NAM_("token_embd.weight"),typNUMBER::F32,{n_embd, n_vocab},true,0x0);
    // _norm.BuildX("output_norm", {n_embd},dolphin,0x0);
    // _output.isBias = false;
    // _output.BuildX("output", {n_embd, n_vocab},dolphin,0x0);
    // assert(gensors.size()==0);
    return 0x0;
}

string GPT2::__repr__(string& suffix, string& prefix, int flag) {
    char buf[5012]  = "\0";
    const char* tab = prefix.c_str();
    string sBasic   = NLP_AutoRegressive::__repr__(suffix, prefix, flag);
    sprintf(buf + strlen(buf), "%s", sBasic.c_str());
    _INFO("GPT2:    Bias=(normal=%d,slp=%d) AttOnBC=%d\n========\n", config.model.isNormalBias, config.model.isSLPBias, isAttOnBC);
    return buf;
}

std::string CDict_GPT2::T2STR(TOKEN_ID tok, int flag) {
#ifdef _TENSOR_G_
    // [todo] call gpt2 tokenizer in next version
#else
    assert(0);
#endif
    return "";
};
int CDict_GPT2::STR2T(const char* txt, int txt_len, std::vector<TOKEN_ID>& btch, int flag) {
    //  https://github.com/wangkuiyi/huggingface-tokenizer-in-cxx
    //  https://www.daoplays.org/blog/gpt2_p1
    btch = {12518, 262, 7523, 318, 1016, 866, 11};  //  "when the smoke is going down"
    return btch.size();
    assert(0);
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

/*
bool CDict_GPT2::LoadTokenizer(const char *filename,int flag) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        _WARN("%s: Failed to open the tokenizer file @%s\n", filename);
        init_ok = 0;
        return false;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328);
    int version = header[1];
    n_vocab = header[2];
    if (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        assert(n_vocab == 50257); // let's be defensive here
        eot_token = 50256;
    } else if (version == 2) {
        eot_token = header[3];
    } else {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned char length;
    token_table = (char **)mallocCheck(n_vocab * sizeof(char *));
    for (uint32_t i = 0; i < n_vocab; i++) {
        freadCheck(&length, sizeof(unsigned char), 1, file);
        assert(length > 0); // every token should be at least one character
        char *token_bytes = (char *)mallocCheck(length + 1);
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    init_ok = true;
    return init_ok;
}*/