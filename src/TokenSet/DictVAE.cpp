

int DictVAE::STR2T(const char* txt, int txt_len, std::vector<TOKEN_ID>& btch, int flag) {
    int n_tokens = -1;
    if (wiki_tutor != nullptr) {
        n_tokens = wiki_tutor->STR2T(txt, btch, flag);
    } else {
        assert(0);
    }
    // if(tokenizer_add_bos)
    //     assert(btch[0]==bos);
    return n_tokens;
}
std::string DictVAE::T2STR(TOKEN_ID tok, int flag) {
    string word = "";

    if (wiki_tutor != nullptr) {
        word = wiki_tutor->T2STR(tok, flag);
    } else {
        assert(hDict != nullptr);
        word = hDict->T2STR(tok, flag);
    }
    return word;
}