/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *  
 *  
 * 
 *  \brief Serialization
 *  \author Yingshi Chen
 */
#include <set>
#include <sys/mman.h>
#include "Fish.hpp"
#include "../TokenSet/Dictionary.hpp"
#include "Serialize.hpp"
#include "../Utils/GST_os.hpp"

void* MMAP_json(JSON& header,void**objs,size_t*objs_nz,const std::string&path, bool isSave, int flag){
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        return nullptr;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return nullptr;
    }        
    size_t size = st.st_size;
    void *data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        return nullptr;
    }
    #ifdef __linux__
    // increases readahead buffer size, resulting in faster cold loads
    posix_fadvise(fd, 0, size, POSIX_FADV_SEQUENTIAL);
    #endif
    close(fd); // fd can be closed after mmap returns without invalidating the mapping

    // Parse the metadata JSON and the tensors
    if (size < sizeof(uint64_t)) {
        munmap(data, size);
        return nullptr;
    }

    uint64_t json_size = *(uint64_t*)data;
    if (json_size == 0 || json_size > size - sizeof(uint64_t)) {
        munmap(data, size);
        return nullptr;        
    }

    char* json_ptr = (char*)data + sizeof(uint64_t);
    void* bytes_ptr = (char*)data + sizeof(uint64_t) + json_size;
    size_t bytes_size = size - sizeof(uint64_t) - json_size;

    std::string json_str(json_ptr, json_size);
    header = JSON::parse(json_str);
    *objs = bytes_ptr;       *objs_nz = bytes_size;
    return data;
}
bool Fish::HF_Serialize(bool isSave, int flag){
    return false;
}

bool MODEL_CARD::OnJsonCALM(const std::string&path,const JSON &meta,int flag){
    sTokenPath = path;
    dim = jKVs(meta,{"dim"},dim);           //atoi(meta["dim"]);
	hidden_dim = jKVs(meta,{"hidden_dim"},hidden_dim);    //atoi(meta["hidden_dim"]);
	n_layers = jKVs(meta,{"n_layers"},n_layers);      //atoi(meta["n_layers"]);
	n_heads = jKVs(meta,{"n_heads"},n_heads);      //atoi(meta["n_heads"]);
	n_kv_heads = jKVs(meta,{"n_kv_heads"},n_kv_heads);      //atoi(meta["n_kv_heads"]);
    head_dim = jKVs(meta,{"head_dim"},head_dim);
    assert(dim == head_dim*n_heads);
    layerps.clear();
    for(int i=0;i<n_layers;i++){
        LAY_PARAM lay(n_heads,n_kv_heads,hidden_dim);
        layerps.push_back(lay);
    }
    string info = "";
    info = jKVs(meta,{"dtype"},info);   
    tpWeight = tpNumOf(info);
    int wbit = BitPE(tpWeight);
    fDotW = fnDot(tpWeight);

    jModelParam = meta;
	vocab_size = jKVs(meta,{"vocab_size"},0);
    bos_token_id = jKVs(meta,{"bos_token_id"},0);
    eos_token_id = jKVs(meta,{"eos_token_id"},0);
	// 
	// const char* max_seq_len = meta["max_seq_len"];
	// config.seq_len = max_seq_len && atoi(max_seq_len) < 4096 ? atoi(max_seq_len) : 4096;
    seq_len = jKVs(meta,{"max_seq_len"},0);
    seq_len = std::min(seq_len,4096);// for now limit seq_len to 4096 to avoid KV cache OOM for models like Mistral since window size isn't correctly specified
	// if (context) {
	// 	config.seq_len = context;
	// }

	// config.rope_theta = atof(meta["rope_theta"]);
	// config.rotary_dim = atoi(meta["rotary_dim"]);
    rope_theta = jKVs(meta,{"rope_theta"},rope_theta);
    rotary_dim = jKVs(meta,{"rotary_dim"},rotary_dim);
	// if (meta["n_experts"]) {
	// 	config.n_experts = atoi(meta["n_experts"]);
	// 	config.n_experts_ac = atoi(meta["n_experts_active"]);
	// }
    n_experts = jKVs(meta,{"n_experts"},n_experts);
    n_experts_ac = jKVs(meta,{"n_experts_active"},n_experts_ac);
	// const char* norm_eps = meta["norm_eps"];
	norm_eps = jKVs(meta,{"norm_eps"},norm_eps);

	// const char* act_type = meta["act_type"];
	act_type = jKVs(meta,{"act_type"},act_type);   //act_type && strcmp(act_type, "gelu"] == 0;

	// const char* norm_type = meta["norm_type"];
    norm_type = jKVs(meta,{"norm_type"},norm_type);   //act_type && strcmp(act_type, "gelu"] == 0;
	// config.norm_ln = norm_type && strncmp(norm_type, "layernorm", 9) == 0;  // note: we currently don't support layernorm bias
	// config.norm_par = norm_type && strcmp(norm_type, "layernorm_par"] == 0; // note: we currently don't support layernorm bias

	// const char* qkv_clip = meta["qkv_clip"];
	// config.qkv_clip = qkv_clip ? atof(qkv_clip) : FLT_MAX;
    clip_qkv = jKVs(meta,{"qkv_clip"},clip_qkv);  
    return true;
}

bool Fish::CALM_Serialize(const std::string&path, bool isOnlyVocab, int flag){
try{   
    JSON header;
    size_t objs_size;
    int nSerialT = 0;
    void *objs,*data = MMAP_json(header,&objs,&objs_size,path,false,flag);     
    hGTensor tokens=std::make_shared<GTensor>(),scores=std::make_shared<GTensor>();   
    if(data==nullptr)   
        return false;      

    for (auto& [key, val] : header.items()) {
        if (key == "__metadata__") {
            JSON metadata = val;
            std::cout << "read metadata " << metadata << std::endl << std::endl;
            config.model.OnJsonCALM(path,metadata,0x0);
            // InitDictTokenset();
            if(isOnlyVocab)
                continue;
        } else if(key=="tokenizer.tokens") {
            tokens->SerialJSON(key, val, objs, objs_size,flag | GTensor::F_NOALLOC);
        } else if(key=="tokenizer.scores") {
            scores->SerialJSON(key, val, objs, objs_size,flag | GTensor::F_NOALLOC);
        } else{
            if(isOnlyVocab)
                continue;

            hGensor target = GetGensor(key);    //  "model.embed.weight"    model.layers.0.attn_norm.weight  
            if(target==nullptr){
                _INFO("\t[SERIAL] Failed @%s!\n",key.c_str());
                continue;
            }
                       
            if (target->SerialJSON(key, val, objs, objs_size) != 0) {
                munmap(data, size);
                return -1;
            }   //  
            
            if(G_Has_(target->name,{"mlp.w1.weight"}))  {      // "layers.27.mlp.w1.weight" wk.weight wq.weight wv.weight wo.weight ,"w2.weight","w3.weight"
                BIT_SET(target->flags,GTensor::F_TERNARY);
                // target->ToTernary();
            }
            if(DUMP()){
                _INFO("  >>>>  %d typ=%s\t data=%p grad=%p \t sz=%ld @%s\n",nSerialT,cNameOf(target->type),target->data,target->grad,tBYTE(target),target->name);
            }
            // if(strcmp(target->name,"model.layers.27.mlp.norm.weight")==0){   //only for debug model.output.weight
            //     target->Print(key,0,-1);                //PrintTensor<f8e5m2_t>("wout",target->data,target->ne[0],dim);
            // }
            nSerialT++;
        }
    }
    _INFO("[SERIAL] n_T=%d\n", nSerialT);
    hDict->InitFrom(this,tokens,scores,0x0);
    tokens.reset();
    scores.reset();
    return true;
}catch(...){
    return false;
}
}

bool Fish::YALM_Serialize(const std::string&path, bool isSave, int flag){
try{    
    if(isSave){        
    }else{
        std::vector<std::string> files;
        DIR* dir = opendir(path.c_str());
        if (dir == nullptr) {
            std::cout << "failed to open directory" << std::endl;
            return -1;
        }
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            // Skip . and .. directory entries
            if (filename != "." && filename != "..") {
            files.push_back(path + "/" + filename);
            }
        }
        closedir(dir);
        if (files.empty()) {
            std::cout << "no files found" << std::endl;
            return -1;
        }
        std::sort(files.begin(), files.end());

        // Read first file with metadata
        if (CALM_Serialize(files[0], true) != 0) {
            std::cout << "failed to read metadata" << std::endl;
            return -1;
        }      
        // Read remaining files without metadata
        for (size_t i = 1; i < files.size(); i++) {
            if (CALM_Serialize(files[i], false) != 0) {
            std::cout << "failed to read file " << files[i] << std::endl;
            return -1;
            }
        }

        return true;
    }
    return false;
}catch(...){
    return false;
}
}

std::string LoadSomeText(const string&fpath,const int nMost,int flag)   {
    assert(nMost>0);
    string txt="";
    FILE *fp = std::fopen(fpath.c_str(), "rt");
    if(fp==NULL)    return txt;

    // std::fseek(fp, 42, SEEK_SET);
    char buf[nMost+1] = "\0";
    size_t sz = std::fread(buf, 1, nMost, fp);
    buf[sz] = '\0';
    txt = buf;  
    return txt;
}
