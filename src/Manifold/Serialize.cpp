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
#include "Serialize.hpp"

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
    *objs = json_ptr;       *objs_nz = bytes_size;
    return data;
}
bool Fish::HF_Serialize(bool isSave, int flag){
    return false;
}
bool Fish::CALM_Serialize(const std::string&path, bool isSave, int flag){
try{    
    if(isSave){        
    }else{
        JSON header;
        size_t objs_size;
        void *objs,*data = MMAP_json(header,&objs,&objs_size,path,isSave,flag);        
        if(data==nullptr)   
            return false;      

        for (auto& [key, val] : header.items()) {
            if (key == "__metadata__") {
                JSON metadata = val;
                std::cout << "read metadata " << metadata << std::endl << std::endl;
            } else {
                hGensor target = GetGensor(key);    //  "model.embed.weight"    model.layers.0.attn_norm.weight  
                if(target==nullptr){
                    _INFO("\t[SERIAL] Failed @%s!",key.c_str());
                    continue;
                }
                if (target->SerialJSON(key, val, objs, objs_size) != 0) {
                    munmap(data, size);
                    return -1;
                }
            }
        }

        return true;
    }
    return false;
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
