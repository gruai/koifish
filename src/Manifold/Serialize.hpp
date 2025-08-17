/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT 
 *  
 *  Serialization
 * 
 *  \brief Serialization
 *  \author Yingshi Chen
 */
#pragma once
#include "../CLI_params.hpp"
#include "../g_stddef.hpp"

class FSerial{
    FILE *_stream=NULL;
    bool _valid = false;
public:
    FSerial(const std::string& sPath,bool isSave,int flag){
    try{
        if(isSave){
            if((_stream=fopen(sPath.c_str(),"wb"))!=NULL){
                _valid = true;
            }
        }else{
            if((_stream=fopen(sPath.c_str(),"rb"))!=NULL){
                _valid = true;
            }
        }
        
    }catch(...){
        
        _valid = false;
    }
    }

    virtual ~FSerial(){
    try{
        // if(_stream!=NULL)    
        //     fclose(_stream);
    }   catch(...){

    }  
    }

    virtual bool isValid()  {   return _valid; }

    template<typename T>
    bool Serial(T *val,int nz,bool isSave,int flag=0x0){
        if(!isValid())  return false;
        if(isSave){
            if(fwrite(val,sizeof(T),nz,_stream)!=nz)
                return false;
            fflush(_stream);
        }else{
            if(fread(val,sizeof(T),nz,_stream)!=nz)
                return false;
        }
        return true;
    }

    template<typename T>
    bool Serial(T &val,bool isSave,int flag=0x0){        
        return Serial(&val,1,isSave,flag);
    }
        
    template<typename T>
    bool Serial(std::vector<T>& arrT,bool isSave,int flag=0x0){
        if(!isValid())  return false;
        size_t nT=arrT.size(),i;
        Serial(&nT,1,isSave);
        if(nT==0){
            return true;
        }
        if(isSave){
            if(fwrite(arrT.data(),sizeof(T),nT,_stream)!=nT)
                return false;
            fflush(_stream);
        }else{
            arrT.clear();       arrT.resize(nT);
            if(fread(arrT.data(),sizeof(T),nT,_stream)!=nT)
                return false;
            // T* buf = new T[nT];
            // size_t nRead = fread((void*)(buf),sizeof(T),nT,_stream);
            // if(nRead!=nT)
            //     return false;    
            // std::copy(buf, buf + nT, std::back_inserter(arrT));
            // delete[] buf;
        }
        return true;
    }

    template<typename T,typename Tchild>
    bool Serial_Vector(std::vector<T*>& arrT,bool isSave,int flag=0x0){
        if(!isValid())  return false;
        size_t nT=arrT.size(),i;
        Serial(&nT,1,isSave);
        if(isSave){
            for(auto obj0:arrT){
                Tchild *obj = dynamic_cast<Tchild*>(obj0);
                assert(obj!=nullptr);
                if(!obj->Serialize(*this,isSave,flag))
                    return false;
            }
        }else{
            arrT.clear();       arrT.resize(nT);
            for(i=0;i<nT;i++){
                Tchild *obj = new Tchild();
                if(!obj->Serialize(*this,isSave,flag))
                    return false;
                arrT[i] = obj;
            }
        }
        return true;
    }
    
};

// only load jConfig from safeternsor model
bool SAFETENSOR_Load_jconfig(const std::string&path, JSON&jsConfig, int flag=0x0);
std::string LoadSomeText(const string&fpath,const int nMost=2048,int flag=0x0);