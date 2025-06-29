/**
 *  SPDX-FileCopyrightText: 2018-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief LiteMORT: A memory efficient gradient boosting tree system on adaptive compact distributions
 *  \author Yingshi Chen
 */

#include <iostream>

#include "./tree/GBRT.hpp"
#include "./tree/ManifoldTree.hpp"
using namespace Grusoft;

#ifdef PYBIND11
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using arr    = py::array;
// https://stackoverflow.com/questions/30388170/sending-a-c-array-to-python-and-back-extending-c-with-numpy
int test() {
    py::scoped_interpreter python;
    py::object np = py::module::import("numpy");
    auto v        = np.attr("sqrt")(py::cast(36.0));
    std::cout << "sqrt(36.0) = " << v.cast<double>() << std::endl;

    py::module sys = py::module::import("sys");
    py::print(sys.attr("path"));
    arr a;

    py::module t = py::module::import("tttt");
    t.attr("add")(1, 2);
    return 0;
}
#endif

/*
    Predict Contextual Sparsity of LLM
*/
int CS_Predict(FeatsOnFold *hTrainData, int flag) {
    // FeatsOnFold *hTrainData = FeatsOnFold::read_json("F:/Project/LiteMORT/data/nomad2018_train.json", 0x0);
    hTrainData->nCls = 2;
    int nTree        = 64;
    GBRT *hGBRT      = new GBRT(hTrainData, nullptr, 0.333, BoostingForest::CLASIFY, nTree);
    hGBRT->Train("", 50, 0x0);
    // hGBRT->Test("",);
    delete hGBRT;
    // delete hTrainData;		//�ѱ�hGBRTɾ��
    // test();
    return 0;
}
/*
    Lite Maniflod on Regression Tree
    v0.1	cys
        7/14/2018

int main(){
    //FeatsOnFold *hTrainData = FeatsOnFold::read_json("F:/Project/LiteMORT/data/1.json",0x0);
    FeatsOnFold *hTrainData = FeatsOnFold::read_json("F:/Project/LiteMORT/data/nomad2018_train.json", 0x0);
    hTrainData->nCls =2;
    int nTree = 64;
    GBRT *hGBRT = new GBRT(hTrainData,nullptr, 0.333, BoostingForest::CLASIFY, nTree);
    hGBRT->Train("", 50, 0x0);
    //hGBRT->Test("",);
    delete hGBRT;
    //delete hTrainData;		//�ѱ�hGBRTɾ��
    //test();
    return 0;
}*/
