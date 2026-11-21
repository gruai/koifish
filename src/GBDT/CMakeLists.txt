cmake_minimum_required (VERSION 3.27)
project(gbdt VERSION 0.2.0 LANGUAGES CXX)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(DF ${CMAKE_CURRENT_SOURCE_DIR}/data_fold)
set(TREE ${CMAKE_CURRENT_SOURCE_DIR}/tree)
set(L ${CMAKE_CURRENT_SOURCE_DIR}/learn)
set(E ${CMAKE_CURRENT_SOURCE_DIR}/EDA)
 
set(mort_sources ${PWD}/LiteMORT.cpp ${PWD}/python/pyMORT_DLL.cpp 
    ${DF}/Histogram.cpp ${DF}/EDA.cpp ${DF}/DataFold.cpp 
    ${DF}/FeatVector.cpp
    ${TREE}/ManifoldTree.cpp	${TREE}/GBRT.cpp	${TREE}/BoostingForest.cpp	${TREE}/BiSplit.cpp
    ${L}/LOSS.cpp	${L}/DCRIMI_.cpp	${L}/Pruning.cpp
    ${E}/Feat_Selection.cpp	${E}/SA_salp.cpp)
    # PARENT_SCOPE )

message(STATUS "GBDT_lib  CXX=17 device_sources=${mort_sources}")

add_library(gbdt STATIC ${mort_sources})        #SHARED

target_include_directories(gbdt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/Utils/)

set_target_properties(gbdt PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON   # Fail if C++17 isn't available
    CXX_EXTENSIONS OFF         # Use pure C++17, not gnu++17
)


