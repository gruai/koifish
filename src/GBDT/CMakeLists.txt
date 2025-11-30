# Add this directory for internal users.
include_directories(.)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(DF ${CMAKE_CURRENT_SOURCE_DIR}/data_fold)
set(TREE ${CMAKE_CURRENT_SOURCE_DIR}/tree)
set(L ${CMAKE_CURRENT_SOURCE_DIR}/learn)
set(E ${CMAKE_CURRENT_SOURCE_DIR}/EDA)
 
set(mort_sources ${PWD}/LiteMORT.cpp ${PWD}/python/pyMORT_DLL.cpp 
    ${DF}/Histogram.cpp ${DF}/EDA.cpp ${DF}/DataFold.cpp ${DF}/FeatVector.cpp
    ${TREE}/ManifoldTree.cpp	${TREE}/GBRT.cpp	${TREE}/BoostingForest.cpp	${TREE}/BiSplit.cpp
    ${L}/LOSS.cpp	${L}/DCRIMI_.cpp	${L}/Pruning.cpp
    ${E}/Feat_Selection.cpp	${E}/SA_salp.cpp

    PARENT_SCOPE )
# add_compile_options(-Wunused -Wextra -Wcast-qual -Wno-unused-variable -Wunused-parameter -D_USE_OPENBLAS_)

