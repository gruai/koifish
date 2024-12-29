#pragma once

#include "ggml.h"

#include <fstream>
#include <vector>
#include <string>

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

//
// logging
//
#define GGML_DEBUG 0
#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

#define GGML_PRINT(...) printf(__VA_ARGS__)

#ifdef GGML_PERF
#define ggml_perf_time_ms()       ggml_time_ms()
#define ggml_perf_time_us()       ggml_time_us()
#define ggml_perf_cycles()        ggml_cycles()
#define ggml_perf_cycles_per_ms() ggml_cycles_per_ms()
#else
#define ggml_perf_time_ms()       0
#define ggml_perf_time_us()       0
#define ggml_perf_cycles()        0
#define ggml_perf_cycles_per_ms() 0
#endif

enum ggml_ftype ggml_parse_ftype(const char * str);

void ggml_print_ftypes(FILE * fp = stderr);

bool ggml_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip);

struct ggml_context {
        size_t mem_size;
        void * mem_buffer;
        bool   mem_buffer_owned;
        bool   no_alloc;
        bool   no_alloc_save; // this is used to save the no_alloc state when using scratch buffers

        int    n_objects;

        struct ggml_object * objects_begin;
        struct ggml_object * objects_end;

        struct ggml_scratch scratch;
        struct ggml_scratch scratch_save;
};

typedef double ggml_float;



#ifdef __cplusplus
extern "C" {
#endif
    //
    // For TGraph
    //
        size_t ggml_hash_size(size_t min_sz);
        size_t ggml_graph_nbytes(size_t size, bool grads);
        struct ggml_object * ggml_new_object(struct ggml_context * ctx, enum ggml_object_type type, size_t size);
        //
        // ggml context
        //

        

        void * ggml_graph_compute_thread(void * data);
        void clear_numa_thread_affinity(void);
        int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads);
        
        // enum ggml_opt_result ggml_train(struct ggml_context * ctx,struct ggml_opt_context * opt,
        //         struct ggml_opt_params params,struct ggml_tensor * f,struct ggml_cgraph * gf,struct ggml_cgraph * gb,struct train_opt_callback_data * callback_data);
#ifdef __cplusplus
}
#endif