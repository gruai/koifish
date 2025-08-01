    #pragma once

#include <cstdlib>

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "matmul.h"
#include "pointwise.h"
#include "rng.h"
#include "softmax.h"
#include "paged_cache_load.h"

namespace cudnn_frontend::graph {

namespace attn::score_modifiers {

// clang-format off
inline float get_negative_inf_value();

inline std::shared_ptr<Tensor_attributes> causal_mask(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<Tensor_attributes> attention_score
);

inline std::shared_ptr<Tensor_attributes> bias(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<Tensor_attributes> attention_score,
    std::shared_ptr<Tensor_attributes> bias_tensor
);

inline std::shared_ptr<Tensor_attributes> causal_mask_bottom_right(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<Tensor_attributes> attention_score,
    std::shared_ptr<Tensor_attributes> seq_len_q,
    std::shared_ptr<Tensor_attributes> seq_len_kv
);

inline std::shared_ptr<Tensor_attributes> padding_mask(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<Tensor_attributes> attention_score,
    std::shared_ptr<Tensor_attributes> seq_len_kv,
    std::shared_ptr<Tensor_attributes> seq_len_q
);

inline std::shared_ptr<Tensor_attributes> sliding_window_mask(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<Tensor_attributes> attention_score,
    DiagonalAlignment_t diagonal_alignment,
    std::optional<int64_t> left_window,
    std::optional<int64_t> right_window,
    int64_t s_q,
    int64_t s_kv,
    std::shared_ptr<Tensor_attributes> s_q_ptr,
    std::shared_ptr<Tensor_attributes> s_kv_ptr
);

inline std::shared_ptr<Tensor_attributes> alibi_mask(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<Tensor_attributes> attention_score,
    std::shared_ptr<Tensor_attributes>& alibi_slopes,
    int64_t h_q,
    int64_t& alibi_slopes_size
);
// clang-format on

}  // namespace attn::score_modifiers

class SDPANode : public NodeCRTP<SDPANode> {
    using input_names  = SDPA_attributes::input_names;
    using output_names = SDPA_attributes::output_names;

    std::shared_ptr<Tensor_attributes> rng_output;
    std::shared_ptr<Tensor_attributes> alibi_slopes;
    int64_t alibi_slopes_size = 0;

   public:
    SDPA_attributes attributes;

    SDPANode(SDPA_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    bool
    is_paged_v() const {
        auto page_table_v_it = attributes.inputs.find(input_names::Page_table_V);
        return ((page_table_v_it) != attributes.inputs.end() && page_table_v_it->second != nullptr);
    }

    bool
    is_paged_k() const {
        auto page_table_k_it = attributes.inputs.find(input_names::Page_table_K);
        return ((page_table_k_it) != attributes.inputs.end() && page_table_k_it->second != nullptr);
    }

    // Helper function to infer KV sequence length
    // Note that it cannot be run as part of infer_properties_node as
    // this is being used in pre_validate_node
    int64_t
    infer_s_kv() const {
        int64_t s_kv = -1;

        auto get_input_dim = [this](const SDPA_attributes::input_names& input_name) {
            auto const input_it = attributes.inputs.find(input_name);
            if (input_it != attributes.inputs.end()) {
                return input_it->second->get_dim();
            } else {
                return std::vector<int64_t>({-1, -1, -1, -1});
            }
        };

        auto const& k_dim = get_input_dim(input_names::K);
        auto const& v_dim = get_input_dim(input_names::V);

        // If s_kv was set explicitly, use that
        if (attributes.max_seq_len_kv.has_value()) {
            s_kv = attributes.max_seq_len_kv.value();
        }
        // When one of K or V cache are paged, s_kv can be extracted directly
        else if (!is_paged_k()) {
            s_kv = k_dim[2];

        } else if (!is_paged_v()) {
            s_kv = v_dim[2];
        } else {
            CUDNN_FE_LOG_LABEL_ENDL(
                "WARNING: maximum kv sequence length is being inferred. To set it explicitly, please use  "
                "\"set_paged_attention_max_seq_len_kv\"");

            auto bias_it = attributes.inputs.find(input_names::Bias);
            auto rng_it  = attributes.outputs.find(output_names::RNG_DUMP);

            // If there is a bias, extract it from there
            if (bias_it != attributes.inputs.end() && bias_it->second != nullptr) {
                s_kv = get_input_dim(input_names::Bias)[3];
                // If there is an rng_dump output, extract it from there
            } else if (rng_it != attributes.outputs.end() && rng_it->second != nullptr) {
                s_kv = rng_it->second->get_dim()[3];
                // When both caches are paged, and the above failed, we need to infer s_kv from the page table and
                // container
            } else {
                // [b, 1, ceil(s_kv/block_size), 1]
                auto page_table_dim_k = get_input_dim(input_names::Page_table_K);
                // [b, h_k, block_size, d_k]
                auto const container_dim_k = get_input_dim(input_names::K);
                int64_t s_k                = page_table_dim_k[2] * container_dim_k[2];

                // [b, 1, ceil(s_kv/block_size), 1]
                auto page_table_dim_v = get_input_dim(input_names::Page_table_V);
                // [b, h_v, block_size, d_v]
                auto const container_dim_v = get_input_dim(input_names::V);
                int64_t s_v                = page_table_dim_v[2] * container_dim_v[2];

                s_kv = std::min(s_k, s_v);
            }
        }

        return s_kv;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating SDPANode " << attributes.name << "...");

        // check that Q, K, V, O tensors has been assigned
        // check that dim and strides has been assigned and last stride is 1
#define CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(port, port_map)                                                        \
    {                                                                                                            \
        std::shared_ptr<Tensor_attributes> tensor_ptr = port_map.at(port);                                       \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_dim().size() != 4,                                        \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                          \
                                       "The dim for " + std::string(#port) + " is invalid");                     \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_stride().size() != 4,                                     \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                          \
                                       "The stride for " + std::string(#port) + " is invalid");                  \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                          \
            tensor_ptr->get_stride()[3] != 1,                                                                    \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                   \
            "The stride for the last dimension corresponding to the embedding size per head should be 1 for " +  \
                std::string(#port));                                                                             \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                          \
            tensor_ptr->get_stride()[2] == 0,                                                                    \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                   \
            "The stride for the dimension corresponding to the sequence lengths per head should not be 0 for " + \
                std::string(#port));                                                                             \
    }

        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::Q, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::K, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::V, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(output_names::O, attributes.outputs);

        // validate options for is_inference and stats tensor
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.is_inference.has_value() == false,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "is_inference attribute not set");

        if (attributes.is_inference.value() == false) {
            CUDNN_FE_VALIDATE_OUTPUT_TENSOR(output_names::Stats);
        }

#undef CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE

        // validate backend limitations for the operation
        // clang-format off
        int64_t s_q  = attributes.inputs.at(input_names::Q)->get_dim()[2];
        int64_t s_kv = infer_s_kv(); // When using paged attention K/V dimensions are implicit
        int64_t h_q  = attributes.inputs.at(input_names::Q)->get_dim()[1];
        int64_t h_k  = attributes.inputs.at(input_names::K)->get_dim()[1];
        int64_t h_v  = attributes.inputs.at(input_names::V)->get_dim()[1];
        int64_t d_qk = attributes.inputs.at(input_names::Q)->get_dim()[3];
        int64_t d_v  = attributes.inputs.at(input_names::V)->get_dim()[3];

        bool const is_ragged = attributes.inputs.at(input_names::Q)->get_ragged_offset() ||
                               attributes.inputs.at(input_names::K)->get_ragged_offset() ||
                               attributes.inputs.at(input_names::V)->get_ragged_offset() ||
                               attributes.outputs.at(output_names::O)->get_ragged_offset();

        auto const& bias_mask = attributes.inputs.find(input_names::Bias);
        bool const is_bias   = (bias_mask != attributes.inputs.end() && bias_mask->second != nullptr);

        auto const& dropout_mask     = attributes.inputs.find(input_names::Dropout_mask);
        bool const is_dropout_custom = (dropout_mask != attributes.inputs.end()) && (dropout_mask->second != nullptr);
        bool const is_dropout        = attributes.dropout_probability.has_value() || is_dropout_custom;

        bool const is_paged  = is_paged_k() || is_paged_v();

        auto const& rng_tensor = attributes.outputs.find(output_names::RNG_DUMP);
        bool const is_rng   = (rng_tensor != attributes.outputs.end() && rng_tensor->second != nullptr);
        
        bool const max_seq_kv_explicit = attributes.max_seq_len_kv.has_value();

        // validation TODO:
        //    - validate stats has valid dims
        cudaDeviceProp prop;
        int device;
        CHECK_CUDA_ERROR(detail::cuda_get_device(&device));
        CHECK_CUDA_ERROR(detail::cuda_get_device_properties(&prop, device));

        RETURN_CUDNN_FRONTEND_ERROR_IF((attributes.attention_score_modifier != nullptr) &&
                    (attributes.alibi_mask || attributes.has_causal_like_masking() || attributes.padding_mask ||
                     attributes.left_bound.has_value()),error_code_t::GRAPH_NOT_SUPPORTED, "Attention score mod enabled and hence other subgraphs are disabled.");

        // validate basic dimension requirements
        RETURN_CUDNN_FRONTEND_ERROR_IF((d_qk % 8 != 0) || (d_v % 8 != 0),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "hidden_dim should be multiple of 8");

        RETURN_CUDNN_FRONTEND_ERROR_IF((h_q % h_k != 0) || (h_q % h_v != 0),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For group-query attention, number of heads for key and query must be a factor of number of heads for query");

        // validate options for attn_scale
        auto const& attn_scale    = attributes.inputs.find(input_names::Attn_scale);
        bool const has_attn_scale = (attn_scale != attributes.inputs.end()) && (attn_scale->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_attn_scale && attributes.attn_scale_value.has_value(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "attn_scale with tensor and value cannot be set at the same time.");

        // validate alibi requirements
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.alibi_mask && !(attributes.right_bound.has_value() && attributes.right_bound.value() == 0),
                        error_code_t::GRAPH_NOT_SUPPORTED,
                        "When alibi mask is used, diagonal_band_right_bound needs to be set to 0.");

        // validate options for bias mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && (bias_mask->second->get_data_type() == DataType_t::BOOLEAN),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Bias mask data type cannot be boolean");

        RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && detail::get_backend_version() < 8906, error_code_t::GRAPH_NOT_SUPPORTED, "Bias mask is not  supported below cudnn version  8.9.6");

        RETURN_CUDNN_FRONTEND_ERROR_IF((detail::get_backend_version() >= 8906 && detail::get_backend_version() < 90000) &&
                                       (context.get_sm_version() > 0 && context.get_sm_version() < 90), error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Post scale Bias mask is not supported below Hopper for cudnn version" + std::to_string(detail::get_backend_version()));

        // validate options for padding mask
        auto const& seq_len_q     = attributes.inputs.find(input_names::SEQ_LEN_Q);
        bool const has_seq_len_q  = (seq_len_q != attributes.inputs.end()) && (seq_len_q->second != nullptr);
        auto const& seq_len_kv    = attributes.inputs.find(input_names::SEQ_LEN_KV);
        bool const has_seq_len_kv = (seq_len_kv != attributes.inputs.end()) && (seq_len_kv->second != nullptr);

        RETURN_CUDNN_FRONTEND_ERROR_IF((attributes.padding_mask || attributes.alibi_mask || attributes.has_causal_mask_bottom_right()) && (detail::get_backend_version() < 8906),
                                         error_code_t::GRAPH_NOT_SUPPORTED,  "Only causal mask is supported in cudnn versions below 8.9.6 ");   

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.padding_mask && (!has_seq_len_q || !has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Padding mask requires seq_len_q and seq_len_kv to be set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF((!attributes.padding_mask && !attributes.attention_score_modifier) && (has_seq_len_q || has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "seq_len_q and seq_len_kv needs to be set only if padding mask is enabled.");

        // validate options for bottom right causal mask

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && (detail::get_backend_version() < 90300), error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Causal bottom right masking requires cudnn 9.3.0 and above");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && (!attributes.padding_mask) && s_q > s_kv,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask does not support max_s_q > max_s_kv. Please virtually slice the Q tensor and pass it as max_s_q == max_s_kv");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && (is_bias || attributes.alibi_mask || (is_ragged && !attributes.padding_mask) || is_dropout),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask is only supported with is_bias=False, is_alibi=False, is_dropout=False. Further is_ragged==True is only allowed when padding_mask=True.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && (detail::get_backend_version() < 90600) && ((s_q % 64 != 0) || (s_kv % 64 != 0)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask is only supported with s_q multiple of 64, and s_kv multiple of 64, for cudnn version below 9.6.0");

        //  NVTE_SBHD or NVTE_BSHD is only supported for bottom right causal mask and sliding window

        // Combination of mask and bias
        RETURN_CUDNN_FRONTEND_ERROR_IF((is_bias && (attributes.has_causal_like_masking() || attributes.padding_mask) && (detail::get_backend_version() < 8906)), error_code_t::GRAPH_NOT_SUPPORTED,
                        "Bias + padding or causal mask is only supported in 8.9.6 and above");

        // validate options for sliding window length
        RETURN_CUDNN_FRONTEND_ERROR_IF((attributes.left_bound.has_value() && detail::get_backend_version() < 90200), error_code_t::GRAPH_NOT_SUPPORTED,
                                        "sliding window is only supported 9.2.0 and above");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.left_bound.has_value() && attributes.left_bound.value() <= 0 && detail::get_backend_version() < 91000,
                                       error_code_t::INVALID_VALUE,
                                       "Left bound (Sliding window length) should be greater than zero when set.");
                                       
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.left_bound.has_value() && (!attributes.padding_mask) && s_q > s_kv,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Sliding window attention is only supported with max_s_q <= max_s_kv.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.left_bound.has_value() && (s_q * attributes.left_bound.value() == s_kv * attributes.left_bound.value()) && (detail::get_backend_version() <= 90900) && (prop.major == 9) && attributes.has_causal_mask_bottom_right(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "On Hopper architecture, this specific combination of s_q, s_kv, and left_bound + right_bound + bottom right diagonal alignment is not supported for backend version 9.9 or below");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.left_bound.has_value()&& (!attributes.has_causal_like_masking() || is_dropout || is_bias || (is_ragged && !attributes.padding_mask)),
                                    error_code_t::GRAPH_NOT_SUPPORTED,
                                    "Left and right bounds are only supported with is_dropout=False, is_bias=False. Further is_ragged==True is only allowed when padding_mask=True. Lastly the diagonal alignment must be set.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.right_bound.has_value() && attributes.right_bound.value() < 0,
                                       error_code_t::INVALID_VALUE,
                                       "Right bound needs to be larger than or equal to zero");

        // validate options for dropout mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.dropout_probability.has_value() && is_dropout_custom,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Using both, custom dropout mask and internal-mask generation using dropout probability, is ill-formed.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.dropout_probability.has_value() && attributes.dropout_probability.value() == 1.0,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Dropout probability cannot be 1 as corresponding scale wont be well formed.");

        // Validate options for s_q == 1
        const bool is_decode_only = (s_q == 1);
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_decode_only && (prop.major == 10) && (d_qk > 128 || d_v > 128) && (detail::get_backend_version() <= 90900), error_code_t::GRAPH_NOT_SUPPORTED, "decode only mode, i.e. s_q == 1 not supported for blackwell architecture with d_qk or d_v > 128 for backend version 9.9 or below");
        
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_decode_only && (detail::get_backend_version() <= 90900) && (attributes.right_bound.has_value()), error_code_t::GRAPH_NOT_SUPPORTED, "decode only mode, i.e. s_q == 1, not supported with masking (right_bound is set) for backend version 9.9 or below");
        
        // validate options for paged attention
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_paged && (d_qk > 128 || d_v > 128) && detail::get_backend_version() <= 90900, error_code_t::GRAPH_NOT_SUPPORTED, "Paged attention only supported with d_qk and d_v <= 128 for backend version 9.9 or below");
        
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_paged && is_ragged && detail::get_backend_version() < 90700,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Paged caches are not supported in combination with ragged offsets.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(is_paged && (!has_seq_len_q || !has_seq_len_kv),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Paged caches can only be used in combination with padding mask and variable sequence lengths for both Q and KV.");

       RETURN_CUDNN_FRONTEND_ERROR_IF(!is_paged && max_seq_kv_explicit,
            error_code_t::GRAPH_NOT_SUPPORTED, "When not using paged attention, there is no need to explicitly set max kv sequence length.");
        
        if (max_seq_kv_explicit){
           auto max_seq_kv = attributes.max_seq_len_kv.value();
           
           RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && (bias_mask->second->get_dim()[3] != max_seq_kv),
            error_code_t::GRAPH_NOT_SUPPORTED, "Value set through set_paged_attention_max_seq_len_kv is incompatible with the sequence length of the bias");

           RETURN_CUDNN_FRONTEND_ERROR_IF(is_rng &&
                    rng_tensor->second->get_dim()[3] != max_seq_kv,
            error_code_t::GRAPH_NOT_SUPPORTED, "Value set through set_paged_attention_max_seq_len_kv is incompatible with the sequence length of the RNG_DUMP");
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            ((is_paged_k() && attributes.inputs.at(input_names::Page_table_K)->get_ragged_offset()) 
         || (is_paged_v() && attributes.inputs.at(input_names::Page_table_V)->get_ragged_offset())) &&
            detail::get_backend_version() < 91002,
            error_code_t::GRAPH_NOT_SUPPORTED, "Paged attention with packed page tables only supported with cudnn version 9.10.2 and above");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 8903, error_code_t::GRAPH_NOT_SUPPORTED,
                                        "SDPA OP requires cudnn version 8.9.3 and above");

        // If user has set sm_version allow SM specific checks
        if (context.get_sm_version() > 0) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(80 > context.get_sm_version(), error_code_t::GRAPH_NOT_SUPPORTED,
                                        "cudnn SDPA operation requires Ampere and above");
        }
 
        // (cudnn_runtime_version < 8907 && num_attn_heads == num_gqa_groups FIXME

        // version specific validation
        if(prop.major == 8) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() <= 90900 && ((d_qk > 128) || (d_v > 128)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "head_dim should be less than or equal to 128 for backend version 9.9 or below on ampere architecture");
        }
        if(prop.major == 9) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() <= 90900 && ((d_qk > 256) || (d_v > 256)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "head_dim should be less than or equal to 256 for backend version 9.9 or below on hopper architecture");
        }
        if(prop.major == 10) {
            RETURN_CUDNN_FRONTEND_ERROR_IF((detail::get_backend_version() < 90900) && ((d_qk > 128) || (d_v > 128)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "head_dim should be less than or equal to 128 for backend version 9.8 or below on blackwell architecture");
        }


        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 8906 && ((s_kv % 64 != 0) || (d_qk % 64 != 0)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 8.9.6, s_kv not a multiple of 64 or d not a multiple of 64 is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 8907 && (s_kv % 64 != 0) && (!(attributes.padding_mask)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 8.9.7, s_kv not a multiple of 64 is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90000 && ((s_q % 64 != 0) || (s_kv % 64 != 0)) && (attributes.padding_mask || is_dropout),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.0.0, s_q/s_kv not a multiple of 64 with padding/dropout mask is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90200 && attributes.left_bound.has_value(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.2.0, sliding window attention is not supported");
        
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90500 && is_paged,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.5.0, paged caches are not supported");

        if (is_ragged) {
            RETURN_CUDNN_FRONTEND_ERROR_IF((context.get_sm_version() > 0  && context.get_sm_version() < 90), error_code_t::GRAPH_NOT_SUPPORTED, "THD (ragged offset) is only supported in Hopper and above");
        }
        // TODO add version check once fixed
        RETURN_CUDNN_FRONTEND_ERROR_IF(prop.major == 10 && is_rng,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "dropout RNG dump is not supported for Blackwell architecture");

        // validate that datatype is set for the graph
        RETURN_CUDNN_FRONTEND_ERROR_IF(context.get_intermediate_data_type() == DataType_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Intermediate tensor data type needs to be set as internal tensors require it.");
        // clang-format on
        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        if (attributes.is_inference.value() == false) {
            auto stats     = attributes.outputs.at(output_names::Stats);
            auto stats_dim = stats->get_dim();

            if (stats_dim.empty()) {
                // Fill properties of virtual tensors
                auto const& p_dim = attributes.inputs[input_names::Q]->get_dim();
                auto b            = p_dim[0];
                auto h            = p_dim[1];
                auto s_q          = p_dim[2];
                stats->set_dim({b, h, s_q, 1}).set_stride({h * s_q, s_q, 1, 1});
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    expand_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferrencing properties for Scaled_dot_product_flash_attention node  "
                                << attributes.name << "...");

        // DO NOT REMOVE
        // input data type is needed for:
        // - aType of bmm2
        // - dropout scale in pre 8.9.3
        attributes.fill_from_context(context);

        // Gather dim to fill properties of virtual tensors
        auto const& q_dim = attributes.inputs[input_names::Q]->get_dim();
        auto b            = q_dim[0];
        auto h_q          = q_dim[1];
        auto s_q          = q_dim[2];
        auto d_qk         = q_dim[3];
        auto const& k_dim = attributes.inputs[input_names::K]->get_dim();
        auto h_k          = k_dim[1];
        auto const& v_dim = attributes.inputs[input_names::V]->get_dim();
        auto h_v          = v_dim[1];
        auto d_v          = v_dim[3];
        // Infer s_kv
        int64_t s_kv = infer_s_kv();

        std::shared_ptr<Tensor_attributes> k_cache;
        if (!is_paged_k()) {
            // 1. map K->KT
            // cuDNN frontend API attention requires Q, K, V where
            // Q = {b, h_q, s_q, d_qk}
            // K = {b, h_k, s_kv, d_qk}
            // V = {b, h_v, s_kv, d_v}
            // but cuDNN backend API attention requires Q, KT, V
            // Q = {b, h_q, s_q, d_qk}
            // KT = {b, h_k, d_qk, s_kv}
            // V = {b, h_v, s_kv, d_v}
            // So the code below maps the K->KT
            std::vector<int64_t> temp_vec;

            temp_vec = attributes.inputs[input_names::K]->get_dim();
            std::swap(temp_vec[2], temp_vec[3]);
            attributes.inputs[input_names::K]->set_dim(temp_vec);

            temp_vec = attributes.inputs[input_names::K]->get_stride();
            std::swap(temp_vec[2], temp_vec[3]);
            attributes.inputs[input_names::K]->set_stride(temp_vec);

            // 2. Set k_cache
            k_cache = attributes.inputs[input_names::K];
        } else {
            // Create a paged cache load operation
            auto paged_cache_load_attributes_k = PagedCacheLoad_attributes().set_name("paged_k_cache_operation");
            // Need to create virtual tensor descriptor for yOut here as it cannot be inferred
            // K-cache has BHDS layout
            k_cache = std::make_shared<Tensor_attributes>();
            k_cache->set_is_virtual(true);
            k_cache->set_dim({b, h_k, d_qk, s_kv});
            k_cache->set_stride({d_qk * s_kv * h_k, d_qk * s_kv, 1, d_qk});
            k_cache->set_data_type(attributes.inputs[input_names::K]->get_data_type());
            paged_cache_load(attributes.inputs[input_names::K],
                             attributes.inputs[input_names::SEQ_LEN_KV],
                             attributes.inputs[input_names::Page_table_K],
                             paged_cache_load_attributes_k,
                             k_cache);
        }

        std::shared_ptr<Tensor_attributes> last_output;

        auto bmm1_attributes = Matmul_attributes()
                                   .set_name("bmm1")
                                   .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                                   .set_n_override(attributes.inputs[input_names::SEQ_LEN_KV]);

        if (attributes.padding_mask) {
            bmm1_attributes.set_padding(0.0);
        }

        auto const& bmm1_output = matmul(attributes.inputs[input_names::Q], k_cache, bmm1_attributes);
        // Setting dim and strides as pointwise op wont have knowledge of how to do it for mha.
        bmm1_output->set_dim({b, h_q, s_q, s_kv}).set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});
        last_output = bmm1_output;

        // Optional scale
        if (attributes.attn_scale_value.has_value()) {
            attributes.inputs[input_names::Attn_scale] =
                std::make_shared<Tensor_attributes>(attributes.attn_scale_value.value());
        }
        if (attributes.inputs[input_names::Attn_scale]) {
            Pointwise_attributes scale_attributes;
            scale_attributes.set_name("attn_scale").set_mode(PointwiseMode_t::MUL);
            auto const& attn_scale_output =
                pointwise(last_output, attributes.inputs[input_names::Attn_scale], scale_attributes);
            last_output = attn_scale_output;
        }

        if (attributes.attention_score_modifier != nullptr) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output                  = attributes.attention_score_modifier(graph_, last_output);
            sub_nodes.emplace_back(node_);
        }

        // Optional bias
        if (attributes.inputs.find(input_names::Bias) != attributes.inputs.end() &&
            attributes.inputs[input_names::Bias]) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output = attn::score_modifiers::bias(graph_, last_output, attributes.inputs[input_names::Bias]);
            sub_nodes.emplace_back(node_);
        }

        if (attributes.alibi_mask) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output = attn::score_modifiers::alibi_mask(graph_, last_output, alibi_slopes, h_q, alibi_slopes_size);
            sub_nodes.emplace_back(node_);
        }

        // There are two cases of applying padding mask
        // 1. when actual seq_len is less than or equal to max_seq_len
        if (attributes.padding_mask) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output                  = attn::score_modifiers::padding_mask(graph_,
                                                              last_output,
                                                              attributes.inputs[input_names::SEQ_LEN_KV],
                                                              attributes.inputs[input_names::SEQ_LEN_Q]);
            sub_nodes.emplace_back(node_);
        }

        // 2. (bug in cudnn backend) no padding with max_seq_len%64!=0
        if ((s_kv % 64 != 0) && (!(attributes.padding_mask)) && (detail::get_backend_version() < 90000)) {
            auto col_index_attributes =
                Pointwise_attributes().set_name("gen_col_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
            auto col_index_output = pointwise(last_output, col_index_attributes);
            // scalar seq_kv only needs to be passed in case there in no padding mask and seq_kv is not multiple of 64.
            // Also future versions of cudnn will not need it, hence tensor is pre-fixed with WAR.
            auto WAR_scalar_max_seq_kv = std::make_shared<Tensor_attributes>(static_cast<int32_t>(s_kv));

            auto col_less_seq_kv_attributes =
                Pointwise_attributes().set_name("col_less_seq_kv").set_mode(PointwiseMode_t::CMP_LT);
            auto col_less_seq_kv_output =
                pointwise(col_index_output, WAR_scalar_max_seq_kv, col_less_seq_kv_attributes);

            // Lower attributes to binary select attributes
            auto negative_inf_padding =
                std::make_shared<Tensor_attributes>(attn::score_modifiers::get_negative_inf_value());
            auto binary_select_attributes =
                Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT);
            auto padding_mask_output =
                pointwise(last_output, negative_inf_padding, col_less_seq_kv_output, binary_select_attributes);
            last_output = padding_mask_output;
        }

        // Apply (bottom-right) causal masking (with right bound) and/or set the left bound
        if (attributes.left_bound.has_value() || attributes.right_bound.has_value()) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;

            auto s_kv_ptr = attributes.inputs.find(input_names::SEQ_LEN_KV) != attributes.inputs.end()
                                ? attributes.inputs[input_names::SEQ_LEN_KV]
                                : nullptr;
            auto s_q_ptr  = attributes.inputs.find(input_names::SEQ_LEN_Q) != attributes.inputs.end()
                                ? attributes.inputs[input_names::SEQ_LEN_Q]
                                : nullptr;

            last_output = attn::score_modifiers::sliding_window_mask(graph_,
                                                                     last_output,
                                                                     attributes.diagonal_alignment,
                                                                     attributes.left_bound,
                                                                     attributes.right_bound,
                                                                     s_q,
                                                                     s_kv,
                                                                     s_q_ptr,
                                                                     s_kv_ptr);
            sub_nodes.emplace_back(node_);
        }

        // Lower attributes to softmax attributes
        auto softmax_output = std::make_shared<Tensor_attributes>();
        softmax_output->set_is_virtual(true);

        // Create a virtual output for stats if inference step otherwise output.Stats is already set
        auto softmax_stats = attributes.outputs[output_names::Stats];
        if (attributes.is_inference.value() == true) {
            softmax_stats = std::make_shared<Tensor_attributes>();
            softmax_stats->set_is_virtual(true);
        }

        auto softmax_attributes =
            Softmax_attributes().set_name("softmax").has_stats(true).has_M_Zinv(false);  // As this is flash attention
        // Special non-functional-style call. Needed because output already created and provided to user.
        softmax(last_output, softmax_attributes, softmax_output, softmax_stats);
        last_output = softmax_output;

        // Two cases for training: dropout present or not
        bool dropout_present         = false;
        auto const& dropout_mask     = attributes.inputs.find(input_names::Dropout_mask);
        bool const is_dropout_custom = (dropout_mask != attributes.inputs.end()) && (dropout_mask->second != nullptr);
        if (attributes.dropout_probability.has_value()) {
            dropout_present = true;
            // Special case: Skip dropout when 0.0 probability. Only do for 8.9.3 and up as rng isn't optional earlier.
            if (detail::get_backend_version() > 8902 && attributes.dropout_probability.value() == 0.0) {
                dropout_present = false;
            }
        } else if (is_dropout_custom) {
            dropout_present = true;
        }

        if (dropout_present) {
            if (is_dropout_custom) {
                auto dropout_scale_attributes =
                    Pointwise_attributes().set_name("dropout_scale_mul").set_mode(PointwiseMode_t::MUL);
                auto const& dropout_scale_output =
                    pointwise(last_output, attributes.inputs[input_names::Dropout_scale], dropout_scale_attributes);

                auto mask_attributes =
                    Pointwise_attributes().set_name("dropout_mask_mul").set_mode(PointwiseMode_t::MUL);
                auto const& dropout_mask_output =
                    pointwise(dropout_scale_output, dropout_mask->second, mask_attributes);
                last_output = dropout_mask_output;
            } else {
                if (attributes.outputs[output_names::RNG_DUMP] != nullptr) {
                    rng_output = attributes.outputs[output_names::RNG_DUMP];
                    rng(attributes.inputs[input_names::Seed],
                        attributes.inputs[input_names::Offset],
                        Rng_attributes()
                            .set_name("rng")
                            .set_distribution(RngDistribution_t::BERNOULLI)
                            .set_bernoulli_probability(1.0 - attributes.dropout_probability.value()),
                        rng_output);
                } else {
                    rng_output = rng(attributes.inputs[input_names::Seed],
                                     attributes.inputs[input_names::Offset],
                                     Rng_attributes()
                                         .set_name("rng")
                                         .set_distribution(RngDistribution_t::BERNOULLI)
                                         .set_bernoulli_probability(1.0 - attributes.dropout_probability.value()));
                    rng_output
                        // Hard coding dim and strides as rng output can no inputs to infer it from.
                        ->set_dim({b, h_q, s_q, s_kv})
                        .set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});
                }

                auto mask_attributes =
                    Pointwise_attributes().set_name("dropout_mask_mul").set_mode(PointwiseMode_t::MUL);
                auto const& dropout_mask_output = pointwise(last_output, rng_output, mask_attributes);
                last_output                     = dropout_mask_output;

                std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> dropout_scale = nullptr;

                if (detail::get_backend_version() < 8903) {
                    half dropout_scale_value = __float2half(1.0f / (1.0f - attributes.dropout_probability.value()));
                    dropout_scale            = std::make_shared<Tensor_attributes>(dropout_scale_value);
                } else {
                    float dropout_scale_value = (1.0f / (1.0f - attributes.dropout_probability.value()));
                    dropout_scale             = std::make_shared<Tensor_attributes>(dropout_scale_value);
                }

                auto dropout_scale_attributes =
                    Pointwise_attributes().set_name("dropout_scale").set_mode(PointwiseMode_t::MUL);
                auto const& dropout_scale_output = pointwise(last_output, dropout_scale, dropout_scale_attributes);
                last_output                      = dropout_scale_output;
            }
        }

        // Lower attributes to bmm2 attributes
        // Requirement by cudnn backend to take in bmm2 aType as i/o type.
        last_output->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

        auto const& seq_len_q  = attributes.inputs[input_names::SEQ_LEN_Q];
        auto const& seq_len_kv = attributes.inputs[input_names::SEQ_LEN_KV];
        // auto const& V          = attributes.inputs[input_names::V];
        auto const& O = attributes.outputs[output_names::O];

        std::shared_ptr<Tensor_attributes> v_cache;

        if (!is_paged_v()) {
            v_cache = attributes.inputs[input_names::V];
        } else {
            auto paged_cache_load_attributes_v = PagedCacheLoad_attributes().set_name("paged_v_cache_operation");
            v_cache                            = std::make_shared<Tensor_attributes>();
            v_cache->set_dim({b, h_v, s_kv, d_v})
                .set_stride({d_v * s_kv * h_v, d_v * s_kv, d_v, 1})
                .set_data_type(attributes.inputs[input_names::V]->get_data_type());
            v_cache->set_is_virtual(true);
            paged_cache_load(attributes.inputs[input_names::V],
                             attributes.inputs[input_names::SEQ_LEN_KV],
                             attributes.inputs[input_names::Page_table_V],
                             paged_cache_load_attributes_v,
                             v_cache);
        }

        auto bmm2_attributes =
            Matmul_attributes().set_name("bmm2").set_m_override(seq_len_q).set_k_override(seq_len_kv);
        // Special non-functional-style call. Needed because output already created and provided to user.
        matmul(last_output, v_cache, bmm2_attributes, O);

        return {error_code_t::OK, ""};
    }

    error_t
    post_validate_node() const override final {
#define CUDNN_FE_VALIDATE_STRIDE(port, port_map)                                                                \
    {                                                                                                           \
        auto const& t = port_map.find(port);                                                                    \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                         \
            t->second->get_stride().back() != 1,                                                                \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                  \
            "The stride for the last dimension corresponding to the embedding size per head should be 1 for " + \
                std::string(#port));                                                                            \
    }

        CUDNN_FE_VALIDATE_STRIDE(output_names::O, attributes.outputs);

#undef CUDNN_FE_VALIDATE_STRIDE

        return {error_code_t::OK, ""};
    }

    virtual int64_t
    get_fe_workspace_size_node() const override final {
        int64_t size = 0;

        // align alibi slopes memory to 16 bytes
        size += ((alibi_slopes_size + 15) / 16 * 16);

        return size;
    }

    virtual error_t
    collect_tensors_in_workspace_node(
        std::unordered_map<Tensor_attributes::uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>&
            workspace_modifications,
        int64_t& offset) const override final {
        if (attributes.alibi_mask) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(Q, input_names::Q);
            int64_t const h_q     = Q->second->get_dim()[1];
            auto alibi_slopes_vec = detail::get_abili_slope(h_q);
            workspace_modifications.emplace(alibi_slopes->get_uid(), std::make_tuple(0, offset, alibi_slopes_vec));
            int64_t alibi_slopes_size_padded = ((alibi_slopes_size + 15) / 16 * 16);
            offset                           = offset + alibi_slopes_size_padded;
        }
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "SDPA_FWD"})"_json);
    }
#endif
};

class SDPABackwardNode : public NodeCRTP<SDPABackwardNode> {
    using input_names  = SDPA_backward_attributes::input_names;
    using output_names = SDPA_backward_attributes::output_names;

   private:
    // non-virtual node gpu tensors
    std::shared_ptr<Tensor_attributes> dQ_accum;
    int64_t dQ_accum_size = 0;
    std::shared_ptr<Tensor_attributes> dK_fullhead;
    int64_t dK_fullhead_size = 0;
    std::shared_ptr<Tensor_attributes> dV_fullhead;
    int64_t dV_fullhead_size = 0;
    std::shared_ptr<Tensor_attributes> softmax_sum;
    int64_t softmax_sum_size = 0;
    std::shared_ptr<Tensor_attributes> alibi_slopes;
    int64_t alibi_slopes_size = 0;

   public:
    SDPA_backward_attributes attributes;

    SDPABackwardNode(SDPA_backward_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating SDPABackwardNode" << attributes.name << "...");

        // check that Q, K, V, O, stats, dO, dQ, dK, dV tensors has been assigned
        // check that dim and strides has been assigned and last stride is 1
#define CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(port, port_map)                                                        \
    {                                                                                                            \
        std::shared_ptr<Tensor_attributes> tensor_ptr = port_map.at(port);                                       \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_dim().size() != 4,                                        \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                          \
                                       "The dim for " + std::string(#port) + " is invalid");                     \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_stride().size() != 4,                                     \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                          \
                                       "The stride for " + std::string(#port) + " is invalid");                  \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                          \
            tensor_ptr->get_stride()[3] != 1,                                                                    \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                   \
            "The stride for the last dimension corresponding to the embedding size per head should be 1 for " +  \
                std::string(#port));                                                                             \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                          \
            tensor_ptr->get_stride()[2] == 0,                                                                    \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                   \
            "The stride for the dimension corresponding to the sequence lengths per head should not be 0 for " + \
                std::string(#port));                                                                             \
    }

        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::Q, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::K, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::V, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::O, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::Stats, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::dO, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(output_names::dQ, attributes.outputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(output_names::dK, attributes.outputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(output_names::dV, attributes.outputs);

#undef CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE

        // validate backend limitations for the operation
        // clang-format off
        int64_t s_q  = attributes.inputs.at(input_names::Q)->get_dim()[2];
        int64_t s_kv = attributes.inputs.at(input_names::V)->get_dim()[2];
        int64_t h_q  = attributes.inputs.at(input_names::Q)->get_dim()[1];
        int64_t h_k  = attributes.inputs.at(input_names::K)->get_dim()[1];
        int64_t h_v  = attributes.inputs.at(input_names::V)->get_dim()[1];
        int64_t d_qk = attributes.inputs.at(input_names::Q)->get_dim()[3];
        int64_t d_v  = attributes.inputs.at(input_names::V)->get_dim()[3];

        bool const is_ragged = attributes.inputs.at(input_names::Q)->get_ragged_offset() ||
                               attributes.inputs.at(input_names::K)->get_ragged_offset() ||
                               attributes.inputs.at(input_names::V)->get_ragged_offset() ||
                               attributes.inputs.at(input_names::O)->get_ragged_offset();

        auto const& bias_mask = attributes.inputs.find(input_names::Bias);
        bool const is_bias   = (bias_mask != attributes.inputs.end() && bias_mask->second != nullptr);
        auto const& dbias_mask = attributes.outputs.find(output_names::dBias);
        bool const is_dbias   = (dbias_mask != attributes.outputs.end() && dbias_mask->second != nullptr);

        auto const& dropout_mask     = attributes.inputs.find(input_names::Dropout_mask);
        bool const is_dropout_custom = (dropout_mask != attributes.inputs.end()) && (dropout_mask->second != nullptr);
        bool const is_dropout        = attributes.dropout_probability.has_value() || is_dropout_custom;

        auto const& rng_tensor = attributes.outputs.find(output_names::RNG_DUMP);
        bool const is_rng   = (rng_tensor != attributes.outputs.end() && rng_tensor->second != nullptr);

        // validation TODO:
        //    - validate stats has valid dims
        //    - validate Q and dQ have the same dims

        cudaDeviceProp prop;
        int device;
        CHECK_CUDA_ERROR(detail::cuda_get_device(&device));
        CHECK_CUDA_ERROR(detail::cuda_get_device_properties(&prop, device));

        if (prop.major == 9) { 
            // validate basic dimension hquirements
            RETURN_CUDNN_FRONTEND_ERROR_IF((d_qk > 256) || (d_qk % 8 != 0) || (d_v > 256) || (d_v % 8 != 0),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Num hidden_dim should be less than or equal to 256 and hidden_dim should be multiple of 8");
        } else if (prop.major == 10 && detail::get_backend_version() >= 91100) {
            // validate basic dimension requirements
            RETURN_CUDNN_FRONTEND_ERROR_IF((d_qk % 8 != 0) || (d_v % 8 != 0),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Num hidden_dim should be multiple of 8");
        } else {
            // validate basic dimension requirements
            RETURN_CUDNN_FRONTEND_ERROR_IF((d_qk > 128) || (d_qk % 8 != 0) || (d_v > 128) || (d_v % 8 != 0),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Num hidden_dim should be less than or equal to 128 and hidden_dim should be multiple of 8");
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF((attributes.attention_score_modifier != nullptr) &&
                    (attributes.alibi_mask || attributes.padding_mask || attributes.has_causal_like_masking() ||
                     attributes.left_bound.has_value()), error_code_t::GRAPH_NOT_SUPPORTED,"Attention score mod enabled and hence other subgraphs are disabled.");

        RETURN_CUDNN_FRONTEND_ERROR_IF((h_q % h_k != 0) || (h_q % h_v != 0),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For group-query attention, number of heads for key and query must be a factor of number of heads for query");

        // validate options for attn_scale
        auto const& attn_scale    = attributes.inputs.find(input_names::Attn_scale);
        bool const has_attn_scale = (attn_scale != attributes.inputs.end()) && (attn_scale->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_attn_scale && attributes.attn_scale_value.has_value(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "attn_scale with tensor and value cannot be set at the same time.");

        // validate alibi requirements
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.alibi_mask && !(attributes.right_bound.has_value() && attributes.right_bound.value() == 0),
                        error_code_t::GRAPH_NOT_SUPPORTED,
                        "When alibi mask is used, diagonal_band_right_bound needs to be set to 0.");

        // validate options for bias mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && (bias_mask->second->get_data_type() == DataType_t::BOOLEAN),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Bias mask data type cannot be boolean");

        // validate options for padding mask
        auto const& seq_len_q     = attributes.inputs.find(input_names::SEQ_LEN_Q);
        bool const has_seq_len_q  = (seq_len_q != attributes.inputs.end()) && (seq_len_q->second != nullptr);
        auto const& seq_len_kv    = attributes.inputs.find(input_names::SEQ_LEN_KV);
        bool const has_seq_len_kv = (seq_len_kv != attributes.inputs.end()) && (seq_len_kv->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.padding_mask && (!has_seq_len_q || !has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Padding mask requires seq_len_q and seq_len_kv to be set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF((!attributes.padding_mask && !attributes.attention_score_modifier) && (has_seq_len_q || has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "seq_len_q and seq_len_kv needs to be set only if padding mask is enabled.");

        // validate options for max_total_seq_len
        RETURN_CUDNN_FRONTEND_ERROR_IF((attributes.max_total_seq_len_q.has_value() || attributes.max_total_seq_len_kv.has_value()) && !is_ragged,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "max_total_seq_len_q is only supported with packed layout");

        // validate options for bottom right causal mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && (!attributes.padding_mask) && s_q > s_kv,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask does not support max_s_q > max_s_kv. Please virtually slice the Q tensor and pass it as max_s_q == max_s_kv");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && (is_bias || attributes.alibi_mask || (is_ragged && !attributes.padding_mask) || is_dropout),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask is only supported with is_bias=False, is_alibi=False, is_dropout=False. Further is_ragged==True is only allowed when padding_mask=True.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && (detail::get_backend_version() < 90600) && ((s_q % 64 != 0) || (s_kv % 64 != 0)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask is only supported with s_q multiple of 64, and s_kv multiple of 64, for cudnn version below 9.6.0");

        // validate options for sliding window length
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.left_bound.has_value() && attributes.left_bound.value() <= 0,
                                       error_code_t::INVALID_VALUE,
                                       "Left bound (Sliding window length) should be greater than or equals to zero when set.");

         RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.left_bound.has_value() && (s_q * attributes.left_bound.value() == s_kv * attributes.left_bound.value()) && (detail::get_backend_version() <= 90900) && (prop.major == 9) && attributes.has_causal_mask_bottom_right(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "On Hopper architecture, this specific combination of s_q, s_kv, and left_bound + right_bound + bottom right diagonal alignment is not supported for backend version 9.9 or below");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.left_bound.has_value() && (!attributes.padding_mask) && s_q > s_kv,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Sliding window attention is only supported with max_s_q <= max_s_kv.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.left_bound.has_value() && (! attributes.has_causal_like_masking() || is_dropout || is_bias || (is_ragged && !attributes.padding_mask)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Left and right bounds are only supported with is_dropout=False, is_bias=False. Further is_ragged==True is only allowed when padding_mask=True. Lastly the diagonal alignment must be set.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.right_bound.has_value() && attributes.right_bound.value() < 0,
                                       error_code_t::INVALID_VALUE,
                                       "Right bound needs to be larger than or equal to zero");

        // validate options for dropout mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.dropout_probability.has_value() && is_dropout_custom,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Using both, custom dropout mask and internal-mask generation using dropout probability, is ill-formed.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.dropout_probability.has_value() && attributes.dropout_probability.value() == 1.0,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Dropout probability cannot be 1 as corresponding scale wont be well formed.");

        // version specific validation
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 8906 && ((s_kv % 64 != 0) || (d_qk % 64 != 0)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 8.9.6, s_kv not a multiple of 64 or d not a multiple of 64 is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 8907 && (s_kv % 64 != 0) && (!(attributes.padding_mask)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 8.9.7, s_kv not a multiple of 64 is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90000 && ((s_q % 64 != 0) || (s_kv % 64 != 0)) && (attributes.padding_mask || is_dropout),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.0.0, s_q/s_kv not a multiple of 64 with padding/dropout mask is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90000 && (s_q < 64),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
            "                          Sequence length must be greater than or equal to 64 for cudnn version prior to v9.0.0");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90200 && attributes.left_bound.has_value(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.2.0, sliding window attention is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90500 && is_dbias && attributes.padding_mask,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.5.0, dBias with variable sequence lengths is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90500 && is_dbias && ((s_q % 64 != 0) || (s_kv % 64 != 0)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.5.0, dBias not support s_q/s_kv which aren't multiple of 64");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90600 && is_ragged && ((h_q != h_k) || (h_q != h_v)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.6.0, group-query attention with raggged offset is not supported");

        // TODO add version check once fixed
        RETURN_CUDNN_FRONTEND_ERROR_IF(prop.major == 10 && is_rng,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Dropout RNG dump is not supported for SM Major version 10");

        // TODO add version check once fixed
        RETURN_CUDNN_FRONTEND_ERROR_IF(prop.major == 10 && is_ragged && (is_dbias || attributes.is_deterministic_algorithm),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Deterministic kernel or dbias with ragged is not supported for SM Major version 10");

        // validate that datatype is set for the graph
        RETURN_CUDNN_FRONTEND_ERROR_IF(context.get_intermediate_data_type() == DataType_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Intermediate tensor data type needs to be set as internal tensors require it.");
        // clang-format on

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        // clang-format off
        if (detail::get_backend_version() < 90600 && (attributes.max_total_seq_len_q.has_value() || attributes.max_total_seq_len_kv.has_value())) {
            CUDNN_FE_LOG_LABEL_ENDL("WARNING: sdpa_backward.attributes.max_total_seq_len has been set, but cuDNN version is below 9.6.0 does not support max_total_seq_len_q. The workspace memory size required to execute this graph may be unexpectedly large");
            attributes.max_total_seq_len_q.reset();
            attributes.max_total_seq_len_kv.reset();
        }

        // TODO add version check once fixed
        int64_t d_qk = attributes.inputs.at(input_names::Q)->get_dim()[3];
        int64_t d_v  = attributes.inputs.at(input_names::V)->get_dim()[3];
        if ((attributes.max_total_seq_len_q.has_value() || attributes.max_total_seq_len_kv.has_value()) && (d_qk % 16 != 0 || d_v % 16 != 0)) {
            CUDNN_FE_LOG_LABEL_ENDL("WARNING: sdpa_backward.attributes.max_total_seq_len has been set, but d is not a multiple of 16 has a known functional issue. The workspace memory size required to execute this graph may be unexpectedly large");
            attributes.max_total_seq_len_q.reset();
            attributes.max_total_seq_len_kv.reset();
        }
        // clang-format on

        return {error_code_t::OK, ""};
    }

    error_t
    expand_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferrencing properties for SDPABackwardNode " << attributes.name);

        attributes.fill_from_context(context);

        // Gather dim to fill properties of virtual tensors
        auto const& q_dim = attributes.inputs[input_names::Q]->get_dim();
        auto b            = q_dim[0];
        auto h_q          = q_dim[1];
        auto s_q          = q_dim[2];
        auto d_qk         = q_dim[3];
        auto const& k_dim = attributes.inputs[input_names::K]->get_dim();
        auto h_k          = k_dim[1];
        auto s_kv         = k_dim[2];
        auto const& v_dim = attributes.inputs[input_names::V]->get_dim();
        auto h_v          = v_dim[1];
        auto d_v          = v_dim[3];

        // cuDNN frontend API attention requires Q, K, V where
        // Q = {b, h_q, s_q, d_qk}
        // K = {b, h_k, s_kv, d_qk}
        // V = {b, h_v, s_kv, d_v}
        // but cuDNN backend API attention requires Q, KT, VT
        // Q = {b, h_q, s_q, d_qk}
        // KT = {b, h_k, d_qk, s_kv}
        // VT = {b, h_v, d_v, s_kv}
        // So the code below maps the K->KT and V->VT
        std::vector<int64_t> temp_vec;

        temp_vec = attributes.inputs[input_names::K]->get_dim();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::K]->set_dim(temp_vec);

        temp_vec = attributes.inputs[input_names::K]->get_stride();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::K]->set_stride(temp_vec);

        temp_vec = attributes.inputs[input_names::V]->get_dim();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::V]->set_dim(temp_vec);

        temp_vec = attributes.inputs[input_names::V]->get_stride();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::V]->set_stride(temp_vec);

        std::shared_ptr<Tensor_attributes> last_output, exp_s_output, dS_output, rng_output;

        // --------------Initialize and create tensors before creating nodes--------------------
        // one_tensor is needed for non-dropout graphs
        // one_tensor is passed by the node
        auto one_tensor = std::make_shared<Tensor_attributes>(1.0f);

        if (attributes.attn_scale_value.has_value()) {
            attributes.inputs[input_names::Attn_scale] =
                std::make_shared<Tensor_attributes>(attributes.attn_scale_value.value());
        }

        // if dropout_mask is used, then the user passes scale and scale_inverse
        bool is_dropout_prob = (attributes.dropout_probability.has_value());
        bool is_dropout_mask = (attributes.inputs[input_names::Dropout_mask] != nullptr);
        if (is_dropout_prob) {
            float dropout_scale_value     = 1.0f / (1.0f - attributes.dropout_probability.value());
            float dropout_scale_inv_value = (1.0f - attributes.dropout_probability.value());

            attributes.inputs[input_names::Dropout_scale] = std::make_shared<Tensor_attributes>(dropout_scale_value);
            attributes.inputs[input_names::Dropout_scale_inv] =
                std::make_shared<Tensor_attributes>(dropout_scale_inv_value);
        }

        // ---------------------input tensor workarounds---------------------------

        bool use_dp_workspace = false;

        if (detail::get_backend_version() >= 8905 && detail::get_backend_version() < 90000) {
            // workspace optimization is enabled by default when:
            //   8.9.5 <= cudnn version < 9.0.0
            //   device >= hopper
            //   batch * num_heads * seq_len_q * seq_len_kv * 2 <= dP workspace limit
            //
            // This following environment variable allows you to control the dP workspace limit.
            // From cuDNN version 9.0.0, this option is obsolete will be ignored.
            // CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=unset  - enable workspace opt. until the default 256MB limit.
            // CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=-1     - always enable workspace opt.
            // CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0      - always disable workspace opt.
            // CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=n      - enable workspace opt. until the n byte limit
            struct cudaDeviceProp prop;
            CHECK_CUDA_ERROR(detail::cuda_get_device_properties(&prop, 0));

            // hopper or above
            if (prop.major >= 9) {
                // default upper limit for workspace 256MB
                int64_t max_dp_workspace_bytes = 256 * 1024 * 1024;

                // allow setting the upper limit with envvars
                char* env_dp_workspace_limit_char = std::getenv("CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT");
                if (env_dp_workspace_limit_char) {
                    char* end_ptr          = nullptr;
                    max_dp_workspace_bytes = std::strtoll(env_dp_workspace_limit_char, &end_ptr, 10);

                    if (*end_ptr != '\0') {
                        RETURN_CUDNN_FRONTEND_ERROR_IF(true,
                                                       error_code_t::ATTRIBUTE_NOT_SET,
                                                       "Invalid argument for CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT "
                                                       "(int64_t; in bytes)");
                    }
                }

                int64_t workspace_s_q               = ((s_q + 64 - 1) / 64) * 64;
                int64_t workspace_s_kv              = ((s_kv + 64 - 1) / 64) * 64;
                int64_t required_dp_workspace_bytes = b * h_q * workspace_s_q * workspace_s_kv * 2;

                if (max_dp_workspace_bytes == -1) {
                    use_dp_workspace = true;
                } else if (max_dp_workspace_bytes == 0) {
                    use_dp_workspace = false;
                } else {
                    use_dp_workspace = (required_dp_workspace_bytes <= max_dp_workspace_bytes);
                }
            }
        }

        // Force dP workspace implementation if:
        //  - dBias is enabled (dBias is only supported on workspace implementation)
        //  - the user force requests deterministic algorithm
        if (attributes.outputs[output_names::dBias] || attributes.is_deterministic_algorithm) {
            use_dp_workspace = true;
        }

        // --------------RNG node--------------------

        if (is_dropout_prob) {
            if (attributes.outputs[output_names::RNG_DUMP] != nullptr) {
                rng_output = attributes.outputs[output_names::RNG_DUMP];
                rng(attributes.inputs[input_names::Seed],
                    attributes.inputs[input_names::Offset],
                    Rng_attributes()
                        .set_name("rng")
                        .set_distribution(RngDistribution_t::BERNOULLI)
                        .set_bernoulli_probability(1.0f - attributes.dropout_probability.value()),
                    rng_output);
            } else {
                rng_output = rng(attributes.inputs[input_names::Seed],
                                 attributes.inputs[input_names::Offset],
                                 Rng_attributes()
                                     .set_name("rng")
                                     .set_distribution(RngDistribution_t::BERNOULLI)
                                     .set_bernoulli_probability(1.0f - attributes.dropout_probability.value()));
                rng_output->set_dim({b, h_q, s_q, s_kv}).set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});
            }
        } else if (is_dropout_mask) {
            rng_output = attributes.inputs[input_names::Dropout_mask];
        }

        // --------------"dO * o => softmax_sum" chain--------------------

        // last_output = dO * O
        last_output = pointwise(attributes.inputs[input_names::dO],
                                attributes.inputs[input_names::O],
                                Pointwise_attributes().set_name("mul_dO_O").set_mode(PointwiseMode_t::MUL));
        last_output->set_dim({b, h_q, s_q, d_v}).set_stride({h_q * s_q * d_v, s_q * d_v, h_q * d_v, 1});

        // last_output = reduce(last_output, "b hq sq dv -> b hq sq 1")
        last_output =
            reduction(last_output, Reduction_attributes().set_name("reduce_dO_o").set_mode(ReductionMode_t::ADD));
        last_output->set_dim({b, h_q, s_q, 1}).set_stride({h_q * s_q, s_q, 1, 1});

        // softmax_sum = last_output * dropout_scale
        last_output = pointwise(last_output,
                                attributes.inputs[input_names::Dropout_scale_inv]
                                    ? attributes.inputs[input_names::Dropout_scale_inv]
                                    : one_tensor,
                                Pointwise_attributes().set_name("scale_dropout_inv").set_mode(PointwiseMode_t::MUL));
        last_output->set_dim({b, h_q, s_q, 1}).set_stride({h_q * s_q, s_q, 1, 1});

        softmax_sum = last_output;
        softmax_sum->set_is_virtual(false);
        softmax_sum->set_dim({b, h_q, s_q, 1});
        softmax_sum->set_data_type(DataType_t::FLOAT);

        if (attributes.inputs[input_names::Stats]->get_ragged_offset() && attributes.max_total_seq_len_q.has_value()) {
            // sized TH1 softmax_sum
            softmax_sum->set_stride(attributes.inputs[input_names::Stats]->get_stride());
            softmax_sum->set_ragged_offset(attributes.inputs[input_names::Stats]->get_ragged_offset());
            softmax_sum_size = attributes.max_total_seq_len_q.value() *
                               (attributes.inputs[input_names::Stats]->get_stride())[2] * sizeof(float);
        } else {
            // sized BHS1 softmax_sum
            softmax_sum->set_stride({h_q * s_q, s_q, 1, 1});
            softmax_sum_size = b * h_q * s_q * 1 * sizeof(float);
        }

        // --------------"Q @ KT => exp_softmax => dV" chain--------------------

        // s = einsum(q, k, "b hq sq dqk, b (hk g) skv dqk -> b hq sq skv", g=hq//hk)
        last_output = matmul(attributes.inputs[input_names::Q],
                             attributes.inputs[input_names::K],
                             Matmul_attributes()
                                 .set_name("matmul_Q_KT")
                                 .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                                 .set_n_override(attributes.inputs[input_names::SEQ_LEN_KV]));
        last_output->set_dim({b, h_q, s_q, s_kv}).set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});

        // last_output = last_output * attention_scale
        if (attributes.inputs[input_names::Attn_scale]) {
            last_output = pointwise(last_output,
                                    attributes.inputs[input_names::Attn_scale],
                                    Pointwise_attributes().set_name("mul_s_attn_scale").set_mode(PointwiseMode_t::MUL));
        }

        if (attributes.attention_score_modifier != nullptr) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output                  = attributes.attention_score_modifier(graph_, last_output);
            sub_nodes.emplace_back(node_);
        }

        // (optional) last_output = last_output + bias
        if (attributes.inputs.find(input_names::Bias) != attributes.inputs.end() &&
            attributes.inputs[input_names::Bias]) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output = attn::score_modifiers::bias(graph_, last_output, attributes.inputs[input_names::Bias]);
            sub_nodes.emplace_back(node_);
        }

        // (optional) last_output = last_output + alibi_mask
        if (attributes.alibi_mask) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output = attn::score_modifiers::alibi_mask(graph_, last_output, alibi_slopes, h_q, alibi_slopes_size);
            sub_nodes.emplace_back(node_);
        }

        // (optional) Apply padding mask
        if (attributes.padding_mask) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output                  = attn::score_modifiers::padding_mask(graph_,
                                                              last_output,
                                                              attributes.inputs[input_names::SEQ_LEN_KV],
                                                              attributes.inputs[input_names::SEQ_LEN_Q]);
            sub_nodes.emplace_back(node_);
        }

        // last_output = last_output - stats
        last_output = pointwise(last_output,
                                attributes.inputs[input_names::Stats],
                                Pointwise_attributes().set_name("sub_s_m").set_mode(PointwiseMode_t::SUB));

        // WAR for bug 4475073 by explicitly putting the padding value again after the stats have been loaded
        if (attributes.padding_mask && detail::get_backend_version() >= 90000 &&
            detail::get_backend_version() < 91000) {
            auto row_idx_output = pointwise(last_output,
                                            Pointwise_attributes()
                                                .set_name("gen_row_idx_2nd_padding")
                                                .set_mode(PointwiseMode_t::GEN_INDEX)
                                                .set_axis(2)
                                                .set_compute_data_type(DataType_t::INT32));
            row_idx_output->set_data_type(DataType_t::INT32);

            auto col_idx_output = pointwise(last_output,
                                            Pointwise_attributes()
                                                .set_name("gen_col_idx_2nd_padding")
                                                .set_mode(PointwiseMode_t::GEN_INDEX)
                                                .set_axis(3)
                                                .set_compute_data_type(DataType_t::INT32));
            col_idx_output->set_data_type(DataType_t::INT32);

            auto row_mask_output = pointwise(row_idx_output,
                                             attributes.inputs[input_names::SEQ_LEN_Q],
                                             Pointwise_attributes()
                                                 .set_name("lt_row_sq_2nd_padding")
                                                 .set_mode(PointwiseMode_t::CMP_LT)
                                                 .set_compute_data_type(DataType_t::BOOLEAN));
            row_mask_output->set_data_type(DataType_t::BOOLEAN);

            auto col_mask_output = pointwise(col_idx_output,
                                             attributes.inputs[input_names::SEQ_LEN_KV],
                                             Pointwise_attributes()
                                                 .set_name("lt_col_skv_2nd_padding")
                                                 .set_mode(PointwiseMode_t::CMP_LT)
                                                 .set_compute_data_type(DataType_t::BOOLEAN));
            col_mask_output->set_data_type(DataType_t::BOOLEAN);

            auto padding_mask_output = pointwise(row_mask_output,
                                                 col_mask_output,
                                                 Pointwise_attributes()
                                                     .set_name("and_row_col_2nd_padding")
                                                     .set_mode(PointwiseMode_t::LOGICAL_AND)
                                                     .set_compute_data_type(DataType_t::BOOLEAN));
            padding_mask_output->set_data_type(DataType_t::BOOLEAN);
            auto negative_inf_padding =
                std::make_shared<Tensor_attributes>(attn::score_modifiers::get_negative_inf_value());

            last_output = pointwise(
                last_output,
                negative_inf_padding,
                padding_mask_output,
                Pointwise_attributes().set_name("select_2nd_padding").set_mode(PointwiseMode_t::BINARY_SELECT));
        }

        // Apply (bottom-right) causal masking (with right bound) and/or set the left bound
        if (attributes.left_bound.has_value() || attributes.right_bound.has_value()) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;

            auto s_kv_ptr = attributes.inputs.find(input_names::SEQ_LEN_KV) != attributes.inputs.end()
                                ? attributes.inputs[input_names::SEQ_LEN_KV]
                                : nullptr;
            auto s_q_ptr  = attributes.inputs.find(input_names::SEQ_LEN_Q) != attributes.inputs.end()
                                ? attributes.inputs[input_names::SEQ_LEN_Q]
                                : nullptr;

            last_output = attn::score_modifiers::sliding_window_mask(graph_,
                                                                     last_output,
                                                                     attributes.diagonal_alignment,
                                                                     attributes.left_bound,
                                                                     attributes.right_bound,
                                                                     s_q,
                                                                     s_kv,
                                                                     s_q_ptr,
                                                                     s_kv_ptr);
            sub_nodes.emplace_back(std::move(node_));
        }

        // last_output = exp(last_output)
        last_output = pointwise(last_output, Pointwise_attributes().set_name("exp_s").set_mode(PointwiseMode_t::EXP));

        exp_s_output = last_output;

        // (optional) last_output = last_output * dropout rng_output
        if (is_dropout_prob || is_dropout_mask) {
            last_output =
                pointwise(last_output,
                          rng_output,
                          Pointwise_attributes().set_name("mul_p_dropout_mask").set_mode(PointwiseMode_t::MUL));
        }

        // (optional) last_output = last_output * dropout_scale
        if (attributes.inputs[input_names::Dropout_scale]) {
            last_output =
                pointwise(last_output,
                          attributes.inputs[input_names::Dropout_scale],
                          Pointwise_attributes().set_name("mul_p_dropout_scale").set_mode(PointwiseMode_t::MUL));
        }

        // dV = einsum(p, dO, "b hq sq skv", "b hq sq dv -> b hq skv dv")
        // if GQA, then dV = reduce(dV, "b (hv g) skv dv -> b hv skv dv", g=hq//hv)
        // as reshape + matmul
        last_output = reshape(last_output, Reshape_attributes().set_name("reshape_p"));
        last_output->set_dim({b, h_q, s_kv, s_q}).set_stride({h_q * s_q * s_kv, s_q * s_kv, 1, s_kv});
        last_output->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

        if (h_q == h_v) {
            // for MHA
            matmul(last_output,
                   attributes.inputs[input_names::dO],
                   Matmul_attributes()
                       .set_name("matmul_pT_dO")
                       .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                       .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]),
                   attributes.outputs[output_names::dV]);
        } else {
            // for GQA and MQA
            dV_fullhead = matmul(last_output,
                                 attributes.inputs[input_names::dO],
                                 Matmul_attributes()
                                     .set_name("matmul_pT_dO")
                                     .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                                     .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]));

            dV_fullhead->set_dim({b, h_q, s_kv, d_v});
            dV_fullhead->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

            if (attributes.outputs[output_names::dV]->get_ragged_offset() &&
                attributes.max_total_seq_len_kv.has_value()) {
                // hack 1 - map dV strides to dV_fullhead strides
                std::vector<int64_t> dV_fullhead_stride = attributes.outputs[output_names::dV]->get_stride();
                dV_fullhead_stride[2]                   = dV_fullhead_stride[2] * (h_q / h_v);  // sequence stride
                dV_fullhead_stride[0]                   = dV_fullhead_stride[0] * (h_q / h_v);  // batch stride
                dV_fullhead->set_stride(dV_fullhead_stride);
                // hack 2 - map dV ragged offset to dV_fullhead ragged offset with implicit multiplier
                // implicit multiplier = h_q / h_v
                dV_fullhead->set_ragged_offset(attributes.outputs[output_names::dV]->get_ragged_offset());
                // hack 3 - non virtual dV full head
                dV_fullhead->set_is_virtual(false);
                dV_fullhead_size = attributes.max_total_seq_len_kv.value() * dV_fullhead_stride[2] * sizeof(float);
            } else {
                // sized BHSD dQ_accum
                dV_fullhead->set_stride({h_q * s_kv * d_v, s_kv * d_v, d_v, 1});
            }

            reduction(dV_fullhead,
                      Reduction_attributes().set_name("red_dV_head").set_mode(ReductionMode_t::ADD),
                      attributes.outputs[output_names::dV]);
        }

        // --------------"dO @ VT => dS_output => dK" chain--------------------

        // dP = einsum(dO, v, "b hq sq dv, b (hv g) skv dv -> b hq sq skv", g=hq//hv)
        last_output = matmul(attributes.inputs[input_names::dO],
                             attributes.inputs[input_names::V],
                             Matmul_attributes()
                                 .set_name("matmul_dO_VT")
                                 .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                                 .set_n_override(attributes.inputs[input_names::SEQ_LEN_KV]));
        last_output->set_dim({b, h_q, s_q, s_kv}).set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});

        // last_output = last_output(dP) * mask
        if (is_dropout_prob || is_dropout_mask) {
            last_output = pointwise(last_output,
                                    rng_output,
                                    Pointwise_attributes().set_name("dP_dropout_mask").set_mode(PointwiseMode_t::MUL));
        }

        // last_output = last_output - softmax_sum
        last_output = pointwise(last_output,
                                softmax_sum,
                                Pointwise_attributes().set_name("sub_dP_softmax_sum").set_mode(PointwiseMode_t::SUB));

        // last_output = last_output * exp_s_output
        last_output = pointwise(
            last_output, exp_s_output, Pointwise_attributes().set_name("mul_dP_exp_s").set_mode(PointwiseMode_t::MUL));

        // (optional) last_output = last_output * dropout_scale
        if (attributes.inputs[input_names::Dropout_scale]) {
            last_output =
                pointwise(last_output,
                          attributes.inputs[input_names::Dropout_scale],
                          Pointwise_attributes().set_name("mul_dS_dropout_scale").set_mode(PointwiseMode_t::MUL));
        }

        if (attributes.outputs[output_names::dBias]) {
            reduction(last_output,
                      Reduction_attributes().set_name("red_dP_dBias").set_mode(ReductionMode_t::ADD),
                      attributes.outputs[output_names::dBias]);
        }

        // apply the bprop of attention score modifier
        if (attributes.attention_score_modifier_bprop != nullptr) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output                  = attributes.attention_score_modifier_bprop(graph_, last_output);
            sub_nodes.emplace_back(node_);
        }

        // (optional) last_output = last_output * bmm_scale
        if (attributes.inputs[input_names::Attn_scale]) {
            last_output =
                pointwise(last_output,
                          attributes.inputs[input_names::Attn_scale],
                          Pointwise_attributes().set_name("mul_dS_attn_scale").set_mode(PointwiseMode_t::MUL));
        }

        dS_output = last_output;

        // dK = einsum(dS, Q, "b hq sq skv", "b hq sq dqk -> b hq skv dqk")
        // if GQA, then dK = reduce(dK, "b (hk g) skv dqk -> b hk skv dqk", hq//hk)
        // as reshape + matmul
        last_output = reshape(last_output, Reshape_attributes().set_name("reshape_dS"));
        last_output->set_dim({b, h_q, s_kv, s_q}).set_stride({h_q * s_q * s_kv, s_q * s_kv, 1, s_kv});
        last_output->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

        if (h_q == h_k) {
            // for MHA
            matmul(last_output,
                   attributes.inputs[input_names::Q],
                   Matmul_attributes()
                       .set_name("matmul_dST_Q")
                       .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                       .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]),
                   attributes.outputs[output_names::dK]);
        } else {
            // for GQA and MQA
            dK_fullhead = matmul(last_output,
                                 attributes.inputs[input_names::Q],
                                 Matmul_attributes()
                                     .set_name("matmul_dST_Q")
                                     .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                                     .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]));

            dK_fullhead->set_dim({b, h_q, s_kv, d_qk});
            dK_fullhead->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

            if (attributes.outputs[output_names::dK]->get_ragged_offset() &&
                attributes.max_total_seq_len_kv.has_value()) {
                // sized THD dK_full_heads
                // hack 1 - map dK strides to dK_fullhead strides
                std::vector<int64_t> dK_fullhead_stride = attributes.outputs[output_names::dK]->get_stride();
                dK_fullhead_stride[0]                   = dK_fullhead_stride[0] * (h_q / h_k);  // batch stride
                dK_fullhead_stride[2]                   = dK_fullhead_stride[2] * (h_q / h_k);  // sequence stride
                dK_fullhead->set_stride(dK_fullhead_stride);
                // hack 2 - map dK ragged offset to dK_fullhead ragged offset with implicit multiplier
                // implicit multiplier = h_q / h_k
                dK_fullhead->set_ragged_offset(attributes.outputs[output_names::dK]->get_ragged_offset());
                // hack 3 - non virtual dK full head
                dK_fullhead->set_is_virtual(false);
                dK_fullhead_size = attributes.max_total_seq_len_kv.value() * dK_fullhead_stride[2] * sizeof(float);
            } else {
                // sized BHSD dQ_accum
                dK_fullhead->set_stride({h_q * s_kv * d_qk, s_kv * d_qk, d_qk, 1});
            }

            reduction(dK_fullhead,
                      Reduction_attributes().set_name("red_dK_head").set_mode(ReductionMode_t::ADD),
                      attributes.outputs[output_names::dK]);
        }

        // --------------"dp_scaled @ K => dQ" chain--------------------

        auto const& kt_dim    = attributes.inputs[input_names::K]->get_dim();
        auto const& kt_stride = attributes.inputs[input_names::K]->get_stride();

        // dQ = einsum(dS, K, "b hq sq skv, b (hk g) skv dqk -> b hq sq dqk", g=hq//hk)
        // as reshape + matmul
        last_output = reshape(attributes.inputs[input_names::K], Reshape_attributes().set_name("reshape_k"));
        last_output->set_dim({kt_dim[0], kt_dim[1], kt_dim[3], kt_dim[2]})
            .set_stride({kt_stride[0], kt_stride[1], kt_stride[3], kt_stride[2]});

        if (attributes.inputs[input_names::K]->get_ragged_offset() != nullptr) {
            last_output->set_ragged_offset(attributes.inputs[input_names::K]->get_ragged_offset());
        }

        if (!use_dp_workspace) {
            dQ_accum = std::make_shared<Tensor_attributes>();
            dQ_accum->set_is_virtual(false);
            dQ_accum->set_dim({b, h_q, s_q, d_qk});
            dQ_accum->set_data_type(DataType_t::FLOAT);

            if (attributes.outputs[output_names::dQ]->get_ragged_offset() &&
                attributes.max_total_seq_len_q.has_value()) {
                // sized THD dQ_accum
                dQ_accum->set_stride(attributes.outputs[output_names::dQ]->get_stride());
                dQ_accum->set_ragged_offset(attributes.outputs[output_names::dQ]->get_ragged_offset());
                dQ_accum_size = attributes.max_total_seq_len_q.value() *
                                (attributes.outputs[output_names::dQ]->get_stride())[2] * sizeof(float);
            } else {
                // sized BHSD dQ_accum
                dQ_accum->set_stride({h_q * s_q * d_qk, s_q * d_qk, d_qk, 1});
                dQ_accum_size = b * h_q * s_q * d_qk * sizeof(float);
            }

            matmul(dS_output,
                   last_output,
                   Matmul_attributes()
                       .set_name("matmul_dS_K")
                       .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                       .set_k_override(attributes.inputs[input_names::SEQ_LEN_KV]),
                   dQ_accum);

            pointwise(dQ_accum,
                      Pointwise_attributes().set_name("identity_dQ").set_mode(PointwiseMode_t::IDENTITY),
                      attributes.outputs[output_names::dQ]);
        } else {
            matmul(dS_output,
                   last_output,
                   Matmul_attributes()
                       .set_name("matmul_dS_K")
                       .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                       .set_k_override(attributes.inputs[input_names::SEQ_LEN_KV]),
                   attributes.outputs[output_names::dQ]);
        }

        return {error_code_t::OK, ""};
    }

    virtual int64_t
    get_fe_workspace_size_node() const override final {
        int64_t size = 0;

        size += ((alibi_slopes_size + 15) / 16 * 16);  // align alibi slopes memory to 16 bytes
        size += dQ_accum_size;
        size += dK_fullhead_size;
        size += dV_fullhead_size;
        size += softmax_sum_size;

        return size;
    }

    virtual error_t
    collect_tensors_in_workspace_node(
        std::unordered_map<Tensor_attributes::uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>&
            workspace_modifications,
        int64_t& offset) const override final {
        if (attributes.alibi_mask) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(Q, input_names::Q);
            int64_t const h_q     = Q->second->get_dim()[1];
            auto alibi_slopes_vec = detail::get_abili_slope(h_q);
            workspace_modifications.emplace(alibi_slopes->get_uid(), std::make_tuple(0, offset, alibi_slopes_vec));
            int64_t alibi_slopes_size_padded = ((alibi_slopes_size + 15) / 16 * 16);
            offset                           = offset + alibi_slopes_size_padded;
        }

        if (dQ_accum && !dQ_accum->get_is_virtual()) {
            if (detail::get_backend_version() < 90600) {
                // prior to cuDNN 9.6.0, dQ_accum needed to be memset by frontend
                workspace_modifications.emplace(dQ_accum->get_uid(),
                                                std::make_tuple(1, offset, std::vector<float>{(float)dQ_accum_size}));
            } else {
                workspace_modifications.emplace(dQ_accum->get_uid(), std::make_tuple(2, offset, std::vector<float>()));
            }
            offset = offset + dQ_accum_size;
        }

        if (dK_fullhead && !dK_fullhead->get_is_virtual()) {
            workspace_modifications.emplace(dK_fullhead->get_uid(), std::make_tuple(2, offset, std::vector<float>()));
            offset = offset + dK_fullhead_size;
        }

        if (dV_fullhead && !dV_fullhead->get_is_virtual()) {
            workspace_modifications.emplace(dV_fullhead->get_uid(), std::make_tuple(2, offset, std::vector<float>()));
            offset = offset + dV_fullhead_size;
        }

        if (softmax_sum && !softmax_sum->get_is_virtual()) {
            workspace_modifications.emplace(softmax_sum->get_uid(), std::make_tuple(2, offset, std::vector<float>()));
            offset = offset + softmax_sum_size;
        }

        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "SDPA_BWD"})"_json);
    }
#endif
};

}  // namespace cudnn_frontend::graph
