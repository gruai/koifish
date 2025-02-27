void ggml_compute_backward(struct ggml_context * ctx, struct ggml_tensor * tensor, struct ggml_hash_set * zero_table) {
    struct ggml_tensor * src0 = tensor->src[0];
    struct ggml_tensor * src1 = tensor->src[1];
    struct ggml_tensor * src2 = tensor->src[2];

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                if (src0->grad) {
                    src0->grad = ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
                }
            } break;
        case GGML_OP_ADD:
            {
                if (src0->grad) {
                    src0->grad = ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
                }
                if (src1->grad) {
                    if (ggml_are_same_shape(src0, src1)) {
                        src1->grad = ggml_add_or_set(ctx, src1->grad,                       tensor->grad,        zero_table);
                    } else {
                        src1->grad = ggml_add_or_set(ctx, src1->grad, ggml_repeat_back(ctx, tensor->grad, src1), zero_table);
                    }
                }
            } break;
        case GGML_OP_ADD1:
            {
                if (src0->grad) {
                    src0->grad = ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
                }
                if (src1->grad) {
                    src1->grad = ggml_add_or_set(ctx,
                        src1->grad,
                        ggml_mean(ctx, tensor->grad), // TODO: should probably be sum instead of mean
                        zero_table);
                }
            } break;
        case GGML_OP_ACC:
            {
                if (src0->grad) {
                    src0->grad = ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
                }
                if (src1->grad) {
                    const size_t nb1     = ((int32_t *) tensor->op_params)[0];
                    const size_t nb2     = ((int32_t *) tensor->op_params)[1];
                    const size_t nb3     = ((int32_t *) tensor->op_params)[2];
                    const size_t offset  = ((int32_t *) tensor->op_params)[3];

                    struct ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                        tensor->grad,
                        src1->grad->ne[0],
                        src1->grad->ne[1],
                        src1->grad->ne[2],
                        src1->grad->ne[3],
                        nb1, nb2, nb3, offset);

                    src1->grad =
                        ggml_add_or_set(ctx,
                            src1->grad,
                            ggml_reshape(ctx,
                                ggml_cont(ctx, tensor_grad_view),
                                src1->grad),
                            zero_table);
                }
            } break;
        case GGML_OP_SUB:
            {
                if (src0->grad) {
                    src0->grad = ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
                }
                if (src1->grad) {
                    src1->grad = ggml_sub_or_set(ctx, src1->grad, tensor->grad, zero_table);
                }
            } break;
        case GGML_OP_MUL:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx,
                                src0->grad,
                                ggml_mul(ctx, src1, tensor->grad),
                                zero_table);
                }
                if (src1->grad) {
                    src1->grad =
                        ggml_add_or_set(ctx,
                                src1->grad,
                                ggml_mul(ctx, src0, tensor->grad),
                                zero_table);
                }
            } break;
        case GGML_OP_DIV:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx,
                                src0->grad,
                                ggml_div(ctx, tensor->grad, src1),
                                zero_table);
                }
                if (src1->grad) {
                    src1->grad =
                        ggml_sub_or_set(ctx,
                                src1->grad,
                                ggml_mul(ctx,
                                    tensor->grad,
                                    ggml_div(ctx, tensor, src1)),
                                zero_table);
                }
            } break;
        case GGML_OP_SQR:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx,
                                src0->grad,
                                ggml_scale(ctx,
                                    ggml_mul(ctx, src0, tensor->grad),
                                    2.0f),
                                zero_table);
                }
            } break;
        case GGML_OP_SQRT:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx,
                                src0->grad,
                                ggml_scale(ctx,
                                    ggml_div(ctx,
                                        tensor->grad,
                                        tensor),
                                    0.5f),
                                zero_table);
                }
            } break;
        case GGML_OP_LOG:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx,
                                src0->grad,
                                ggml_div(ctx,
                                    tensor->grad,
                                    src0),
                                zero_table);
                }
            } break;
        case GGML_OP_SIN:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx,
                                src0->grad,
                                ggml_mul(ctx,
                                    tensor->grad,
                                    ggml_cos(ctx, src0)),
                                zero_table);
                }
            } break;
        case GGML_OP_COS:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_sub_or_set(ctx,
                                src0->grad,
                                ggml_mul(ctx,
                                    tensor->grad,
                                    ggml_sin(ctx, src0)),
                                zero_table);
                }
            } break;
        case GGML_OP_SUM:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add1_or_set(ctx,
                                src0->grad,
                                tensor->grad,
                                zero_table);
                }
            } break;
        case GGML_OP_SUM_ROWS:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx,
                                src0->grad,
                                ggml_repeat(ctx,
                                    tensor->grad,
                                    src0->grad),
                                zero_table);
                }
            } break;
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                GGML_ABORT("fatal error"); // TODO: implement
            }
        case GGML_OP_REPEAT:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad = ggml_add_or_set(ctx,
                            src0->grad,
                            ggml_repeat_back(ctx, tensor->grad, src0->grad),
                            zero_table);
                }
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                if (src0->grad) {
                    // TODO: test this
                    src0->grad = ggml_add_or_set(ctx,
                            src0->grad,
                            ggml_repeat(ctx, tensor->grad, src0->grad),
                            zero_table);
                }
            } break;
        case GGML_OP_CONCAT:
            {
                GGML_ABORT("fatal error"); // TODO: implement
            }
        case GGML_OP_SILU_BACK:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_NORM:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_RMS_NORM:
            {
                // necessary for llama
                if (src0->grad) {
                    float eps;
                    memcpy(&eps, tensor->op_params, sizeof(float));

                    src0->grad = ggml_add_or_set(ctx,
                            src0->grad,
                            ggml_rms_norm_back(ctx, src0, tensor->grad, eps),
                            zero_table);
                }
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_GROUP_NORM:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_MUL_MAT:
            {
                // https://cs231n.github.io/optimization-2/#staged
                // # forward pass
                // s0 = np.random.randn(5, 10)
                // s1 = np.random.randn(10, 3)
                // t = s0.dot(s1)

                // # now suppose we had the gradient on t from above in the circuit
                // dt = np.random.randn(*t.shape) # same shape as t
                // ds0 = dt.dot(s1.T) #.T gives the transpose of the matrix
                // ds1 = t.T.dot(dt)

                // tensor.shape [m,p,qq,rr]
                // src0.shape   [n,m,q1,r1]
                // src1.shape   [n,p,qq,rr]

                // necessary for llama
                if (src0->grad) {
                    struct ggml_tensor * s1_tg =
                        ggml_out_prod(ctx, // [n,m,qq,rr]
                            src1,          // [n,p,qq,rr]
                            tensor->grad); // [m,p,qq,rr]
                    const int64_t qq = s1_tg->ne[2];
                    const int64_t rr = s1_tg->ne[3];
                    const int64_t q1 = src0->ne[2];
                    const int64_t r1 = src0->ne[3];
                    const bool ne2_broadcasted = qq > q1;
                    const bool ne3_broadcasted = rr > r1;
                    if (ne2_broadcasted || ne3_broadcasted) {
                        // sum broadcast repetitions of s1_tg into shape of src0
                        s1_tg = ggml_repeat_back(ctx, s1_tg, src0);
                    }
                    src0->grad =
                        ggml_add_or_set(ctx,
                                src0->grad, // [n,m,q1,r1]
                                s1_tg,      // [n,m,q1,r1]
                                zero_table);
                }
                if (src1->grad) {
                    src1->grad =
                        ggml_add_or_set(ctx,
                                src1->grad,                            // [n,p,qq,rr]
                                // ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                                //     ggml_cont(ctx,                  // [m,n,q1,r1]
                                //         ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                                //     tensor->grad),                  // [m,p,qq,rr]

                                // // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                                // // avoid transpose of src0, rather transpose smaller tensor->grad
                                // // and then use ggml_out_prod
                                ggml_out_prod(ctx,                  // [n,p,qq,rr]
                                    src0,                           // [n,m,q1,r1]
                                    ggml_transpose(ctx,             // [p,m,qq,rr]
                                        tensor->grad)),             // [m,p,qq,rr]
                                zero_table);
                }
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_OUT_PROD:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_SCALE:
            {
                // necessary for llama
                if (src0->grad) {
                    float s;
                    memcpy(&s, tensor->op_params, sizeof(float));

                    src0->grad =
                        ggml_add_or_set(ctx,
                            src0->grad,
                            ggml_scale_impl(ctx, tensor->grad, s, false),
                            zero_table);
                }
            } break;
        case GGML_OP_SET:
            {
                const size_t nb1     = ((int32_t *) tensor->op_params)[0];
                const size_t nb2     = ((int32_t *) tensor->op_params)[1];
                const size_t nb3     = ((int32_t *) tensor->op_params)[2];
                const size_t offset  = ((int32_t *) tensor->op_params)[3];

                struct ggml_tensor * tensor_grad_view = NULL;

                if (src0->grad || src1->grad) {
                    GGML_ASSERT(src0->type == tensor->type);
                    GGML_ASSERT(tensor->grad->type == tensor->type);
                    GGML_ASSERT(tensor->grad->type == src1->grad->type);

                    tensor_grad_view = ggml_view_4d(ctx,
                        tensor->grad,
                        src1->grad->ne[0],
                        src1->grad->ne[1],
                        src1->grad->ne[2],
                        src1->grad->ne[3],
                        nb1, nb2, nb3, offset);
                }

                if (src0->grad) {
                    src0->grad = ggml_add_or_set(ctx,
                        src0->grad,
                        ggml_acc_impl(ctx,
                            tensor->grad,
                            ggml_neg(ctx, tensor_grad_view),
                            nb1, nb2, nb3, offset, false),
                        zero_table);
                }

                if (src1->grad) {
                    src1->grad =
                        ggml_add_or_set(ctx,
                            src1->grad,
                            ggml_reshape(ctx,
                                ggml_cont(ctx, tensor_grad_view),
                                src1->grad),
                            zero_table);
                }
            } break;
        case GGML_OP_CPY:
            {
                // necessary for llama
                // cpy overwrites value of src1 by src0 and returns view(src1)
                // the overwriting is mathematically equivalent to:
                // tensor = src0 * 1 + src1 * 0
                if (src0->grad) {
                    // dsrc0 = dtensor * 1
                    src0->grad = ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
                }
                if (src1->grad) {
                    // dsrc1 = dtensor * 0 -> noop
                }
            } break;
        case GGML_OP_CONT:
            {
                // same as cpy
                if (src0->grad) {
                    GGML_ASSERT(ggml_is_contiguous(src0->grad));
                    GGML_ASSERT(ggml_is_contiguous(tensor->grad));
                    src0->grad = ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
                }
            } break;
        case GGML_OP_RESHAPE:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx, src0->grad,
                            ggml_reshape(ctx,
                                ggml_is_contiguous(tensor->grad)
                                    ? tensor->grad
                                    : ggml_cont(ctx, tensor->grad),
                                src0->grad),
                        zero_table);
                }
            } break;
        case GGML_OP_VIEW:
            {
                // necessary for llama
                if (src0->grad) {
                    size_t offset;

                    memcpy(&offset, tensor->op_params, sizeof(offset));

                    size_t nb1     = tensor->nb[1];
                    size_t nb2     = tensor->nb[2];
                    size_t nb3     = tensor->nb[3];

                    if (src0->type != src0->grad->type) {
                        // gradient is typically F32, but src0 could be other type
                        size_t ng = ggml_element_size(src0->grad);
                        size_t n0 = ggml_element_size(src0);
                        GGML_ASSERT(offset % n0 == 0);
                        GGML_ASSERT(nb1 % n0 == 0);
                        GGML_ASSERT(nb2 % n0 == 0);
                        GGML_ASSERT(nb3 % n0 == 0);
                        offset = (offset / n0) * ng;
                        nb1 = (nb1 / n0) * ng;
                        nb2 = (nb2 / n0) * ng;
                        nb3 = (nb3 / n0) * ng;
                    }

                    src0->grad = ggml_acc_or_set(ctx, src0->grad, tensor->grad, nb1, nb2, nb3, offset, zero_table);
                }
            } break;
        case GGML_OP_PERMUTE:
            {
                // necessary for llama
                if (src0->grad) {
                    int32_t * axes = (int32_t *) tensor->op_params;
                    int axis0 = axes[0] & 0x3;
                    int axis1 = axes[1] & 0x3;
                    int axis2 = axes[2] & 0x3;
                    int axis3 = axes[3] & 0x3;
                    int axes_backward[4] = {0,0,0,0};
                    axes_backward[axis0] = 0;
                    axes_backward[axis1] = 1;
                    axes_backward[axis2] = 2;
                    axes_backward[axis3] = 3;
                    src0->grad =
                        ggml_add_or_set(ctx, src0->grad,
                            ggml_permute(ctx,
                                tensor->grad,
                                axes_backward[0],
                                axes_backward[1],
                                axes_backward[2],
                                axes_backward[3]),
                            zero_table);
                }
            } break;
        case GGML_OP_TRANSPOSE:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx, src0->grad,
                            ggml_transpose(ctx, tensor->grad),
                        zero_table);
                }
            } break;
        case GGML_OP_GET_ROWS:
            {
                // necessary for llama (only for tokenizer)
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx, src0->grad,
                            // last ggml_get_rows_back argument src0->grad is only
                            // necessary to setup correct output shape
                            ggml_get_rows_back(ctx, tensor->grad, src1, src0->grad),
                        zero_table);
                }
                if (src1->grad) {
                    // noop
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_DIAG:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_DIAG_MASK_INF:
            {
                // necessary for llama
                if (src0->grad) {
                    const int n_past = ((int32_t *) tensor->op_params)[0];
                    src0->grad =
                        ggml_add_or_set(ctx, src0->grad,
                            /* ggml_diag_mask_inf_impl() shouldn't be here */
                            /* ref:  https://github.com/ggerganov/llama.cpp/pull/4203#discussion_r1412377992 */
                            ggml_diag_mask_zero_impl(ctx, tensor->grad, n_past, false),
                        zero_table);
                }
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
            {
                // necessary for llama
                if (src0->grad) {
                    const int n_past = ((int32_t *) tensor->op_params)[0];
                    src0->grad =
                        ggml_add_or_set(ctx, src0->grad,
                            ggml_diag_mask_zero_impl(ctx, tensor->grad, n_past, false),
                        zero_table);
                }
            } break;
        case GGML_OP_SOFT_MAX:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx, src0->grad,
                            ggml_soft_max_back(ctx, tensor->grad, tensor),
                        zero_table);
                }

            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_ROPE:
            {
                // necessary for llama
                if (src0->grad) {
                    //const int n_past = ((int32_t *) tensor->op_params)[0];
                    const int n_dims     = ((int32_t *) tensor->op_params)[1];
                    const int mode       = ((int32_t *) tensor->op_params)[2];
                    //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                    const int n_ctx_orig = ((int32_t *) tensor->op_params)[4];
                    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;

                    memcpy(&freq_base,   (int32_t *) tensor->op_params +  5, sizeof(float));
                    memcpy(&freq_scale,  (int32_t *) tensor->op_params +  6, sizeof(float));
                    memcpy(&ext_factor,  (int32_t *) tensor->op_params +  7, sizeof(float));
                    memcpy(&attn_factor, (int32_t *) tensor->op_params +  8, sizeof(float));
                    memcpy(&beta_fast,   (int32_t *) tensor->op_params +  9, sizeof(float));
                    memcpy(&beta_slow,   (int32_t *) tensor->op_params + 10, sizeof(float));

                    src0->grad = ggml_add_or_set(ctx,
                            src0->grad,
                            ggml_rope_back(ctx,
                                tensor->grad,
                                src1,
                                src2,
                                n_dims,
                                mode,
                                n_ctx_orig,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                beta_fast,
                                beta_slow),
                            zero_table);
                }
            } break;
        case GGML_OP_ROPE_BACK:
            {
                if (src0->grad) {
                    //const int n_past = ((int32_t *) tensor->op_params)[0];
                    const int n_dims     = ((int32_t *) tensor->op_params)[1];
                    const int mode       = ((int32_t *) tensor->op_params)[2];
                    //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                    const int n_ctx_orig = ((int32_t *) tensor->op_params)[4];
                    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;

                    memcpy(&freq_base,   (int32_t *) tensor->op_params +  5, sizeof(float));
                    memcpy(&freq_scale,  (int32_t *) tensor->op_params +  6, sizeof(float));
                    memcpy(&ext_factor,  (int32_t *) tensor->op_params +  7, sizeof(float));
                    memcpy(&attn_factor, (int32_t *) tensor->op_params +  8, sizeof(float));
                    memcpy(&beta_fast,   (int32_t *) tensor->op_params +  9, sizeof(float));
                    memcpy(&beta_slow,   (int32_t *) tensor->op_params + 10, sizeof(float));

                    src0->grad = ggml_add_or_set(ctx,
                            src0->grad,
                            ggml_rope_impl(ctx,
                                tensor->grad,
                                src1,
                                src2,
                                n_dims,
                                mode,
                                n_ctx_orig,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                beta_fast,
                                beta_slow,
                                false),
                            zero_table);
                }
            } break;
        case GGML_OP_CLAMP:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_IM2COL:
            {
                if (src1->grad) {
                    const int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                    const int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                    const int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                    const int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                    const int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                    const int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                    const bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                    src1->grad = ggml_add_or_set(ctx,
                            src1->grad,
                            ggml_im2col_back(ctx, src0, tensor->grad, src1->ne, s0, s1, p0, p1, d0, d1, is_2D),
                            zero_table);
                }
            } break;
        case GGML_OP_IM2COL_BACK:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_POOL_1D:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_POOL_2D:
            {
                if (src0->grad) {
                    const enum ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                    const      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                    const      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                    const      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                    const      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                    const      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                    const      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                    src0->grad = ggml_add_or_set(ctx,
                            src0->grad,
                            ggml_pool_2d_back(ctx, tensor->grad, src0, op, k0, k1, s0, s1, p0, p1),
                            zero_table);
                }
            } break;
        case GGML_OP_POOL_2D_BACK:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_UPSCALE:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_PAD:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_ARANGE:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_ARGSORT:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_LEAKY_RELU:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_FLASH_ATTN_EXT:
            {
                struct ggml_tensor * flash_grad = NULL;
                if (src0->grad || src1->grad || tensor->src[2]->grad) {
                    int32_t t = ggml_get_op_params_i32(tensor, 0);
                    GGML_ASSERT(t == 0 || t == 1);
                    bool masked = t != 0;
                    flash_grad =
                        ggml_flash_attn_back(ctx,
                            src0,
                            src1,
                            tensor->src[2],
                            tensor->grad,
                            masked);
                }

                const int64_t elem_q = ggml_nelements(src0);
                const int64_t elem_k = ggml_nelements(src1);
                const int64_t elem_v = ggml_nelements(src2);

                enum ggml_type result_type = flash_grad->type;
                GGML_ASSERT(ggml_blck_size(result_type) == 1);
                const size_t tsize = ggml_type_size(result_type);

                const size_t offs_q = 0;
                const size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
                const size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

                if (src0->grad) {
                    struct ggml_tensor * view_q = ggml_view_1d(ctx, flash_grad, elem_q, offs_q);
                    struct ggml_tensor * grad_q = ggml_reshape(ctx, view_q, src0);
                    src0->grad = ggml_add_or_set(ctx,
                            src0->grad,
                            grad_q,
                            zero_table);
                }
                if (src1->grad) {
                    struct ggml_tensor * view_k = ggml_view_1d(ctx, flash_grad, elem_k, offs_k);
                    struct ggml_tensor * grad_k = ggml_reshape(ctx, view_k, src1);
                    src1->grad = ggml_add_or_set(ctx,
                            src1->grad,
                            grad_k,
                            zero_table);
                }
                if (src2->grad) {
                    struct ggml_tensor * view_v = ggml_view_1d(ctx, flash_grad, elem_v, offs_v);
                    struct ggml_tensor * grad_v = ggml_reshape(ctx, view_v, src2);
                    src2->grad = ggml_add_or_set(ctx,
                            src2->grad,
                            grad_v,
                            zero_table);
                }
            } break;
        case GGML_OP_FLASH_ATTN_BACK:
            {
                GGML_ABORT("fatal error"); // not supported
            }
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
            {
                GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_UNARY:
            {
                switch (ggml_get_unary_op(tensor)) {
                    case GGML_UNARY_OP_ABS:
                        {
                            if (src0->grad) {
                                src0->grad =
                                    ggml_add_or_set(ctx,
                                            src0->grad,
                                            ggml_mul(ctx,
                                                ggml_sgn(ctx, src0),
                                                tensor->grad),
                                            zero_table);
                            }
                        } break;
                    case GGML_UNARY_OP_SGN:
                        {
                            if (src0->grad) {
                                // noop
                            }
                        } break;
                    case GGML_UNARY_OP_NEG:
                        {
                            if (src0->grad) {
                                src0->grad = ggml_sub_or_set(ctx, src0->grad, tensor->grad, zero_table);
                            }
                        } break;
                    case GGML_UNARY_OP_STEP:
                        {
                            if (src0->grad) {
                                // noop
                            }
                        } break;
                    case GGML_UNARY_OP_TANH:
                        {
                            GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case GGML_UNARY_OP_ELU:
                        {
                            GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case GGML_UNARY_OP_RELU:
                        {
                            if (src0->grad) {
                                src0->grad = ggml_add_or_set(ctx,
                                        src0->grad,
                                        ggml_mul(ctx,
                                            ggml_step(ctx, src0),
                                            tensor->grad),
                                        zero_table);
                            }
                        } break;
                    case GGML_UNARY_OP_SIGMOID:
                        {
                            GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case GGML_UNARY_OP_GELU:
                        {
                            GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case GGML_UNARY_OP_GELU_QUICK:
                        {
                            GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case GGML_UNARY_OP_SILU:
                        {
                            // necessary for llama
                            if (src0->grad) {
                                src0->grad = ggml_add_or_set(ctx,
                                        src0->grad,
                                        ggml_silu_back(ctx, src0, tensor->grad),
                                        zero_table);
                            }
                        } break;
                    case GGML_UNARY_OP_EXP:
                        {
                            if (src0->grad) {
                                src0->grad = ggml_add_or_set(ctx,
                                        src0->grad,
                                        ggml_mul(ctx, tensor, tensor->grad),
                                        zero_table);
                            }
                        } break;
                    default:
                        GGML_ABORT("fatal error");
                }
            } break;
        case GGML_OP_GET_REL_POS:
        case GGML_OP_ADD_REL_POS:
        case GGML_OP_RWKV_WKV:
        case GGML_OP_MAP_UNARY:
        case GGML_OP_MAP_BINARY:
        case GGML_OP_MAP_CUSTOM1_F32:
        case GGML_OP_MAP_CUSTOM2_F32:
        case GGML_OP_MAP_CUSTOM3_F32:
        case GGML_OP_MAP_CUSTOM1:
        case GGML_OP_MAP_CUSTOM2:
        case GGML_OP_MAP_CUSTOM3:
            {
                GGML_ABORT("fatal error"); // not supported
            }
        case GGML_OP_CROSS_ENTROPY_LOSS:
            {
                if (src0->grad) {
                    src0->grad = ggml_add_or_set(ctx,
                                src0->grad,
                                ggml_cross_entropy_loss_back(ctx,
                                    src0,
                                    src1,
                                    tensor->grad),
                                zero_table);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS_1:
            {
                if (src0->grad) {
                    src0->grad = ggml_add_or_set(ctx,
                                src0->grad,
                                ggml_cross_entropy_loss_back_1(ctx,
                                    src0,
                                    src1,
                                    tensor->grad),
                                zero_table);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                GGML_ABORT("fatal error"); // not supported
            }
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK_1:
            {
                GGML_ABORT("fatal error"); // not supported
            }
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (tensor->src[i] && tensor->src[i]->grad) {
            GGML_ASSERT(ggml_are_same_shape(tensor->src[i], tensor->src[i]->grad));
        }
    }
}