/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Che
 */

#include "GeQuant.hpp"

#include "../Manifold/Fish.hpp"
#include "../Manifold/Neuron.hpp"
#include "GTensor.hpp"

hQUANT GeQuant::MakeInstance(GeNeuron* hNeuron, const std::string& nam_, int flag) {
    hQUANT hQuant = nullptr;
    SHAPE sp;
    QUANT_MODE type = hNeuron->Config().quant.Type(nam_);
    switch (type) {
        case SINQ:
            hQuant = std::make_shared<Q_SinkNormal<float>>(nam_ + "_quant", hNeuron, 0x0);
            break;
        case KV_JL: {
            SelfAttention* qkv = dynamic_cast<SelfAttention*>(hNeuron);
            assert(qkv != nullptr);
            // hQuant = std::make_shared<Q_JL<float,float>>(nam_+"_quant",SHAPE({head_dim,head_dim*2}), 256,this,0x0);
        } break;
        default:
            break;
    }

    return hQuant;
}

GeQuant::GeQuant(const std::string& nam_, SHAPE sp, void* hBase, int flag) : name(nam_), shape(sp) {
    hNeuron = (GeNeuron*)(hBase);
    std::hash<std::string> hasher;
    size_t hash = hasher(name);  // Returns a size_t (unsigned integer)
    rander.Init(hash);
}

GeQuant::~GeQuant() {}

template <typename T, typename Tproj>
Q_JL<T, Tproj>::Q_JL(const std::string& nam_, SHAPE shape_, int nLier, GeNeuron* hN, int flag) : GeQuant(nam_, shape_, hN, flag) {
    nOutlier           = nLier;
    SelfAttention* qkv = dynamic_cast<SelfAttention*>(hN);
    auto config        = qkv->GetFish()->config;
    assert(qkv != nullptr);
    emb_dim    = qkv->head_dim;
    group_size = 32;
    seq_len    = config.chat_sampler.seq_len;
    assert(seq_len % group_size == 0);
    seq_len = n_size = seq_len / group_size;
    head_size        = qkv->n_head;
    batch_size       = 1, head_size, n_size;
    InitProject(flag);
}

/**
 * 1.


     def init_rot_dir(self):
        rot_matrices = []
        num_chunks = (self.dim[1] + self.dim[0] - 1) // self.dim[0]
        for i in range(num_chunks):
            start_idx = i * self.dim[0]
            end_idx = (i + 1) * self.dim[0]
            q, _ = torch.linalg.qr(self.proj_dir[:, start_idx:end_idx], mode='reduced')
            rot_matrices.append(q)
        return torch.cat(rot_matrices, dim=-1) * math.sqrt(self.dim[0])

    def compose_rand_hadamard_transform(self):
        H = torch.from_numpy(hadamard(self.dim[0], dtype=float) / math.sqrt(self.dim[0])).to(self.device)
        HD = (H * (2. * torch.randint(0, 2, (self.dim[0],), device=self.device) - 1.)).to(self.proj_dir_score.dtype)
        return torch.einsum('dn,dm-> mn', self.proj_dir_score, HD)
 */
template <typename T, typename Tproj>
int Q_JL<T, Tproj>::InitProject(int flag) {
    //  torch.randn(self.dim, generator=rng, dtype=torch.float32, device=self.device)
    hProj = GT(hNeuron->GetFish(), typNUMBER::BF16, shape, 0x0, name + ".JL");  //  hN->hFish
    hProj->Alloc();
    JL   = TO<Tproj>(hProj);
    seed = rander.RandU32();  // !is different with hProj->param_seed
    //  a standard normal distribution (mean=0, std=1).
    hProj->tpInit = INIT_WEIGHT::GAUSSIAN_NORMAL;
    hProj->InitParam(0x0);
    SUM::nInitParam--;  // JL project matrix may also be params in later version
    // hProj->Print(hProj->name,0,-1);
    int batch = 1, sketch_dim = hProj->shape[0], hash_dim = sketch_dim / 8;
    int outlier_sketch_dim = nOutlier, outlier_hash_dim = outlier_sketch_dim / 8;
    // auto key_quant = torch::zeros({batch, head, n, group_size, hash_dim}, options).contiguous();
    key_quant = GT(hNeuron->GetFish(), typNUMBER::BF16, SHAPE{batch, head_size, n_size, group_size, hash_dim}, 0x0, name + ".quant");
    // auto key_outlier_quant = torch::zeros({batch, head, n, group_size, outlier_hash_dim}, options).contiguous();
    key_outlier_quant = GT(hNeuron->GetFish(), typNUMBER::BF16, SHAPE{batch, head_size, n_size, group_size, outlier_hash_dim}, 0x0, name + ".lier_quant");
    // auto outlier_norms = torch::zeros({batch, head, n, group_size}, options_outlier_norm).contiguous();
    outlier_norms = GT(hNeuron->GetFish(), typNUMBER::BF16, SHAPE{batch, head_size, n_size, group_size}, 0x0, name + ".lier_norm");

    return 0x0;
}

// Explicit instantiation for specific types
template class Q_JL<float, float>;
template class Q_JL<bf16, bf16>;