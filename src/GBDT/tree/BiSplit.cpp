#include "BiSplit.hpp"

#include "../data_fold/DataFold.hpp"
#include "../data_fold/FeatVec_2D.hpp"
#include "../data_fold/Loss.hpp"
#include "../util/Object.hpp"
#include "BoostingForest.hpp"

using namespace Grusoft;

double MT_BiSplit::tX = 0;

SAMP_SET::SAMP_SET(size_t nSamp_, tpSAMP_ID *samp_0, int flag) {
    isRef = true;
    nSamp = nSamp_;
    samps = samp_0;
}

void SAMP_SET::Alloc(FeatsOnFold *hData_, size_t nSamp_, int flag) {
    clear();

    isRef = false;
    nSamp = nSamp_;
    if (false) {
        root_set = new tpSAMP_ID[nSamp];
        left     = new tpSAMP_ID[nSamp];
        rigt     = new tpSAMP_ID[nSamp];
    } else {  // ��ʡ�ڴ�
        root_set = hData_->buffer.samp_root_set;
        left     = hData_->buffer.samp_left;
        rigt     = hData_->buffer.samp_rigt;
        isRef    = true;
    }
    for (size_t i = 0; i < nSamp; i++) root_set[i] = i;
    samps = root_set;
}

void SAMP_SET::Alloc(size_t nSamp_, int flag) {
    clear();

    isRef    = false;
    nSamp    = nSamp_;
    root_set = new tpSAMP_ID[nSamp];
    left     = new tpSAMP_ID[nSamp];
    rigt     = new tpSAMP_ID[nSamp];
    for (size_t i = 0; i < nSamp; i++) root_set[i] = i;
    samps = root_set;
}

/*
    v0.2		cys
        3/29/2019
    v0.3		cys
        8/31/2019
*/
void SAMP_SET::SampleFrom(FeatsOnFold *hData_, const BoostingForest *hBoosting, const SAMP_SET *from, size_t nMost, int rnd_seed, int flag) {
    float *weight         = hData_->lossy->GetSampWeight(0x0);
    const tpDOWN *hessian = hData_->GetHessian();
    const tpDOWN *down    = hData_->GetDownDirection();
    Alloc(hData_, nMost);
    size_t i, nFrom = hData_->nSample(), nz = 0, pos, nSmall = 0;
    double T_grad = DBL_MAX, b, rElitism = hData_->config.rElitism;
    float *samp_weight = hData_->lossy->samp_weight;  // assert(samp_weight!=nullptr);
    if (hData_->isTrain() && rElitism > 0) {
        T_grad = hData_->lossy->DOWN_sum_2;
        assert(T_grad > 0);
        T_grad = sqrt(T_grad / nFrom);
        // T_grad = hData_->lossy->DOWN_mean+hData_->lossy->DOWN_devia/2;
    }
    // nElitism = 0;
    if (from == nullptr) {
    } else {
        assert(from != nullptr && nMost < from->nSamp);
        nFrom = from->nSamp;
    }
    if (nMost >= nFrom) {
        for (size_t i = 0; i < nMost; i++) {
            root_set[i] = i;
        }
        // memcpy(root_set, from->samps,sizeof(tpSAMP_ID)*nFrom);
        return;
    }

    size_t T_1     = nFrom / std::log2(nMost * 1.0);
    unsigned int x = 123456789, next;
    // srand(time(0));
    x               = hData_->rander_samp.RandInt32() % nFrom;
    bool isSequence = false && from == nullptr;
    if (isSequence) {                            // ��ʧ֮��...		4/11/2019	cys
        size_t grid = max(int(nMost / 100), 1);  // overlap bagging
        while (nz < nMost) {
            size_t start = hData_->rander_samp.RandInt32() % nFrom, end = min(grid, nFrom - start);
            end = min(end, nMost - nz);
            end += start;
            for (i = start; i < end; i++) {
                root_set[nz++] = i;
            }
        }
        // std::sort(root_set, root_set + nMost);
    } else if (nMost > T_1) {
        /*size_t nz0 = hData_->rander_samp.kSampleInN(root_set,nMost, nFrom);		//case_higgs�������ԣ�ȷʵ��Ч��*/
        for (nz = 0, i = 0; i < nFrom; ++i) {
            if (i == nFrom - 1)
                i = nFrom - 1;
            double prob = min(0.9999, (nMost - nz) * 1.0 / (nFrom - i));
            if (rElitism > 0) {
                if (samp_weight != nullptr) {
                    if (samp_weight[i] < 0.5) {
                        prob /= 10.0;  // nSmall++;
                    } /**/
                    // prob *= samp_weight[i] / 5;	//Ч�����ã���������
                } else {
                    // b = weight==nullptr ? fabs(down[i]) : fabs(down[i])*weight[i];
                    b = fabs(down[i]);
                    if (b < T_grad) {
                        prob /= 10;
                        nSmall++;
                    }
                }
            }
            double c = hData_->rander_samp.Uniform_(0, 1);
            if (c < prob) {
                root_set[nz++] = i;
            }
        }
    } else {
        tpSAMP_ID *mask = new tpSAMP_ID[nFrom]();
        while (nz < nMost) {
            x = (214013 * x + 2531011);
            // static_cast<int>(x & 0x7FFFFFFF);
            next = x % nMost;
            if (mask[next] == 0) {
                root_set[nz++] = next;
                mask[next]     = 1;
                // sample_set.insert(next);
            }
        }
        delete[] mask;
        std::sort(root_set, root_set + nMost);
    }
    assert(nz <= nMost);
    nSamp = nz;
    if (hBoosting->skdu.noT % hData_->config.verbose_eval == 0) {
        // printf("\nnSamp=%ld[%ld=>%ld] nSmall=%ld T_grad=%.6g\t", nFrom, nMost, nz, nSmall, T_grad);
        // printf("\nsamps={%d,%d,%d,...%d,...,%d,%d}", samps[0], samps[1], samps[2], samps[nz / 2], samps[nz - 2], samps[nz - 1]);
    }
}

MT_BiSplit::MT_BiSplit(FeatsOnFold *hData_, const BoostingForest *hBoosting_, int d, int rnd_seed, int flag) : hBForest(hBoosting_), depth(d) {
    assert(hData_ != nullptr);
    double subsample = hData_->config.subsample;

    size_t i, nSamp = hData_->nSample();
    if (subsample < 0.999) {
        size_t nMost = nSamp * subsample;
        // samp_set.SampleFrom(hData_,&(hData_->samp_set),nMost, rnd_seed);
        samp_set.SampleFrom(hData_, hBForest, nullptr, nMost, rnd_seed);
    } else {
        // samp_set.SampleFrom(hData_, &(hData_->samp_set), nSamp, rnd_seed);
        samp_set.SampleFrom(hData_, hBForest, nullptr, nSamp, rnd_seed);
        // samp_set = hData_->samp_set;
        // samp_set.isRef = true;
    }
    size_t nPickSamp = samp_set.nSamp;
    // Init_BFold(hData_);
    if (hData_->atTrainTask()) {
        Observation_AtLocalSamp(hData_);
        hData_->stat.dY = samp_set.Y_1 - samp_set.Y_0;
    }
}

/*
    ����ֻ��ĳ�������Ĺ۲�ֵ!!!
    gain,imputiry,down���뱣��һ��

    v0.1	cys
        9/4/2018
    v0.2	cys
        1/26/2018
*/
void MT_BiSplit::Observation_AtLocalSamp(FeatsOnFold *hData_, int flag) {
    GST_TIC(t1);
    char temp[2000];
    string optimal = hData_->config.leaf_optimal;

    impuri     = 0;
    devia      = 0;
    size_t dim = nSample(), i;
    if (dim == 0)
        return;
    tpDOWN *down = hData_->GetDownDirection(), *hess = hData_->GetHessian();
    hData_->GetY()->STA_at(samp_set);
    Y_mean  = samp_set.a1_sum * 1.0 / dim;
    Y2_mean = samp_set.a2_sum * 1.0 / dim;
    samp_set.ClearStat();
    // double a, x_0 = DBL_MAX, x_1 = -DBL_MAX;
    tpDOWN a, mean = 0, y_0, y_1;
    double a2 = 0.0, DOWN_sum = 0, lambda_l2 = hData_->config.lambda_l2;
    samp_set.STA_at_<tpDOWN>(down, a2, DOWN_sum, y_0, y_1, true);
    mean   = DOWN_sum / dim;
    G_sum  = -DOWN_sum;  // ����Ҫ gradient�����down�����෴
    G2_sum = a2;
    // if (y_0 == y_1 || fabs(y_0-y_1)<1.0e-6*fabs(y_0)) {	//����Ϊ����
    if ZERO_DEVIA (y_0, y_1) {  // ����Ϊ����
    } else {
        // mean /= dim;
        a      = a2 - dim * mean * mean;
        impuri = (double)(a);
        if (impuri < 0 && fabs(impuri) < 1.0e-6 * a2)
            impuri = 0;
        // if (impuri == 0)					assert(0);
        assert(impuri >= 0);
        devia = sqrt(impuri / dim);
    }
    if (G_sum == 0 && a2 != 0.0) {  // �ǳ���ֵ�����
        // G_sum = 0;
    }

    if (optimal == "lambda_0") {
        if (hess == nullptr) {
            H_sum = dim;
        } else {
            double h2;
            tpDOWN h_0, h_1;
            samp_set.STA_at_<tpDOWN>(hess, h2, H_sum, h_0, h_1, false);
            assert(fabs(h_0) < 1000 && fabs(h_1) < 1000);
        }
        if (H_sum == 0) {
            throw "Observation_AtLocalSamp:H_sum = 0!!!";
        }
        impuri    = G_sum * G_sum / (H_sum + lambda_l2);  // ����ȥ����
        down_step = -G_sum / (H_sum + lambda_l2);
        // assert(!IS_NAN_INF(down_step));
        if (false) {  // hess[samp]���ú�С
            double sum = 0;
            for (i = 0; i < dim; i++) {
                tpSAMP_ID samp = samp_set.samps[i];
                sum += fabs(hess[samp] < 0.001) ? 0 : down[samp] / hess[samp];
                assert(fabs(sum / (i + 1)) < 1000);
            }
            down_step = -sum / dim;
        }
        // sprintf(temp, "impuri(%g/%g %d)", G_sum, H_sum, dim);
        // sX = temp;
    } else {
        down_step = mean;
        // down_step = sqrt(a2/dim);		 Ϊɶ�������У�����˼
    }
    assert(fabs(down_step) < 10000);
    assert(!IS_NAN_INF(down_step));
    // printf("%.4g ", down_step);
    double shrink = hData_->config.learning_rate;
    // double init_score=hData_->lossy.init_score;
    down_step = down_step * shrink;

    return;
}

double MT_BiSplit::GetGain(int flag) { return 0x0; }

// ���ڸ�������ԭ�򣬺�С�ĸ���Ҳ��0
double FLOAT_ZERO(double a, double ruler) {
    if (a > -DBL_EPSILON * 1000 * ruler && a < 0)  // Y2_sum,mxmxN���㷽ʽ��һ����ȷʵ���и������
        return 0;
    return a;
}

/*
    �۽�������չ��
        ����������һ��ʱ��ε����Ѽ�¼
*/
double MT_BiSplit::AGG_CheckGain(FeatsOnFold *hData_, FeatVector *hFeat, int flag) {
    // assert(BIT_TEST(hFeat->type, FeatVector::ACCUMULATE));
    throw "MT_BiSplit::AGG_CheckGain is ......";
    /*FRUIT *fruit = nullptr;
    const HistoGRAM *histo = fruit->histo_refer;
    FeatVector *hAF = nullptr;
    int nExpand = 100, i, exp_no =-1;
    double mxmxN = 0;
    for(i=0;i<nExpand;i++){
        hFeat->Samp2Histo(hData_, samp_set, nullptr,histo, hData_->config.feat_quanti);
        histo->GreedySplit_X(hData_, samp_set);
        if (mxmxN < fruit->mxmxN) {
            mxmxN = fruit->mxmxN, exp_no = i;
        }
    }
    hAF->agg_no = exp_no;*/
    return 0;
}

/*
    v0.1	cys
        2/26/2019
*/
int MT_BiSplit::PickOnGain(FeatsOnFold *hData_, const vector<FRUIT *> &arrFruit, int flag) {
    double mxmxN = 0;
    vector<double> cands;
    int pick_id = -1, nFruit = arrFruit.size(), i;
    bool isRanomPick = false;       // hData_->config.random_feat_on_gain
    for (i = 0; i < nFruit; i++) {  // Ϊ�˲���
        FRUIT *fr_ = arrFruit[i];
        if (fr_ == nullptr || fr_->mxmxN <= 0) {
            cands.push_back(-1);
            continue;
        }
        cands.push_back(fr_->mxmxN - impuri);  // BUG ��Ҫ��ȥimpuri
        if (mxmxN < fr_->mxmxN) {
            mxmxN = fr_->mxmxN, pick_id = i;
            // feat_id = picks[pick_id];		feat_regress = -1;
            // fruit = fr_;
        }
    }
    double T = (mxmxN - impuri) * 0.95;
    if (isRanomPick && cands.size() > 10) {
        assert(cands.size() == arrFruit.size());
        vector<size_t> idx_1;
        idx_1.resize(cands.size());  // initialize original index locations
        iota(idx_1.begin(), idx_1.end(), 0);
        // sort indexes based on comparing values in v
        std::sort(idx_1.begin(), idx_1.end(), [&cands](size_t i1, size_t i2) { return cands[i1] > cands[i2]; });
        assert(mxmxN == cands[idx_1[0]]);
        size_t range = max(1, (int)(cands.size() / 10)), pos = -1;
        for (i = 0; i < cands.size() - 1; i++) {
            assert(cands[idx_1[i]] >= cands[idx_1[i + 1]]);
            if (cands[idx_1[i]] > T) {
                range = i + 1;
            }
        }
        pos     = 0;  // rand() % range;
        pick_id = idx_1[pos];
    }
    return pick_id;
}

/*
    v0.1	cys
        9/8/2019
*/
HistoGRAM *MT_BiSplit::GetHistogram(FeatsOnFold *hData_, int pick, bool isInsert, int flag) {
    size_t nSamp = samp_set.nSamp, i;
    // if (H_HISTO.size() == 0)
    //	return nullptr;
    const HistoGRAM_BUFFER *H_buffer = hBForest->histo_buffer;
    HistoGRAM *histo                 = H_buffer->Get(id, pick);
    histo->nSamp                     = nSamp;  // nSamp��̬�仯��H_buffer�޷�ȷ��
    assert(histo != nullptr);
    if (histo->nBins == 0) {
        if (!isInsert)
            return nullptr;
        // if (nSamp == 139)
        //	nSamp = 139;
        FeatVector *hFeat = hData_->Feat(pick);
        HistoGRAM *hP     = parent == nullptr ? nullptr : parent->GetHistogram(hData_, pick, false);
        HistoGRAM *hB     = brother == nullptr ? nullptr : brother->GetHistogram(hData_, pick, false);
        if (hP != nullptr && hB != nullptr) {
            // if(pick==0)			printf("%d@(%d %d) ", nSamp,hP->nSamp, hB->nSamp);
            histo->FromDiff(hP, hB, true);
            // �������ԣ���Ƭ���ڴ�ȷʵ�˷�ʱ��
            // parent->H_HISTO[pick] = nullptr;			delete hP;
            // brother->H_HISTO[pick] = nullptr;			delete hB;
        } else {
            GST_TIC(t333);
            hFeat->Samp2Histo(hData_, samp_set, histo, hData_->config.feat_quanti);
            FeatsOnFold::stat.tSamp2Histo += GST_TOC(t333);
            histo->CompressBins();
        }
        // histo->isFilled=true;
        // H_HISTO[pick] = histo;
        histo->CheckValid(hData_->config);
    } else {
        // histo = H_HISTO[pick];
    }
    // histo->CompressBins();
    return histo;
}
/*
    v0.2	����
*/
double MT_BiSplit::CheckGain(FeatsOnFold *hData_, const vector<int> &pick_feats, int x, int flag) {
    GST_TIC(tick);
    if (this->id == 13) {
        int i = 0;  // �����ڵ���
    }
    if (hData_->merge_lefts.size() > 0) {
        for (auto hFV : hData_->merge_lefts) {
            hFV->Merge4Quanti(&samp_set);
        }
    }
    /*if (samp_set.Y_1 - samp_set.Y_0 < hData_->stat.dY / 10) {
        printf( "\n!!!Tiny Y:::just PASS!!!	Tree=%d node=%d, samp_set=<%g-%g> |y|=%g", hModel->skdu.noT,this->id,
            samp_set.Y_0,samp_set.Y_1, hData_->stat.dY );
        impuri = 0;
        return 0;
    }*/
    int nThread = hData_->config.num_threads, pick_id = -1, node_task = hData_->config.node_task;
    size_t nSamp = samp_set.nSamp, i, step = pick_feats.size();
    if (nSamp < hData_->nSample())
        hData_->PickSample_GH(this);
    string optimal = hData_->config.leaf_optimal;
    // assert(impuri>0);
    assert(nSamp >= hData_->config.min_data_in_leaf);
    // double bst_split=0;
    vector<int> picks = pick_feats;
    // hData_->nPick4Split(picks, hData_->rander_feat);		//�ƺ�Ч��һ�㣬���	3/7/2019		cys
    feat_id = -1;
    // picks.resize(1);		//�����ڲ���
    // if (task == "split_X" || task == "split_Y")
    if (node_task == LiteBOM_Config::split_X || node_task == LiteBOM_Config::histo_X_split_G)
        assert(gain_train == 0);
    bool isEachFruit = false;  // each Fruit for each feature
    vector<FRUIT *> arrFruit;
    if (isEachFruit)
        arrFruit.resize(picks.size());
    tpDOWN *yDown = hData_->GetDownDirection();
    // picks.clear();		picks.push_back(7);		//�����ڵ���
    int num_threads = OMP_FOR_STATIC_1(picks.size(), step), nNewPick = 0, nMostNewPick = (int)(sqrt(picks.size() * 1.0));
    if (false) {
        // BinFold bf(hData_,picks, samp_set);
        // bf.GreedySplit(hData_, picks ,0x0 );
    }

    GST_TIC(t222);
    size_t start = 0, end = picks.size();
#pragma omp parallel for schedule(static)
    for (int i = start; i < end; i++) {
        HistoGRAM *histo = GetHistogram(hData_, picks[i], true);
        histo->fruit_info.Clear();
        if (histo->nBins == 0)
            continue;
        FeatVector *hFeat = hData_->Feat(picks[i]);
        if (hFeat->isCategory()) {
            // histo->GreedySplit_X(hData_, samp_set);
            histo->GreedySplit_Y(hData_, samp_set, false);
        } else {
            if (node_task == LiteBOM_Config::split_X)
                histo->GreedySplit_X(hData_, samp_set);
            else if (node_task == LiteBOM_Config::histo_X_split_G)
                histo->GreedySplit_Y(hData_, samp_set, true);
            else if (node_task == LiteBOM_Config::REGRESS_X)
                histo->Regress(hData_, samp_set);
            else
                throw "MT_BiSplit::CheckGain task is !!!";
        }
    }
    FeatsOnFold::stat.tHisto += GST_TOC(t222);

    // fruit = new FRUIT();
    // arrFruit.push_back(fruit);
    // GST_TIC(t1);
    feat_id             = -1;
    double lambda_mxmxN = 0;
    nNewPick            = 0;
    for (int i = start; i < end; i++) {
        int pick = picks[i];
        if (i == 0 && this->id == 1) {  // �����ڲ���
            // i = 0;
        }
        FeatVector *hFeat = hData_->Feat(pick);
        // HistoGRAM *histo = optimal=="grad_variance" ? new HistoGRAM(nSamp) : new Histo_CTQ(nSamp);
        HistoGRAM *histo = GetHistogram(hData_, pick, true), *histoSwarm = nullptr;
        double a = histo->fruit_info.mxmxN;
        if (hFeat->select.hasCheckGain == false && hData_->config.lambda_Feat < 1) {
            a        = histo->fruit_info.mxmxN * hData_->config.lambda_Feat;
            nNewPick = nNewPick + 1;
            if (nNewPick > nMostNewPick) {
                a = 0;
            }
        }
        if (a > lambda_mxmxN) {
            lambda_mxmxN = a;
            feat_id      = pick;
        }
        if (hFeat->select_bins !=
            nullptr) {  // ����case_poct���ԣ��ƺ�����ͨ����������Ŀռ������׼ȷ��
            bool isSwarm = hFeat->select_bins->isFull();
            histoSwarm   = new HistoGRAM(hFeat, nSamp);
            histoSwarm->CopyBins(*histo, false, 0x0);
            histoSwarm->RandomCompress(hFeat, isSwarm);
            histo = histoSwarm;
        }
        /*if (BIT_TEST(hFeat->type, FeatVector::AGGREGATE)) {
            throw "AGG_CheckGain is ...";		//��Ҫ�������
            //AGG_CheckGain(hData_, hFeat, flag);
        }
        else {
            vector<HistoGRAM *> moreHisto;
            histo->MoreHisto(hData_,moreHisto);
            for (size_t i = 0; i < moreHisto.size();i++) {
                moreHisto[i]->fruit = fruit;	// arrFruit[i];
                moreHisto[i]->GreedySplit_X(hData_, samp_set);
                delete moreHisto[i];
            }
            moreHisto.clear();
        }*/
        if (histoSwarm != nullptr) {
            delete histoSwarm;
        }
    }

    /*if (isEachFruit) {//EachFruit��Ҫ������ƣ���ʱ����
        pick_id = PickOnGain(hData_, arrFruit, flag);
        if (pick_id >= 0) {
            feat_id = picks[pick_id];
            //fruit = arrFruit[pick_id];
        }
    }
    else {
        feat_id = fruit->best_feat_id;
    }*/
    double mxmxN = 0;
    if (feat_id >= 0) {
        FeatVector *hFeat          = hData_->Feat(feat_id);
        hFeat->select.hasCheckGain = true;  // isSelect = true;
        fruit                      = new FRUIT(hData_, GetHistogram(hData_, feat_id, false));
        // fruit->Set(GetHistogram(hData_, feat_id, false));
        mxmxN        = fruit->mxmxN;
        feat_regress = -1;
        assert(mxmxN == fruit->mxmxN);
        // hFeat->UpdateFruit(hData_,this);
        // if (hFeat->hDistri != nullptr && hFeat->hDistri->rNA > 0) {
        // }
        if (hFeat->select_bins != nullptr) {
            hFeat->select_bins->AddCandSalp();
            hFeat->select_bins->SetCost(1);
        } /**/

    } else {  // ȷʵ�п���,�ܶ�������Խ�һ���Ż�����Ҫ����һЩ����������
        // printf("\n\t!!! Failed split at %d nSamp=%d nPick=%d !!!\n", id, nSample(), picks.size());
    }
    if (isEachFruit) {
        for (int i = 0; i < arrFruit.size(); i++) {  // Ϊ�˲���
            if (fruit == arrFruit[i]) {
                continue;
            } else
                delete arrFruit[i];
        }
    }
    arrFruit.clear();
    hData_->stat.nCheckGain++;
    double gain = 0;
    if (feat_id >= 0) {
        if (fruit->split_by == SPLIT_HISTOGRAM::BY_DENSITY) {
            int i = 0;
        }
        if (optimal == "lambda_0") {
            gain = FLOAT_ZERO(mxmxN - impuri, mxmxN);
            if (gain < 0) {
                if (hData_->config.lambda_l2 > 0) {
                    ;  // printf("\tgain(%d:%.8g)N=%d!!!", this->id, gain, nSamp);
                } else {
                    // Observation_AtSamp(hData_);
                    printf("\tgain(%d:%.8g)N=%ld!!!", this->id, gain, nSamp);
                    assert(gain >= 0);
                }
            }
            // return gMax;
        } else {
            double bst_imp = G2_sum - mxmxN;
            bst_imp        = FLOAT_ZERO(G2_sum - mxmxN, mxmxN);
            // if (bst_imp > -DBL_EPSILON*1000* Y2_sum && bst_imp < 0)	//Y2_sum,mxmxN���㷽ʽ��һ����ȷʵ���и������
            //	bst_imp = 0;
            gain = impuri - bst_imp;
            assert(gain >= 0);
            if (!(bst_imp >= 0 && bst_imp < impuri)) {
                printf("\n!!!! bst_imp=%5.3g impuri=%5.3g Y_sum_2=%5.3g mean*mean*N=%5.3g!!!!", bst_imp, impuri, G2_sum, mxmxN);
                // assert(0);
            }
        }
    }
    if (feat_id >= 0) {
        FeatVector *hFeat = hData_->Feat(feat_id);
        hFeat->wGain += gain;
        hFeat->wSplit += 1;
        hFeat->wSplit_last += 1;
        if (hFeat->wBins != nullptr) {
            int tic = fruit->bin_S0.tic;
            hFeat->wBins[tic] += 1;
            tic = fruit->bin_S1.tic;
            hFeat->wBins[tic] += 1;
        }

        /*if (hFeat->select_bins != nullptr) {
            hFeat->select_bins->AddCandSalp();
            hFeat->select_bins->SetCost(gain);
        }*/
    }
    FeatsOnFold::stat.tCheckGain += GST_TOC(tick);
    gain_train = gain;
    gain_      = gain_train;
    /*	���ѵ�����ûɶЧ��
        if ((this->Y_mean == 0 || this->Y_mean == 1)&& gain_>0) {
            //printf("gain_(%.2g,%.5g)\t", Y_mean, gain_);
            //gain_ /= 10;
        }
        double purity = max(Y_mean, 1 - Y_mean);
        gain_ *= (1 - purity);*/
    return gain_;
}

tpDOWN MT_BiSplit::GetDownStep() {
    assert(this->isLeaf());
    // assert(lr_eta == 1.0);
    return down_step * lr_eta;
}

void MT_BiSplit::Init_BFold(FeatsOnFold *hData_, int flag) {}