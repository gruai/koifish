
template <class T>
__device__ inline void CU_X2T8_row(float* gama, T* row, char* terns, int N, int ldT = 64, bool isOverwrite = false, int idrow = -1) {
    float sum = 0.0f, a, average = 0.0f;
    for (int k = 0; k < N; k++) {
        a = CU_T2Float(row + k);
        sum += fabs(a);
    }

    average = (sum / (N));
    *gama   = average;
    // ta = (T)(average), tb = (T)(-average);
    for (int k = 0; k < N; k += ldT, terns++) {
        unsigned char tbyte = 0, bit;
        // #pragma unroll
        for (int bpos = 0; bpos < 8; bpos++, row += 8) {
            a = (CU_T2Float(row) + CU_T2Float(row + 1) + CU_T2Float(row + 2) + CU_T2Float(row + 3) + CU_T2Float(row + 4) + CU_T2Float(row + 5) +
                 CU_T2Float(row + 6) + CU_T2Float(row + 7)) /
                8.0;
            /*if (a > average / 2)
                bit = 1;
            else if (a < -average / 2)
                bit = 0;
            else {
                bit = bpos % 2 == 0;
            }*/
            bit = (-average / 2 <= a && a <= average / 2) ? 0 : 1;
            tbyte |= bit << (7 - bpos);
            if (isOverwrite) {
                // if(idrow==384){    // hack
                //     int debug=0;
                // }
                T a = 0;
                if (bit)
                    a = bpos % 2 == 0 ? average : -average;
                row[0] = a, row[1] = a, row[2] = a, row[3] = a, row[4] = a, row[5] = a, row[6] = a, row[7] = a;  //  0.015982857
            }
        }

        *terns = tbyte;
    }
}
template <class T>
__global__ static void CU_X2T8_(float* gama, T* mat0, char* terns, int M, int N, int ldT = 64, bool isOverwrite = false) {
    int tid = threadIdx.x, idrow = blockIdx.x * blockDim.x + tid, bit = 0;
    if (idrow >= M)
        return;  // guard
    // if(idrow==384){
    //     int debug=0;
    // }
    CU_X2T8_row(gama + idrow, mat0 + idrow * N, terns + (idrow * N) / ldT, N, ldT, isOverwrite, idrow);
}

template <class T>
__global__ void CU_T82X_(float* gama, const char* terns, T* mat0, int M, int N, int ldT = 64, int seed = 0x0) {
    int tid = threadIdx.x, idrow = blockIdx.x * blockDim.x + tid, bit = 0;
    if (idrow >= M)
        return;  // guard

    // if(idrow==384){
    //     int debug=0;
    // }
    float average = gama[idrow];
    T ta = (T)(average), tb = (T)(-average);
    // T ta = CU_Float2T<T>(average, seed), tb = CU_Float2T<T>(-average, seed);
    T* x8            = mat0 + idrow * N;
    const char* tern = terns + (idrow * N) / ldT;
    for (int k = 0; k < N; k += ldT, tern++) {
        unsigned char tbyte = *tern;  // terns[(idrow * N + k) / 8];
                                      // #pragma unroll
        for (int bpos = 0; bpos < 8; bpos++, x8 += 8) {
            bit = BYTE_bit(tbyte, bpos);
            // T a   = bit ? ta : tb;
            T a = 0;
            if (bit)
                a = bpos % 2 == 0 ? ta : tb;

            x8[0] = a, x8[1] = a, x8[2] = a, x8[3] = a, x8[4] = a, x8[5] = a, x8[6] = a, x8[7] = a;
        }
    }
}

template <typename T>
__device__ inline T ByteDot(const char& wA, T* B, int flag = 0x0) {
    T sum = (T)0.0;
    switch (wA) {
        case 0:;
            break;
        case 1:
            sum = B[7];
            break;
        case 2:
            sum = B[6];
            break;
        case 3:
            sum = B[6] + B[7];
            break;
        case 4:
            sum = B[5];
            break;
        case 5:
            sum = B[5] + B[7];
            break;
        case 6:
            sum = B[5] + B[6];
            break;
        case 7:
            sum = B[5] + B[6] + B[7];
            break;
        case 8:
            sum = B[4];
            break;
        case 9:
            sum = B[4] + B[7];
            break;
        case 10:
            sum = B[4] + B[6];
            break;
        case 11:
            sum = B[4] + B[6] + B[7];
            break;
        case 12:
            sum = B[4] + B[5];
            break;
        case 13:
            sum = B[4] + B[5] + B[7];
            break;
        case 14:
            sum = B[4] + B[5] + B[6];
            break;
        case 15:
            sum = B[4] + B[5] + B[6] + B[7];
            break;
        case 16:
            sum = B[3];
            break;
        case 17:
            sum = B[3] + B[7];
            break;
        case 18:
            sum = B[3] + B[6];
            break;
        case 19:
            sum = B[3] + B[6] + B[7];
            break;
        case 20:
            sum = B[3] + B[5];
            break;
        case 21:
            sum = B[3] + B[5] + B[7];
            break;
        case 22:
            sum = B[3] + B[5] + B[6];
            break;
        case 23:
            sum = B[3] + B[5] + B[6] + B[7];
            break;
        case 24:
            sum = B[3] + B[4];
            break;
        case 25:
            sum = B[3] + B[4] + B[7];
            break;
        case 26:
            sum = B[3] + B[4] + B[6];
            break;
        case 27:
            sum = B[3] + B[4] + B[6] + B[7];
            break;
        case 28:
            sum = B[3] + B[4] + B[5];
            break;
        case 29:
            sum = B[3] + B[4] + B[5] + B[7];
            break;
        case 30:
            sum = B[3] + B[4] + B[5] + B[6];
            break;
        case 31:
            sum = B[3] + B[4] + B[5] + B[6] + B[7];
            break;
        case 32:
            sum = B[2];
            break;
        case 33:
            sum = B[2] + B[7];
            break;
        case 34:
            sum = B[2] + B[6];
            break;
        case 35:
            sum = B[2] + B[6] + B[7];
            break;
        case 36:
            sum = B[2] + B[5];
            break;
        case 37:
            sum = B[2] + B[5] + B[7];
            break;
        case 38:
            sum = B[2] + B[5] + B[6];
            break;
        case 39:
            sum = B[2] + B[5] + B[6] + B[7];
            break;
        case 40:
            sum = B[2] + B[4];
            break;
        case 41:
            sum = B[2] + B[4] + B[7];
            break;
        case 42:
            sum = B[2] + B[4] + B[6];
            break;
        case 43:
            sum = B[2] + B[4] + B[6] + B[7];
            break;
        case 44:
            sum = B[2] + B[4] + B[5];
            break;
        case 45:
            sum = B[2] + B[4] + B[5] + B[7];
            break;
        case 46:
            sum = B[2] + B[4] + B[5] + B[6];
            break;
        case 47:
            sum = B[2] + B[4] + B[5] + B[6] + B[7];
            break;
        case 48:
            sum = B[2] + B[3];
            break;
        case 49:
            sum = B[2] + B[3] + B[7];
            break;
        case 50:
            sum = B[2] + B[3] + B[6];
            break;
        case 51:
            sum = B[2] + B[3] + B[6] + B[7];
            break;
        case 52:
            sum = B[2] + B[3] + B[5];
            break;
        case 53:
            sum = B[2] + B[3] + B[5] + B[7];
            break;
        case 54:
            sum = B[2] + B[3] + B[5] + B[6];
            break;
        case 55:
            sum = B[2] + B[3] + B[5] + B[6] + B[7];
            break;
        case 56:
            sum = B[2] + B[3] + B[4];
            break;
        case 57:
            sum = B[2] + B[3] + B[4] + B[7];
            break;
        case 58:
            sum = B[2] + B[3] + B[4] + B[6];
            break;
        case 59:
            sum = B[2] + B[3] + B[4] + B[6] + B[7];
            break;
        case 60:
            sum = B[2] + B[3] + B[4] + B[5];
            break;
        case 61:
            sum = B[2] + B[3] + B[4] + B[5] + B[7];
            break;
        case 62:
            sum = B[2] + B[3] + B[4] + B[5] + B[6];
            break;
        case 63:
            sum = B[2] + B[3] + B[4] + B[5] + B[6] + B[7];
            break;
        case 64:
            sum = B[1];
            break;
        case 65:
            sum = B[1] + B[7];
            break;
        case 66:
            sum = B[1] + B[6];
            break;
        case 67:
            sum = B[1] + B[6] + B[7];
            break;
        case 68:
            sum = B[1] + B[5];
            break;
        case 69:
            sum = B[1] + B[5] + B[7];
            break;
        case 70:
            sum = B[1] + B[5] + B[6];
            break;
        case 71:
            sum = B[1] + B[5] + B[6] + B[7];
            break;
        case 72:
            sum = B[1] + B[4];
            break;
        case 73:
            sum = B[1] + B[4] + B[7];
            break;
        case 74:
            sum = B[1] + B[4] + B[6];
            break;
        case 75:
            sum = B[1] + B[4] + B[6] + B[7];
            break;
        case 76:
            sum = B[1] + B[4] + B[5];
            break;
        case 77:
            sum = B[1] + B[4] + B[5] + B[7];
            break;
        case 78:
            sum = B[1] + B[4] + B[5] + B[6];
            break;
        case 79:
            sum = B[1] + B[4] + B[5] + B[6] + B[7];
            break;
        case 80:
            sum = B[1] + B[3];
            break;
        case 81:
            sum = B[1] + B[3] + B[7];
            break;
        case 82:
            sum = B[1] + B[3] + B[6];
            break;
        case 83:
            sum = B[1] + B[3] + B[6] + B[7];
            break;
        case 84:
            sum = B[1] + B[3] + B[5];
            break;
        case 85:
            sum = B[1] + B[3] + B[5] + B[7];
            break;
        case 86:
            sum = B[1] + B[3] + B[5] + B[6];
            break;
        case 87:
            sum = B[1] + B[3] + B[5] + B[6] + B[7];
            break;
        case 88:
            sum = B[1] + B[3] + B[4];
            break;
        case 89:
            sum = B[1] + B[3] + B[4] + B[7];
            break;
        case 90:
            sum = B[1] + B[3] + B[4] + B[6];
            break;
        case 91:
            sum = B[1] + B[3] + B[4] + B[6] + B[7];
            break;
        case 92:
            sum = B[1] + B[3] + B[4] + B[5];
            break;
        case 93:
            sum = B[1] + B[3] + B[4] + B[5] + B[7];
            break;
        case 94:
            sum = B[1] + B[3] + B[4] + B[5] + B[6];
            break;
        case 95:
            sum = B[1] + B[3] + B[4] + B[5] + B[6] + B[7];
            break;
        case 96:
            sum = B[1] + B[2];
            break;
        case 97:
            sum = B[1] + B[2] + B[7];
            break;
        case 98:
            sum = B[1] + B[2] + B[6];
            break;
        case 99:
            sum = B[1] + B[2] + B[6] + B[7];
            break;
        case 100:
            sum = B[1] + B[2] + B[5];
            break;
        case 101:
            sum = B[1] + B[2] + B[5] + B[7];
            break;
        case 102:
            sum = B[1] + B[2] + B[5] + B[6];
            break;
        case 103:
            sum = B[1] + B[2] + B[5] + B[6] + B[7];
            break;
        case 104:
            sum = B[1] + B[2] + B[4];
            break;
        case 105:
            sum = B[1] + B[2] + B[4] + B[7];
            break;
        case 106:
            sum = B[1] + B[2] + B[4] + B[6];
            break;
        case 107:
            sum = B[1] + B[2] + B[4] + B[6] + B[7];
            break;
        case 108:
            sum = B[1] + B[2] + B[4] + B[5];
            break;
        case 109:
            sum = B[1] + B[2] + B[4] + B[5] + B[7];
            break;
        case 110:
            sum = B[1] + B[2] + B[4] + B[5] + B[6];
            break;
        case 111:
            sum = B[1] + B[2] + B[4] + B[5] + B[6] + B[7];
            break;
        case 112:
            sum = B[1] + B[2] + B[3];
            break;
        case 113:
            sum = B[1] + B[2] + B[3] + B[7];
            break;
        case 114:
            sum = B[1] + B[2] + B[3] + B[6];
            break;
        case 115:
            sum = B[1] + B[2] + B[3] + B[6] + B[7];
            break;
        case 116:
            sum = B[1] + B[2] + B[3] + B[5];
            break;
        case 117:
            sum = B[1] + B[2] + B[3] + B[5] + B[7];
            break;
        case 118:
            sum = B[1] + B[2] + B[3] + B[5] + B[6];
            break;
        case 119:
            sum = B[1] + B[2] + B[3] + B[5] + B[6] + B[7];
            break;
        case 120:
            sum = B[1] + B[2] + B[3] + B[4];
            break;
        case 121:
            sum = B[1] + B[2] + B[3] + B[4] + B[7];
            break;
        case 122:
            sum = B[1] + B[2] + B[3] + B[4] + B[6];
            break;
        case 123:
            sum = B[1] + B[2] + B[3] + B[4] + B[6] + B[7];
            break;
        case 124:
            sum = B[1] + B[2] + B[3] + B[4] + B[5];
            break;
        case 125:
            sum = B[1] + B[2] + B[3] + B[4] + B[5] + B[7];
            break;
        case 126:
            sum = B[1] + B[2] + B[3] + B[4] + B[5] + B[6];
            break;
        case 127:
            sum = B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];
            break;
        case 128:
            sum = B[0];
            break;
        case 129:
            sum = B[0] + B[7];
            break;
        case 130:
            sum = B[0] + B[6];
            break;
        case 131:
            sum = B[0] + B[6] + B[7];
            break;
        case 132:
            sum = B[0] + B[5];
            break;
        case 133:
            sum = B[0] + B[5] + B[7];
            break;
        case 134:
            sum = B[0] + B[5] + B[6];
            break;
        case 135:
            sum = B[0] + B[5] + B[6] + B[7];
            break;
        case 136:
            sum = B[0] + B[4];
            break;
        case 137:
            sum = B[0] + B[4] + B[7];
            break;
        case 138:
            sum = B[0] + B[4] + B[6];
            break;
        case 139:
            sum = B[0] + B[4] + B[6] + B[7];
            break;
        case 140:
            sum = B[0] + B[4] + B[5];
            break;
        case 141:
            sum = B[0] + B[4] + B[5] + B[7];
            break;
        case 142:
            sum = B[0] + B[4] + B[5] + B[6];
            break;
        case 143:
            sum = B[0] + B[4] + B[5] + B[6] + B[7];
            break;
        case 144:
            sum = B[0] + B[3];
            break;
        case 145:
            sum = B[0] + B[3] + B[7];
            break;
        case 146:
            sum = B[0] + B[3] + B[6];
            break;
        case 147:
            sum = B[0] + B[3] + B[6] + B[7];
            break;
        case 148:
            sum = B[0] + B[3] + B[5];
            break;
        case 149:
            sum = B[0] + B[3] + B[5] + B[7];
            break;
        case 150:
            sum = B[0] + B[3] + B[5] + B[6];
            break;
        case 151:
            sum = B[0] + B[3] + B[5] + B[6] + B[7];
            break;
        case 152:
            sum = B[0] + B[3] + B[4];
            break;
        case 153:
            sum = B[0] + B[3] + B[4] + B[7];
            break;
        case 154:
            sum = B[0] + B[3] + B[4] + B[6];
            break;
        case 155:
            sum = B[0] + B[3] + B[4] + B[6] + B[7];
            break;
        case 156:
            sum = B[0] + B[3] + B[4] + B[5];
            break;
        case 157:
            sum = B[0] + B[3] + B[4] + B[5] + B[7];
            break;
        case 158:
            sum = B[0] + B[3] + B[4] + B[5] + B[6];
            break;
        case 159:
            sum = B[0] + B[3] + B[4] + B[5] + B[6] + B[7];
            break;
        case 160:
            sum = B[0] + B[2];
            break;
        case 161:
            sum = B[0] + B[2] + B[7];
            break;
        case 162:
            sum = B[0] + B[2] + B[6];
            break;
        case 163:
            sum = B[0] + B[2] + B[6] + B[7];
            break;
        case 164:
            sum = B[0] + B[2] + B[5];
            break;
        case 165:
            sum = B[0] + B[2] + B[5] + B[7];
            break;
        case 166:
            sum = B[0] + B[2] + B[5] + B[6];
            break;
        case 167:
            sum = B[0] + B[2] + B[5] + B[6] + B[7];
            break;
        case 168:
            sum = B[0] + B[2] + B[4];
            break;
        case 169:
            sum = B[0] + B[2] + B[4] + B[7];
            break;
        case 170:
            sum = B[0] + B[2] + B[4] + B[6];
            break;
        case 171:
            sum = B[0] + B[2] + B[4] + B[6] + B[7];
            break;
        case 172:
            sum = B[0] + B[2] + B[4] + B[5];
            break;
        case 173:
            sum = B[0] + B[2] + B[4] + B[5] + B[7];
            break;
        case 174:
            sum = B[0] + B[2] + B[4] + B[5] + B[6];
            break;
        case 175:
            sum = B[0] + B[2] + B[4] + B[5] + B[6] + B[7];
            break;
        case 176:
            sum = B[0] + B[2] + B[3];
            break;
        case 177:
            sum = B[0] + B[2] + B[3] + B[7];
            break;
        case 178:
            sum = B[0] + B[2] + B[3] + B[6];
            break;
        case 179:
            sum = B[0] + B[2] + B[3] + B[6] + B[7];
            break;
        case 180:
            sum = B[0] + B[2] + B[3] + B[5];
            break;
        case 181:
            sum = B[0] + B[2] + B[3] + B[5] + B[7];
            break;
        case 182:
            sum = B[0] + B[2] + B[3] + B[5] + B[6];
            break;
        case 183:
            sum = B[0] + B[2] + B[3] + B[5] + B[6] + B[7];
            break;
        case 184:
            sum = B[0] + B[2] + B[3] + B[4];
            break;
        case 185:
            sum = B[0] + B[2] + B[3] + B[4] + B[7];
            break;
        case 186:
            sum = B[0] + B[2] + B[3] + B[4] + B[6];
            break;
        case 187:
            sum = B[0] + B[2] + B[3] + B[4] + B[6] + B[7];
            break;
        case 188:
            sum = B[0] + B[2] + B[3] + B[4] + B[5];
            break;
        case 189:
            sum = B[0] + B[2] + B[3] + B[4] + B[5] + B[7];
            break;
        case 190:
            sum = B[0] + B[2] + B[3] + B[4] + B[5] + B[6];
            break;
        case 191:
            sum = B[0] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];
            break;
        case 192:
            sum = B[0] + B[1];
            break;
        case 193:
            sum = B[0] + B[1] + B[7];
            break;
        case 194:
            sum = B[0] + B[1] + B[6];
            break;
        case 195:
            sum = B[0] + B[1] + B[6] + B[7];
            break;
        case 196:
            sum = B[0] + B[1] + B[5];
            break;
        case 197:
            sum = B[0] + B[1] + B[5] + B[7];
            break;
        case 198:
            sum = B[0] + B[1] + B[5] + B[6];
            break;
        case 199:
            sum = B[0] + B[1] + B[5] + B[6] + B[7];
            break;
        case 200:
            sum = B[0] + B[1] + B[4];
            break;
        case 201:
            sum = B[0] + B[1] + B[4] + B[7];
            break;
        case 202:
            sum = B[0] + B[1] + B[4] + B[6];
            break;
        case 203:
            sum = B[0] + B[1] + B[4] + B[6] + B[7];
            break;
        case 204:
            sum = B[0] + B[1] + B[4] + B[5];
            break;
        case 205:
            sum = B[0] + B[1] + B[4] + B[5] + B[7];
            break;
        case 206:
            sum = B[0] + B[1] + B[4] + B[5] + B[6];
            break;
        case 207:
            sum = B[0] + B[1] + B[4] + B[5] + B[6] + B[7];
            break;
        case 208:
            sum = B[0] + B[1] + B[3];
            break;
        case 209:
            sum = B[0] + B[1] + B[3] + B[7];
            break;
        case 210:
            sum = B[0] + B[1] + B[3] + B[6];
            break;
        case 211:
            sum = B[0] + B[1] + B[3] + B[6] + B[7];
            break;
        case 212:
            sum = B[0] + B[1] + B[3] + B[5];
            break;
        case 213:
            sum = B[0] + B[1] + B[3] + B[5] + B[7];
            break;
        case 214:
            sum = B[0] + B[1] + B[3] + B[5] + B[6];
            break;
        case 215:
            sum = B[0] + B[1] + B[3] + B[5] + B[6] + B[7];
            break;
        case 216:
            sum = B[0] + B[1] + B[3] + B[4];
            break;
        case 217:
            sum = B[0] + B[1] + B[3] + B[4] + B[7];
            break;
        case 218:
            sum = B[0] + B[1] + B[3] + B[4] + B[6];
            break;
        case 219:
            sum = B[0] + B[1] + B[3] + B[4] + B[6] + B[7];
            break;
        case 220:
            sum = B[0] + B[1] + B[3] + B[4] + B[5];
            break;
        case 221:
            sum = B[0] + B[1] + B[3] + B[4] + B[5] + B[7];
            break;
        case 222:
            sum = B[0] + B[1] + B[3] + B[4] + B[5] + B[6];
            break;
        case 223:
            sum = B[0] + B[1] + B[3] + B[4] + B[5] + B[6] + B[7];
            break;
        case 224:
            sum = B[0] + B[1] + B[2];
            break;
        case 225:
            sum = B[0] + B[1] + B[2] + B[7];
            break;
        case 226:
            sum = B[0] + B[1] + B[2] + B[6];
            break;
        case 227:
            sum = B[0] + B[1] + B[2] + B[6] + B[7];
            break;
        case 228:
            sum = B[0] + B[1] + B[2] + B[5];
            break;
        case 229:
            sum = B[0] + B[1] + B[2] + B[5] + B[7];
            break;
        case 230:
            sum = B[0] + B[1] + B[2] + B[5] + B[6];
            break;
        case 231:
            sum = B[0] + B[1] + B[2] + B[5] + B[6] + B[7];
            break;
        case 232:
            sum = B[0] + B[1] + B[2] + B[4];
            break;
        case 233:
            sum = B[0] + B[1] + B[2] + B[4] + B[7];
            break;
        case 234:
            sum = B[0] + B[1] + B[2] + B[4] + B[6];
            break;
        case 235:
            sum = B[0] + B[1] + B[2] + B[4] + B[6] + B[7];
            break;
        case 236:
            sum = B[0] + B[1] + B[2] + B[4] + B[5];
            break;
        case 237:
            sum = B[0] + B[1] + B[2] + B[4] + B[5] + B[7];
            break;
        case 238:
            sum = B[0] + B[1] + B[2] + B[4] + B[5] + B[6];
            break;
        case 239:
            sum = B[0] + B[1] + B[2] + B[4] + B[5] + B[6] + B[7];
            break;
        case 240:
            sum = B[0] + B[1] + B[2] + B[3];
            break;
        case 241:
            sum = B[0] + B[1] + B[2] + B[3] + B[7];
            break;
        case 242:
            sum = B[0] + B[1] + B[2] + B[3] + B[6];
            break;
        case 243:
            sum = B[0] + B[1] + B[2] + B[3] + B[6] + B[7];
            break;
        case 244:
            sum = B[0] + B[1] + B[2] + B[3] + B[5];
            break;
        case 245:
            sum = B[0] + B[1] + B[2] + B[3] + B[5] + B[7];
            break;
        case 246:
            sum = B[0] + B[1] + B[2] + B[3] + B[5] + B[6];
            break;
        case 247:
            sum = B[0] + B[1] + B[2] + B[3] + B[5] + B[6] + B[7];
            break;
        case 248:
            sum = B[0] + B[1] + B[2] + B[3] + B[4];
            break;
        case 249:
            sum = B[0] + B[1] + B[2] + B[3] + B[4] + B[7];
            break;
        case 250:
            sum = B[0] + B[1] + B[2] + B[3] + B[4] + B[6];
            break;
        case 251:
            sum = B[0] + B[1] + B[2] + B[3] + B[4] + B[6] + B[7];
            break;
        case 252:
            sum = B[0] + B[1] + B[2] + B[3] + B[4] + B[5];
            break;
        case 253:
            sum = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[7];
            break;
        case 254:
            sum = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6];
            break;
        case 255:
            sum = B[0] + B[1] + B[2] + B[3] + B[4] + B[5] + B[6] + B[7];
            break;
    }
    return sum;
}

// only for debug
template <typename T>
__device__ inline T AT_(T* mat, int r, int c, int M, int N) {
#ifndef NDEBUG
    if (r >= M) {
        return T(0);
    }
    if (c >= N) {
        return T(0);
    }
#endif
    return mat[c * M + r];
}

//  just like mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
// template <typename Ta, typename Tb, typename Tc, const int BM, const int BN, const int BK = 16, const int TM = 8, const int TN = 8>
// __device__ inline void SYNC_AtBC_m8n8k16(Ta *tileA, Tb *tileB, Tc (&sum)[TM][TN], int flag = 0x0) {
//     float a_frag[TM] = {0.}, b_frag[TN] = {0.};
// #pragma unroll
//     for (int i = 0; i < BK; i++) {
// #pragma unroll
//         for (int j = 0; j < TM; j++) {                // i-th column:   r=ty+j,c=i
//             a_frag[j] = tileA[CR2POS(j, i, BM, BK)];  // (ty + j) * BK + i
//         }
// #pragma unroll
//         for (int l = 0; l < TN; l++) {                // i-th row:   r=i,c=tx + l
//             b_frag[l] = tileB[RC2POS(i, l, BK, BN)];  // tx + l + i * BN
//         }
//         // #pragma unroll
//         // 		for (int j = 0; j < TM; j++) {
//         // #pragma unroll
//         // 			for (int l = 0; l < TN; l++) sum[j][l] += a_frag[j] * b_frag[l];
//         // 		}
//     }
// }
#define UNROLL _Pragma("unroll")
//  just like mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
#define SYNC_AtBC_m8n8k16(a_, b_, sum)                                                                        \
    UNROLL for (int i = 0; i < 16; i++) {                                                                     \
        UNROLL for (int j = 0; j < 8; j++) { a_[j] = tileA[CR2POS(j, i, BM, BK)]; }                           \
        UNROLL for (int l = 0; l < 8; l++) { b_[l] = tileB[RC2POS(i, l, Bk, BN)]; }                           \
        UNROLL for (int j = 0; j < 8; j++) { UNROLL for (int l = 0; l < 8; l++) sum[j][l] += a_[j] * b_[l]; } \
    }

// Register=>Global memory
// #define SYNC_REG2M_m8n8(a_,b_,sum)   	\
// UNROLL for (int l = 0; l < 8; l++) {   \ 
// UNROLL for (int j = 0; j < 8; j++) {	\
// 			packed_out[j] = tmp[j][l];	\
// 		}	\
// 		store128(C+CR2POS(ty , tx+l, M, N), packed_out); \
// 	}

// #pragma unroll
//     for (int j = 0; j < TM; j++) {
//         #pragma unroll
//         for (int l = 0; l < TN; l++) C[CR2POS(ty + j, tx + l, M, N)] = tmp[j][l];  //[(ty + j) * N + tx + l]
//     }
