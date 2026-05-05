#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <cstring>

class MD5
{
private:
    uint32_t a, b, c, d;
    uint64_t totalBits;
    uint8_t buffer[64];

    static constexpr uint32_t shift1[4] = {7, 12, 17, 22};
    static constexpr uint32_t shift2[4] = {5, 9, 14, 20};
    static constexpr uint32_t shift3[4] = {4, 11, 16, 23};
    static constexpr uint32_t shift4[4] = {6, 10, 15, 21};

    static uint32_t rotateLeft(uint32_t val, uint32_t n)
    {
        return (val << n) | (val >> (32 - n));
    }

    void transform(const uint8_t* block);
    void encode(uint8_t* output, const uint32_t* input, size_t length);
    void decode(uint32_t* output, const uint8_t* input, size_t length);

public:
    MD5();
    void update(const uint8_t* data, size_t len);
    std::string digest();
};

MD5::MD5()
{
    a = 0x67452301;
    b = 0xEFCDAB89;
    c = 0x98BADCFE;
    d = 0x10325476;
    totalBits = 0;
    std::memset(buffer, 0, sizeof(buffer));
}

void MD5::decode(uint32_t* out, const uint8_t* in, size_t len)
{
    for (size_t i = 0, j = 0; j < len; i++, j += 4)
        out[i] = (uint32_t)in[j] | ((uint32_t)in[j+1] << 8) | ((uint32_t)in[j+2] << 16) | ((uint32_t)in[j+3] << 24);
}

void MD5::encode(uint8_t* out, const uint32_t* in, size_t len)
{
    for (size_t i = 0, j = 0; j < len; i++, j += 4)
    {
        out[j]     = static_cast<uint8_t>(in[i]);
        out[j + 1] = static_cast<uint8_t>(in[i] >> 8);
        out[j + 2] = static_cast<uint8_t>(in[i] >> 16);
        out[j + 3] = static_cast<uint8_t>(in[i] >> 24);
    }
}

void MD5::transform(const uint8_t* block)
{
    uint32_t w[16];
    decode(w, block, 64);
    uint32_t A = a, B = b, C = c, D = d;

    // Round 1
    for (int i = 0; i < 16; ++i)
    {
        uint32_t f = (B & C) | (~B & D);
        uint32_t temp = D;
        D = C;
        C = B;
        B += rotateLeft(A + f + w[i] + 0xD76AA478, shift1[i % 4]);
        A = temp;
    }

    a += A; b += B; c += C; d += D;
}

void MD5::update(const uint8_t* data, size_t len)
{
    size_t bufPos = totalBits % 64;
    totalBits += static_cast<uint64_t>(len) * 8;

    if (bufPos + len >= 64)
    {
        std::memcpy(buffer + bufPos, data, 64 - bufPos);
        transform(buffer);
        size_t i = 64 - bufPos;
        for (; i + 63 < len; i += 64)
            transform(data + i);
        bufPos = len - i;
        std::memcpy(buffer, data + i, bufPos);
    }
    else
    {
        std::memcpy(buffer + bufPos, data, len);
    }
}

std::string MD5::digest()
{
    uint8_t padding[64] = { 0x80 };
    size_t padLen = (totalBits % 512) / 8;
    padLen = (padLen < 56) ? (56 - padLen) : (120 - padLen);
    update(padding, padLen);

    uint8_t bitLen[8];
    encode(bitLen, reinterpret_cast<uint32_t*>(&totalBits), 8);
    update(bitLen, 8);

    uint8_t hash[16];
    encode(hash, reinterpret_cast<uint32_t*>(&a), 16);

    const char* hexTable = "0123456789abcdef";
    std::string res;
    for (int i = 0; i < 16; ++i)
    {
        res += hexTable[(hash[i] >> 4) & 0x0F];
        res += hexTable[hash[i] & 0x0F];
    }
    return res;
}

// Core function: get file MD5
std::string getFileMD5(const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) return "";

    MD5 md5;
    char buf[4096];
    while (file.read(buf, sizeof(buf)))
    {
        md5.update(reinterpret_cast<uint8_t*>(buf), file.gcount());
    }
    md5.update(reinterpret_cast<uint8_t*>(buf), file.gcount());
    return md5.digest();
}

