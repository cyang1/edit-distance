#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <algorithm>

// Maximum integer size is 64 bits
#define MAX_WORD_SIZE 64

void edit_distance_parallel(const char *body_s,
    const unsigned int body_len,
    const char *input_s,
    const unsigned int input_len,
    unsigned int& min_idx,
    unsigned int& min_distance)
{
    if (input_len > MAX_WORD_SIZE) {
        printf("This simple parallel implementation does not support search strings of length greater than %d.\n", MAX_WORD_SIZE);
        return;
    }

    const unsigned char *body = reinterpret_cast<const unsigned char*>(body_s),
                        *input = reinterpret_cast<const unsigned char*>(input_s);

    // Precompute Peq[\sigma]
    uint64_t char_eq[UCHAR_MAX];
    std::fill(char_eq, char_eq + UCHAR_MAX, 0);
    for (unsigned int i = 0; i < input_len; i++) {
        char_eq[input[i]] |= (1ull << i);
    }

    uint64_t pos_vert_diff = 0xFFFFFFFFFFFFFFFFLL;          // Pv = 1^m
    uint64_t neg_vert_diff = 0;                             // Mv = 0
    uint32_t score = input_len;                             // Score = m

    min_distance = input_len;
    min_idx = 0;

    for (unsigned int j = 0; j < body_len; j++) {
        uint64_t Eq = char_eq[body[j]];
        uint64_t Xv = Eq | neg_vert_diff;
        uint64_t Xh = (((Eq & pos_vert_diff) + pos_vert_diff) ^ pos_vert_diff) | Eq;

        uint64_t Ph = neg_vert_diff | ~(Xh | pos_vert_diff);
        uint64_t Mh = pos_vert_diff & Xh;

        if (Ph & (1ull << (input_len - 1)))
            score++;
        else if (Mh & (1ull << (input_len - 1)))
            score--;

        Ph <<= 1;
        Mh <<= 1;
        pos_vert_diff = Mh | ~(Xv | Ph);
        neg_vert_diff = Ph & Xv;

        if (score < min_distance) {
            min_distance = score;
            min_idx = j;
        }
    }
}
