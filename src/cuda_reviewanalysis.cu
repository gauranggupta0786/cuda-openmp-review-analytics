#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;

#define MAX_WORD 32

// =========================
// DEVICE STRING COMPARE
// =========================
__device__ int str_cmp(const char *a, const char *b) {
    while (*a && *b) {
        if (*a != *b) return 0;
        a++; b++;
    }
    return (*a == '\0' && *b == '\0');
}

// =========================
// SENTIMENT KERNEL
// =========================
__global__ void sentimentKernel(char *texts, int *offsets, int n,
                               char *lex_words, float *lex_scores,
                               int *lex_offsets, int lexSize,
                               float *results) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    char *text = texts + offsets[i];
    float score = 0.0f;

    char word[MAX_WORD];
    int idx = 0;

    for (int j = 0; text[j] != '\0'; j++) {

        if (text[j] == ' ' || text[j] == '\n') {

            word[idx] = '\0';

            // match with lexicon
            for (int k = 0; k < lexSize; k++) {
                char *lex_word = lex_words + lex_offsets[k];

                if (str_cmp(word, lex_word)) {
                    score += lex_scores[k];
                }
            }

            idx = 0;
        } else if (idx < MAX_WORD-1) {
            word[idx++] = text[j];
        }
    }

    results[i] = score;
}

// =========================
// MAIN
// =========================
int main() {

    // =========================
    // LOAD LEXICON
    // =========================
    vector<string> words;
    vector<float> scores;

    ifstream lexFile("vader_lexicon.txt");
    string line;

    while (getline(lexFile, line)) {
        stringstream ss(line);
        string w;
        float s;

        ss >> w >> s;
        words.push_back(w);
        scores.push_back(s);
    }

    int lexSize = words.size();
    printf("Lexicon loaded: %d words\n", lexSize);

    // flatten lexicon
    vector<char> flat_lex;
    vector<int> lex_offsets;

    for (int i = 0; i < lexSize; i++) {
        lex_offsets.push_back(flat_lex.size());
        for (char c : words[i]) flat_lex.push_back(c);
        flat_lex.push_back('\0');
    }

    // =========================
    // LOAD REVIEWS
    // =========================
    ifstream file("reviews_sentiment.csv");

    vector<string> texts;
    while (getline(file, line)) {
        texts.push_back(line);
    }

    int n = texts.size();
    printf("Total reviews: %d\n", n);

    // flatten reviews
    vector<char> flat;
    vector<int> offsets;

    for (int i = 0; i < n; i++) {
        offsets.push_back(flat.size());
        for (char c : texts[i]) flat.push_back(c);
        flat.push_back('\0');
    }

    // =========================
    // CUDA MEMORY
    // =========================
    char *d_texts, *d_lex_words;
    int *d_offsets, *d_lex_offsets;
    float *d_lex_scores, *d_results;

    cudaMalloc(&d_texts, flat.size());
    cudaMalloc(&d_offsets, n*sizeof(int));
    cudaMalloc(&d_results, n*sizeof(float));

    cudaMalloc(&d_lex_words, flat_lex.size());
    cudaMalloc(&d_lex_offsets, lexSize*sizeof(int));
    cudaMalloc(&d_lex_scores, lexSize*sizeof(float));

    cudaMemcpy(d_texts, flat.data(), flat.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_lex_words, flat_lex.data(), flat_lex.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lex_offsets, lex_offsets.data(), lexSize*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lex_scores, scores.data(), lexSize*sizeof(float), cudaMemcpyHostToDevice);

    // =========================
    // KERNEL
    // =========================
    sentimentKernel<<<(n+255)/256,256>>>(
        d_texts, d_offsets, n,
        d_lex_words, d_lex_scores, d_lex_offsets, lexSize,
        d_results
    );

    // =========================
    // COPY BACK
    // =========================
    vector<float> results(n);
    cudaMemcpy(results.data(), d_results, n*sizeof(float), cudaMemcpyDeviceToHost);

    // =========================
    // OUTPUT
    // =========================
    printf("\nSentiment Results:\n");

    for (int i = 0; i < n; i++) {
        if (results[i] > 0)
            printf("POSITIVE: %s\n", texts[i].c_str());
        else
            printf("NEGATIVE: %s\n", texts[i].c_str());
    }

    cudaFree(d_texts);
    cudaFree(d_offsets);
    cudaFree(d_results);
    cudaFree(d_lex_words);
    cudaFree(d_lex_offsets);
    cudaFree(d_lex_scores);

    return 0;
}
