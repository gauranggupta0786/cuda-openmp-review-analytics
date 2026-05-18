#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

__global__ void kernel(int *movie, float *rating,
                       float *sum, int *count, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        atomicAdd(&sum[movie[i]], rating[i]);
        atomicAdd(&count[movie[i]], 1);
    }
}

int main() {

    FILE *fp = fopen("reviews.csv", "r");
    if (!fp) {
        printf("Error opening file\n");
        return 0;
    }

    unordered_map<string, int> idMap;
    vector<string> reverseMap;
    vector<int> movie;
    vector<float> rating;

    char line[10000];

    while (fgets(line, sizeof(line), fp)) {

        string str(line);
        stringstream ss(str);

        string asin, rating_str;

        getline(ss, asin, ',');
        getline(ss, rating_str, ',');

        if (asin == "" || rating_str == "") continue;

        asin.erase(remove(asin.begin(), asin.end(), '\n'), asin.end());
        asin.erase(remove(asin.begin(), asin.end(), '\r'), asin.end());

        float r;
        try { r = stof(rating_str); }
        catch (...) { continue; }

        if (idMap.find(asin) == idMap.end()) {
            int id = idMap.size();
            idMap[asin] = id;
            reverseMap.push_back(asin);
        }

        movie.push_back(idMap[asin]);
        rating.push_back(r);
    }

    fclose(fp);

    int n = movie.size();
    int numMovies = idMap.size();

    printf("Total reviews: %d\n", n);
    printf("Total movies: %d\n", numMovies);

    int *d_movie;
    float *d_rating, *d_sum;
    int *d_count;

    cudaMalloc(&d_movie, n * sizeof(int));
    cudaMalloc(&d_rating, n * sizeof(float));
    cudaMalloc(&d_sum, numMovies * sizeof(float));
    cudaMalloc(&d_count, numMovies * sizeof(int));

    cudaMemcpy(d_movie, movie.data(), n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rating, rating.data(), n*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_sum, 0, numMovies*sizeof(float));
    cudaMemset(d_count, 0, numMovies*sizeof(int));

    // ===== TIMING =====
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    kernel<<<(n+255)/256,256>>>(d_movie, d_rating, d_sum, d_count, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Baseline Time: %f ms\n", ms);

    // ===== COPY BACK =====
    float *h_sum = (float*)malloc(numMovies*sizeof(float));
    int *h_count = (int*)malloc(numMovies*sizeof(int));

    cudaMemcpy(h_sum, d_sum, numMovies*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_count, d_count, numMovies*sizeof(int), cudaMemcpyDeviceToHost);

    vector<pair<float,string>> avg;

    for (int i = 0; i < numMovies; i++) {
        if (h_count[i] >= 50) {
            float val = h_sum[i] / h_count[i];
            avg.push_back({val, reverseMap[i]});
        }
    }

    sort(avg.begin(), avg.end(), greater<pair<float,string>>());

    printf("\nTop 10 Movies:\n");
    for (int i = 0; i < 10 && i < avg.size(); i++) {
        printf("%s -> %f\n", avg[i].second.c_str(), avg[i].first);
    }

    cudaFree(d_movie);
    cudaFree(d_rating);
    cudaFree(d_sum);
    cudaFree(d_count);

    free(h_sum);
    free(h_count);

    return 0;
}
