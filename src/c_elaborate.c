#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

// Function to count words
int word_count(const string &s) {
    stringstream ss(s);
    string word;
    int count = 0;
    while (ss >> word) count++;
    return count;
}

int main() {

    FILE *fp = fopen("reviews_text.csv", "r");
    if (!fp) {
        printf("Error opening file\n");
        return 0;
    }

    unordered_map<string, pair<int,int>> mp;
    // reviewerID -> (count_of_elaborate_reviews, total_words)

    char line[20000];

while (fgets(line, sizeof(line), fp)) {

    string str(line);

    if (str.length() < 5) continue;

    int pos = str.find(',');
    if (pos == string::npos) continue;

    string reviewerID = str.substr(0, pos);
    string reviewText = str.substr(pos + 1);

    int wc = word_count(reviewText);

    printf("ID=%s WORDS=%d\n", reviewerID.c_str(), wc);

    if (wc >= 50) {
        mp[reviewerID].first += 1;
        mp[reviewerID].second += wc;
    }
}
    fclose(fp);

    printf("Elaborate Reviewers:\n");

    for (auto &it : mp) {
        int cnt = it.second.first;
        int total_words = it.second.second;

        if (cnt >= 5) {
            float avg = (float)total_words / cnt;
            printf("%s -> Reviews: %d, Avg Length: %.2f\n",
                   it.first.c_str(), cnt, avg);
        }
    }

    return 0;
}
