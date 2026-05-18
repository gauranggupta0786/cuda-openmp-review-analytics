#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX_LINE 20000
#define MAX_USERS 100000

typedef struct {
    int count;
    int totalWords;
} Stats;

Stats stats[MAX_USERS];

unsigned long hash(const char *str) {
    unsigned long h = 5381;
    int c;
    while ((c = *str++))
        h = ((h << 5) + h) + c;
    return h % MAX_USERS;
}

int word_count(const char *s) {
    int count = 0, inWord = 0;

    for (int i = 0; s[i]; i++) {
        if (s[i] != ' ' && s[i] != '\n' && s[i] != '\t' && s[i] != '\r') {
            if (!inWord) {
                count++;
                inWord = 1;
            }
        } else inWord = 0;
    }
    return count;
}

int main() {

    int thread_counts[] = {2, 4, 8, 16};

    for (int t = 0; t < 4; t++) {

        int threads = thread_counts[t];

        for (int i = 0; i < MAX_USERS; i++) {
            stats[i].count = 0;
            stats[i].totalWords = 0;
        }

        FILE *fp = fopen("reviews_text.csv", "r");
        if (!fp) {
            printf("File not found\n");
            return 0;
        }

        double start = omp_get_wtime();
        omp_set_num_threads(threads);

        char line[MAX_LINE];

        #pragma omp parallel
        {
            char local[MAX_LINE];

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < 2000000; i++) {

                #pragma omp critical
                {
                    if (!fgets(line, sizeof(line), fp)) {
                        line[0] = '\0';
                    } else {
                        strcpy(local, line);
                    }
                }

                if (line[0] == '\0') continue;

                char *comma = strchr(local, ',');
                if (!comma) continue;

                *comma = '\0';

                char *id = local;
                char *text = comma + 1;

                int wc = word_count(text);

                if (wc >= 50) {

                    unsigned long idx = hash(id);

                    #pragma omp atomic
                    stats[idx].count++;

                    #pragma omp atomic
                    stats[idx].totalWords += wc;
                }
            }
        }

        double end = omp_get_wtime();

        fclose(fp);

        printf("\nThreads = %d | Time = %f sec\n", threads, end - start);

        printf("Elaborate Reviewers:\n");

        for (int i = 0; i < MAX_USERS; i++) {
            if (stats[i].count >= 5) {
                printf("UserIndex %d -> Reviews: %d AvgLen: %.2f\n",
                       i,
                       stats[i].count,
                       (double)stats[i].totalWords / stats[i].count);
            }
        }
    }

    return 0;
}
