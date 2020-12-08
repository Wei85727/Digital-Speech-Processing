#include "../inc/hmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// initial
static int state_num;
static int time_num;
int observations[MAX_LINE];
char test_seq[MAX_LINE];
static double *Pi;
static double (*A)[MAX_STATE];
static double (*B)[MAX_STATE];
// calculated
static double delta[MAX_LINE][MAX_STATE];

static int max_d(int t, int j){
    double max = 0;
    int max_idx = 0; //
    for (int i = 0 ; i < state_num ; i++) {
		if (delta[t][i] * A[i][j] > max) {
			max = delta[t][i] * A[i][j];
            max_idx = i;
        }
	}
    return max_idx;//
}

// Viterbi Algorithm
static void delta_cal(){
    // Initialization
    for (int i = 0 ; i < state_num ; i++) {
		delta[0][i] = Pi[i] * B[observations[0]][i];
	}
    // Recursion
    for (int t = 1 ; t < time_num ; t++) {
		for (int j = 0 ; j < state_num ; j++) {
			int i = max_d(t-1, j);
			delta[t][j] = delta[t-1][i] * A[i][j] * B[observations[t]][j];;
		}
	}
}

static double max_delta(HMM *hmm){
    state_num = hmm->state_num;
	time_num = strlen(test_seq);
	Pi = hmm->initial;
	A = hmm->transition;
	B = hmm->observation;
    int length_seq = strlen(test_seq);
    for (int t = 0; t < length_seq; t++) {
        observations[t] = test_seq[t] - 'A';
    }
    memset(delta, 0, sizeof(delta));
    delta_cal(); // calculate delta
	int T = time_num - 1;
	double max_delta = 0;
	for (int i = 0 ; i < state_num ; i++) {
		if (delta[T][i] > max_delta) {
	        max_delta = delta[T][i];
	    }
    }
    return max_delta;
}

static void test(HMM *hmm, FILE * test_fp, FILE * result_fp, int model_num)
{
	while (fscanf(test_fp, "%s", test_seq) > 0) {
        double prob = 0;
		double max_prob = 0;
		int model_idx = 0;
		for (int k = 0 ; k < model_num ; k++) {	
            prob = max_delta(&hmm[k]);
            if (prob > max_prob) {
				max_prob = prob; // update the most possible model
				model_idx = k;
			}
		}
		fprintf(result_fp, "%s\n", hmm[model_idx].model_name); // printout result model and its probability and put a space between them
	}
}

int main(int argc, char *argv[]){
	char *models_list_path = argv[1];
	char *seq_path = argv[2];
	char *output_result_path = argv[3];
    HMM hmm[5]; // 5 models
	load_models(models_list_path , hmm, 5); // load model
	FILE *fp = fopen(seq_path, "r");    // read testing data
	FILE *result_fp = fopen(output_result_path, "w");
    // test
	test(hmm, fp, result_fp, 5);
    // finish
    fclose(fp);
    fclose(result_fp);

    return 0;
}