#include "../inc/hmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// initial
int state_num;
int observation_num;
int time_num;
int observations[MAX_LINE];
char train_seq[MAX_LINE];
double *Pi;
double (*A)[MAX_STATE];
double (*B)[MAX_STATE];
// calculated
double alpha[MAX_LINE][MAX_STATE];	
double beta[MAX_LINE][MAX_STATE];
double sum_gamma[MAX_STATE];
double sum_gamma_observation[MAX_OBSERV][MAX_STATE];
double sum_epsilon[MAX_STATE][MAX_STATE];
double Pi_re[MAX_STATE];	
int sample_num;

static void alpha_cal(){
// Initialization
for (int i = 0 ; i < state_num ; i++){
    alpha[0][i] = Pi[i] * B[observations[0]][i];
}
// Induction
for (int t = 1 ; t <= time_num ; t++) {
		for (int j = 0 ; j < state_num ; j++) {
			// do sigma
			double sum = 0;
			for (int i = 0 ; i < state_num ; i++)
				sum += alpha[t-1][i] * A[i][j];
			alpha[t][j] = sum * B[observations[t]][j];
		}
	}
}

static void beta_cal(){
// Initailization
	int T = time_num-1;
	for (int i = 0 ; i < state_num ; i++){
		beta[T][i] = 1;
	}
// Induction
	for (int t = T-1 ; t >= 0 ; t--) {
		for (int i = 0 ; i < state_num ; i++) {
            // do sigma
			double sum = 0;
			for (int j = 0 ; j < state_num ; j++)
				sum += A[i][j] * B[observations[t+1]][j] * beta[t+1][j];
			beta[t][i] = sum;
		}
	}
}

static void gamma_cal(){
	double gamma[state_num];
	for (int t = 0 ; t < time_num ; t++) {
		double sum = 0;
		for (int i = 0 ; i < state_num ; i++){
			gamma[i] = alpha[t][i] * beta[t][i];			
			sum += gamma[i];			
		}
		for (int i = 0 ; i < state_num ; i++){
			gamma[i] /= sum;					
			sum_gamma[i] += gamma[i];					//sum of gamma
			sum_gamma_observation[observations[t]][i] += gamma[i];	//sum of gamma with observation
		}
		// re-estimate Pi
		if (t == 0) {
			for (int i = 0 ; i < state_num ; i++) {
				Pi_re[i] += gamma[i];
			}
		}
	}
}

static void epsilon_cal(){
	double epsilon[state_num][state_num];
	for (int t = 0 ; t < time_num ; t++) {
		double sum = 0;
		for (int i = 0 ; i < state_num ; i++) {
			for (int j = 0 ; j < state_num ; j++) {
				epsilon[i][j] = alpha[t][i] * A[i][j] * B[observations[t+1]][j] * beta[t+1][j];
				sum += epsilon[i][j];									
			}
		}
		for (int i = 0 ; i < state_num ; i++) {
			for (int j = 0 ; j < state_num ; j++) {
				if (sum != 0) {
				    epsilon[i][j] /= sum;
				}		
				sum_epsilon[i][j] += epsilon[i][j];	//　sum of epsilon
			}
		}
	}
}

static void reset(FILE *train_fp){
	memset(Pi_re, 0, sizeof(double) * MAX_STATE);
	memset(sum_gamma, 0, sizeof(double) * MAX_STATE);
	memset(sum_gamma_observation, 0, sizeof(double) * MAX_OBSERV * MAX_STATE);
	memset(sum_epsilon, 0, sizeof(double) * MAX_STATE * MAX_STATE);
	sample_num = 0;
	fseek(train_fp,  0,  SEEK_SET);
}

static void train(HMM *hmm, FILE *train_fp, int iter){
    state_num = hmm->state_num;
	observation_num = hmm->observ_num;
	time_num = 0;
	Pi = hmm->initial;
	A = hmm->transition;
	B = hmm->observation;
    // reset before training
	reset(train_fp);
	for (int i = 0 ; i < iter ; i++) {
		sample_num = 0;
		while (fscanf(train_fp, "%s", train_seq) > 0) {
			memset(alpha, 0, sizeof(alpha)); // reset alpha
			memset(beta, 0, sizeof(beta)); // reset beta
			if (sample_num == 0)
				time_num = strlen(train_seq);
			int length_seq = strlen(train_seq);
			for (int t = 0; t < length_seq; t++) {
                observations[t] = train_seq[t] - 'A';
            }
			alpha_cal();
			beta_cal();
			gamma_cal();
			epsilon_cal();

			sample_num++;
		}
		// update parameters
		for (int i = 0 ; i < state_num ; i++){
		    Pi[i] = Pi_re[i] / sample_num;
		}
		for (int i = 0 ; i < state_num ; i++) {
		    for (int j = 0 ; j < state_num ; j++) {
				A[i][j] = sum_epsilon[i][j] / sum_gamma[i];
		    }
	    }
		for (int ob = 0 ; ob < observation_num ; ob++) {
		    for (int i = 0 ; i < state_num ; i++) {
			    B[ob][i] = sum_gamma_observation[ob][i] / sum_gamma[i];
		    }
	    }
		reset(train_fp);
	}
}

int main(int argc, char *argv[]){   
	clock_t start_time, end_time;
	int iter = atoi(argv[1]);
	char *model_init_path = argv[2];
	char *seq_path = argv[3];
	char *output_model_path = argv[4];
	HMM hmm;
	loadHMM(&hmm, model_init_path); // load model
	FILE *fp = open_or_die(seq_path, "r");	// read training data
 
	start_time = clock();
	// train
	train(&hmm, fp, iter);
	// finish
    end_time = clock();
	float total_time = (end_time - start_time)/CLOCKS_PER_SEC;
	printf("Training time : %f secs \n", total_time);

	FILE* output_model_file = fopen(output_model_path, "w");
	dumpHMM(output_model_file, &hmm);
	fclose(output_model_file);
	
	return 0;
}
