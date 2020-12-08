#include <stdio.h>
#include <string.h>

// double accuracy(FILE * res_fp, FILE * ans_fp){
// 	char res[5];
// 	char ans[5];

// 	int all = 0;
// 	int same = 0;

// 	while ( fscanf(res_fp, "%s", res) != EOF &&
// 			 fscanf(ans_fp, "%s", ans) != EOF) {
// 		if (strcmp(res, ans) == 0) {
// 			same++;
// 		}
// 		all++;
// 		// double a = fscanf(res_fp, "%s", res);
//         // printf("Training time : %f secs \n", a);
// 	}
// 	// double acc = (double)same/all;
// 	double acc = (double)same/all;
// 	return acc;
// }

int main(int argc, char *argv[])
{
	FILE * acc_fp = fopen("acc.txt", "w");

	FILE * res_fp = fopen(argv[1], "r");
	FILE * ans_fp = fopen(argv[2], "r");

	char res[20];
	char ans[20];

	int all = 0;
	int same = 0;

	while ( (fscanf(res_fp, "%s", res) != EOF) &&
			 fscanf(ans_fp, "%s", ans) != EOF) {
		if (strcmp(res, ans) == 0) {
			same++;
		}
		all++;
		fscanf(res_fp, "%s", res);
	}

	double acc = (double)same/all;
	fprintf(acc_fp, "%.6lf", acc);

	fclose(acc_fp);
	fclose(res_fp);
	fclose(ans_fp);

	return 0;
}