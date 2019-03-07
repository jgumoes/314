// This file is ports the find_ratio_flat function into C.
// This should make the shuffle function way, way quicker than it is right now.
// I might put other stuff in here if the main script still isn't fast enough,
// but I don't think I will.

#include <math.h>
#include <stdio.h>


double find_ratio(int fP[], int len_P){
    // the original find_ratio_flat only accepts a numpy array, but it was easiest to make add_bin add straight to an existing array
    // so, I split off the bulk of find_ratio_flat into this function, which accepts a C array, making life way easier while still keeping
    // find_ratio_flat but without bloating the dll
    // Except, as it turns out, this works just fine with numpy. So yeah.
    /*
    double find_ratio_flat(){    //this line is for testing the body of code

    int len_P = 9;
    int fP[] = {123, 135, 146, 158, 168, 178, 188, 198, 207, 249, 247, 244, 240, 235, 230, 223, 215};
    */

    int pX[len_P];
    for(int pos_count = 0; pos_count<len_P; pos_count++){
        pX[pos_count] = fP[pos_count];
    }
    int pY[len_P];
    pY[0] = 250;
    for(int pos_count = 0; pos_count<len_P-1; pos_count++){
        pY[pos_count+1] = fP[pos_count+len_P];
    }

    /* find lengths */
    int no_i = (pow(pY[len_P-1]-pX[len_P-1], 2) > 0);   
    double perimeter =  fabs((double) pX[0]*8);

    if(len_P > 1){
        for(int i = 0; i < len_P-1; i++){
            int X = pX[i+1] - pX[i];
            int Y = pY[i+1] - pY[i];
            perimeter += sqrt(64*(X*X + Y*Y));
        }
    }

    if(no_i==1){
        int Y = pY[len_P-1];
        int X = pX[len_P-1];
        perimeter += sqrt((double) (Y*Y + X*X - 2*Y*X)*32);
    }

    /*find area*/
    int xe[len_P*2 + 2];
    int ye[len_P*2 + 2];
    ye[0] = 250;
    xe[len_P*2 + 1] = 250;

    xe[0] = 0;
    ye[len_P*2 + 1] = 0;
    
    for(int i = 0; i<len_P; i++){
        xe[i+1] = pX[i];
        xe[i+len_P+1] = pY[len_P-1-i];

        ye[i+1] = pY[i];
        ye[i+len_P+1] = pX[len_P-1-i];
    }

    long area = 0;
    for(int i = 0; i <= len_P*2 - 1; i++){
        area += (xe[i+1]-xe[i])*(ye[i+1]+ye[i]);
    }
    area = area*2;

    /*find ratio*/
    double ratio = (area/perimeter);
    return(ratio);
}

double find_ratio_flat(const void * flat_P, int len_P){
    // same function as in the main script, ported to C
    // accepts a flattened P as a numpy array, same as
    // the function in the main script
    // len_P is the number of coordinates in P,
    // so there would be len_P x values and len_P-1 y values
    // in flat_P
    // note: find_ratio_flat only accepts a numpy array, but it was easiest to make add_bin add straight to an existing array
    // so, I split off the bulk of find_ratio_flat into find_ratio function, which accepts a C array, making life way easier while still keeping
    // find_ratio_flat but without bloating the dll
    int * fP = (int *) flat_P;
    return(find_ratio(fP, len_P));
}

void add_bin(int flat_P[], int num, int arr_len){
	// add 1 to elements in array flat_P. position of 1s is selected by num. flat_P is the pointer to an array of length arr_len
    // this function was copied over from c_functions, but I cut some corners to speed things up and save on memory
    // and yes, i do need that memory, because firefox is using 3GB right now and i'll be damned if i close it

	for(int i=0; i<arr_len; i++){
		int n2 = pow(2, arr_len-i-1);
        //printf("%d\t", flat_P[i]);
		if(num - n2 < 0){
            //printf("0");
			//bin_arr[i] = 0;
		}
		else{
            //printf("1");
			num = num - n2;
			flat_P[i] += 1;
		}
        //printf("%d\n", flat_P[i]);
	}
    //printf("\n");
}

double scoot_C_OG(int flat_P[], int num, int len_P){
    // adds num to flat_P (i.e. calls add_bin), then returns the ratio of the new array.
    // accepts flat_P as a C array. this is intended to be called a shuffling function within C
    // turns out, though, it accepts numpy arrays just fine.
    int new_fP[len_P*2];
    for(int i = 0; i < len_P*2-1; i++){
        new_fP[i] = flat_P[i];
        //printf("%d\n", new_fP[i]);
    }
    add_bin(new_fP, num, len_P*2 -1);
    return(find_ratio(new_fP, len_P));
}

double scoot_C(int flat_P[], int num, int len_P){
    // a re-writting of scoot_C where add_bin is incorporated into the pX and pY loops
    // this version doesn't operate on the input array, so a copy doesn't have to be made.
    // this should mean a little less processing and a little less memory usage, which should be useful
    // when this function is called hundreds of thousands of times for each length of coordinates.
    
    int len_fP = len_P*2 -1;
    int pX[len_P];
    for(int pos_count = 0; pos_count<len_P; pos_count++){
        int n2 = pow(2, len_fP-pos_count-1);
        if(num - n2 < 0){
            pX[pos_count] = flat_P[pos_count];
            //printf("0");
        }
        else{
            //printf("1");
            num = num - n2;
            pX[pos_count] = flat_P[pos_count] + 1;
        }
        
    }

    int pY[len_P];
    pY[0] = 250;
    for(int pos_count = 0; pos_count<len_P-1; pos_count++){
        int n2 = pow(2, len_fP-len_P-pos_count-1);
        if(num - n2 < 0){
            //printf("0");
            pY[pos_count+1] = flat_P[pos_count+len_P];
        }
        else{
            //printf("1");
            num = num - n2;
            pY[pos_count+1] = 1+flat_P[pos_count+len_P];
        }
    }
    //printf("\n");
    /* find lengths */
    int no_i = (pow(pY[len_P-1]-pX[len_P-1], 2) > 0);   
    double perimeter =  fabs((double) pX[0]*8);

    if(len_P > 1){
        for(int i = 0; i < len_P-1; i++){
            int X = pX[i+1] - pX[i];
            int Y = pY[i+1] - pY[i];
            perimeter += sqrt(64*(X*X + Y*Y));
        }
    }

    if(no_i==1){
        int Y = pY[len_P-1];
        int X = pX[len_P-1];
        perimeter += sqrt((double) (Y*Y + X*X - 2*Y*X)*32);
    }

    /*find area*/
    int xe[len_P*2 + 2];
    int ye[len_P*2 + 2];
    ye[0] = 250;
    xe[len_P*2 + 1] = 250;

    xe[0] = 0;
    ye[len_P*2 + 1] = 0;
    
    for(int i = 0; i<len_P; i++){
        xe[i+1] = pX[i];
        xe[i+len_P+1] = pY[len_P-1-i];

        ye[i+1] = pY[i];
        ye[i+len_P+1] = pX[len_P-1-i];
    }

    long area = 0;
    for(int i = 0; i <= len_P*2 - 1; i++){
        area += (xe[i+1]-xe[i])*(ye[i+1]+ye[i]);
    }
    area = area*2;

    /*find ratio*/
    double ratio = (area/perimeter);
    return(ratio);
}

int shuffle_C(int flat_P[], int len_P, long num_start, long num_end){
    // NOTE: the function finishes on num_end, not the one before
    // completely replaces the shuffle function in the python script
    // accepts a numpy array and the length of P, calls scoot_C for every
    // value between num_start and num_end, checking if the current ratio is larger
    // than the previous one. it returns the largest ratio and the number associated with the
    // largest ratio.
    // the intention with this function is that python will call it in several processes,
    // hopefully making multi-processing much easier
    // PROTIP: imagine the underscore in shuffle_C is a comma, and also it's pronounced with a transatlantic accent.
    // it makes the code a lot more entertaining
    double R_max = scoot_C(flat_P, num_start, len_P);
    int num_max = num_start;
    double R;
    for(int num = num_start+1; num<num_end+1; num++){
        R = scoot_C(flat_P, num, len_P);
        if(R > R_max){
            R_max = R;
            num_max = num;
        }
        if(num == num_end){
            printf("%d\n", num_end);
        }
    }
    //return R_max, num_max;
    return(num_max);
}

void main(){
    // function for testing other functions
    int len_P = 9;
    int flat_P[] = {123, 135, 146, 158, 168, 178, 188, 198, 207, 249, 247, 244, 240, 235, 230, 223, 215};
    
    //printf("%E", find_ratio_flat());
    printf("%f\n", find_ratio_flat(flat_P, len_P));

    /*
    int flat_P_OG[] = {123, 135, 146, 158, 168, 178, 188, 198, 207, 249, 247, 244, 240, 235, 230, 223, 215};
    add_bin(flat_P, 9, len_P);
    for(int i =0; i<len_P; i++){
        printf("%d\t%d\n", flat_P_OG[i], flat_P[i]);
    }*/

    //printf("%f\n", scoot_C(flat_P, 18, len_P));
   // printf("%f\n", scoot_C(flat_P, 18, len_P));

    long best_num = shuffle_C(flat_P, len_P, 0, pow(2, 17)-1);
    printf("%d\n", best_num);
    printf("%f\n", scoot_C(flat_P, best_num, len_P));
    printf("\n");

    printf("%f\n", scoot_C(flat_P, 100839, len_P));
}

