#include <stdio.h>

char yo() {
	char * out = "Yo wassup my homie";
	printf(out);
}

int main()
{
	yo();
	return 0;
}

void butts(){
	printf("butts");
}

int sum(const void * nparray, int len_array){
	//sums values in a numpy array. copied from https://stackoverflow.com/questions/5862915/passing-numpy-arrays-to-a-c-function-for-input-and-output
	//i don't really understand what's going on with the pointers.
	//requires array to be passed as ctypes.c_void_p(arr.ctypes.data), which takes processing during runtime.
	//either a different approach should be used, or the shuffling should be done in the c function.
	//shuffling in C should be pretty easy as for loops are back on the table!
	const int * array = (int *) nparray;
	int cum = 0;
	int i = 0;
	for (i=0; i<len_array; i++) {
		cum += array[i];
	};
	return cum;
}

int power(int num){
	//return 2**num
	int out = 2;
	//int i = 0;
	if(num==0){
		return 1;
	}
	else{
		for (int i=1; i<num; i++){
			out *= 2;
		}
		return out;
	}
}


int array(){
	// this prints the address of the array, not the array itself
	int arr[5];
	for (int i=0; i<5; i++){
		arr[i] = i*2;
	}
	//printf(arr);	// doesn't like that it's been passed a pointer
	//return arr;	//error message: returning 'int *' from a function with return type 'int' makes integer from pointer without a cast
	for (int i=0; i<5; i++){
		printf("%d \n", arr[i]);
	}
	return arr[-2];		//this doesn't work, because arr points to the first element of the array, and the index is the first
						//element plus the index times the space between each element i.e. for 8-bit integers the space would be
						//8 bits
}

int bin0(int num, int arr_len){
	//converts an integer number to binary, as an array of the same length as flat_P
	//Actually returns the wrong thing for some reason: 
	const int len = arr_len;
	//int arr[len] = {0};
	
	int arr[arr_len];	//ignore this error, it works fine. this is how you create a variable length array in C
	for(int i=0; i<arr_len; i++){																	//seriously
		arr[i] = 0;
	}
	
	//find number of places in binary number. in python this would be int(log2(num))+1. but this isn't python,
	//and I'm struggling to come to terms with that
	int n0 = 0;
	while (power(n0)<num){
		n0++;
	}

	//convert num into a binary number spread accross an array
	for(int i=n0; n0>=0; i++){
		if(num - power(n0)>=0){
			num = num - power(n0);
			arr[i] = 1;
		}
		n0--;
	}


	arr[arr_len-n0] = 1;
	for(int i=0; i<arr_len; i++){
		printf("%d", arr[i]);
	}
	printf("\n");
	return num;
}

int * bin(int num, int arr_len){
	//converts an integer number to binary, as an array of the same length as flat_P
	//returns the array
	
	static int bin_arr[250];	//creates a large non-variable array, but only the first arr_len indexes will be used
								//static variable-length arrays are possible, but the memory block has to be declared i think,
								// so this is probably the same thing, only with less learning involved
	for(int i=0; i<arr_len; i++){
		int n2 = power(arr_len-i-1);
		//printf("i = %d,\t 2**n = %d\n", i, n2);
		if(num - n2 < 0){
			bin_arr[i] = 0;
		}
		else{
			num = num - n2;
			bin_arr[i] = 1;
		}
	}
	/*
	for(int i=0; i<arr_len; i++){
		printf("%d", bin_arr[i]);
	}
	printf("\n");
	*/
	return bin_arr;
}

void print_bin(){
	//runs bin, takes the array, and prints the array
	int *arr; //initialise the array as a pointer
	int arr_len = 11;
	arr = bin(5, arr_len);
	/*for(int i=0; i<arr_len; i++){
		printf("%d, ", arr[i]);
	}*/
	for(int i=0; i<arr_len; i++){
		printf("%d", arr[i]);
	}
}

void change_vals(int * num, double * p_rat){
	// this function is to test if I can change the input variables, so that I can
	// return multiple variables without having to get my head around returning a struct in python
	// this works, but the variables are ctypes, so need to be converted back into native python.
	double rat = *p_rat;
	*num = *num*2 + 1;
	*p_rat = rat*rat;
}