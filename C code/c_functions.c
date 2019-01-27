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
	int i = 0;
	for (i=1; i<num; i++){
		out *= 2;
	}
	return out;
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
	return arr[-2];
}

