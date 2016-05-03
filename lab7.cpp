#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <CL/opencl.h>
#define MAXSOURCE 2048
#define MAX_DEVICE_NAME_SIZE 100
// OpenCL kernel. Each work item takes care of one element of c
/*const char *kernelSource =                                       "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void vecAdd(  __global double *a,                       \n" \
"                       __global double *b,                       \n" \
"                       __global double *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"}                                                               \n" \
                                                                "\n" ;



 */
void print_opencl_error(FILE* fh, cl_int err)
{
#define PRINT_ERR(code) case code : fprintf(fh, #code); break
  switch(err) {
    PRINT_ERR(CL_INVALID_PROGRAM);
    PRINT_ERR(CL_INVALID_VALUE);
    PRINT_ERR(CL_INVALID_DEVICE);
    PRINT_ERR(CL_INVALID_BINARY);
    PRINT_ERR(CL_INVALID_BUILD_OPTIONS);
    PRINT_ERR(CL_INVALID_OPERATION);
    PRINT_ERR(CL_COMPILER_NOT_AVAILABLE);
    PRINT_ERR(CL_BUILD_PROGRAM_FAILURE);
    PRINT_ERR(CL_OUT_OF_RESOURCES);
    PRINT_ERR(CL_OUT_OF_HOST_MEMORY);
  default:
    fprintf(fh, "unknown code");
    break;
  };
  return;
}
                                                               
 
int main( int argc, char* argv[] )
{
    // Length of vectors
    unsigned int n = 10;
 
    // Host input vectors
    float *h_a;
    float *h_b;
    // Host output vector
    float *h_c;
    
    char *kernelSource = (char *) malloc(MAXSOURCE*sizeof(char));
    // Open kernel file
    FILE * file = fopen("lab7_kernel.cl","r");
    if(file == NULL)
    {
        printf("Error: open lab7_kernel.cl\n");
        exit(0);
    }
    // Read kernel code
    size_t source_size = fread(kernelSource, 1, MAXSOURCE, file);
    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;
 	char deviceName[MAX_DEVICE_NAME_SIZE];
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
 	cl_platform_id* platforms;   
 	    cl_uint platformCount;
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(float);
 
    // Allocate memory for each vector on host
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
 
    // Initialize vectors on host
    int i;
    for( i = 0; i < n; i++ )
    {
        h_a[i] = sinf(i)*sinf(i);
        h_b[i] = cosf(i)*cosf(i);
    }
 
    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 64;
 
    // Number of total work items - localSize must be devisor
    globalSize = ceil(n/(float)localSize)*localSize;
 
    // Bind to platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount,platforms, NULL);
 	// find first device
 	err = 1;
 	for (i = 0; i < platformCount && err !=0; i++) {
 		 err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
 	}

   clGetDeviceInfo(device_id, CL_DEVICE_NAME,MAX_DEVICE_NAME_SIZE, deviceName, NULL);

 	if(err!=CL_SUCCESS)
 	{
 		printf("Error device not found\n");
 		exit(0);
 	}
 	printf("Device: %s\n",deviceName);
    // Get ID for the device
   
 
    // Create a context  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    printf("criando contexto %d\n",err);
    // Create a command queue 
    queue = clCreateCommandQueue(context, device_id, 0, &err);
 
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource,(const size_t *) &source_size, &err);
   printf("erro criando proma %d\n",err);
    // Build the program executable 

    err = clBuildProgram(program, 0,NULL, NULL, NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        cl_int logStatus;
        char* buildLog = NULL;
        size_t buildLogSize = 0;
        logStatus = clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, &buildLogSize);
        buildLog = (char*)malloc(buildLogSize);
        memset(buildLog, 0, buildLogSize);
        logStatus = clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
        printf("%s", buildLog);
        free(buildLog);
        return err;
    } else if (err) {
        printf("Unknown build program error !!!! (%d)\n", err);
        return err;
    }
    printf("erro buildprogram %d",err);
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "vecAdd", &err);
   printf("erro criando kernel %d\n",err);
    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
 
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   bytes, h_b, 0, NULL, NULL);
    printf("erro copiando buffers %d\n",err);
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);
    printf("erro set args %d\n",err);
    // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	  fprintf(stderr, "Error: clEnqueueNDRangeKernel(..) returned error code %d (", err);
       print_opencl_error(stderr, err);
       fprintf(stderr, ")\n");

    printf("erro chama kernel: %d , globalsize = %d localsize = %d \n",err,globalSize,localSize);
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                                bytes, h_c, 0, NULL, NULL );
 
    //Sum up vector c and print result divided by n, this should equal 1 within error
    float sum = 0;
    for(i=0; i<n; i++)
	{
	printf("h_c= %d",h_c[i]);
        sum += h_c[i];
	}
    printf("final result: %f %d \n", sum/n,n);
 
    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}
