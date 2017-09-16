#include "cumexhelp.h"
#define EXPORT_FCNS
#ifndef SHRHELP
    #include "shrhelp.h"
#endif
#include "mat_cudnn_test.h"

#define cudaErrCheck(stat) { cudaErrCheck_((stat)); }
void cudaErrCheck_(cudaError_t stat) {
    if (stat != cudaSuccess) {
        mexPrintf("CUDA Error: %s\n", cudaGetErrorString(stat));
        mexErrMsgTxt("CUDA Error");
    }
}
#define cudnnErrCheck(stat) { cudnnErrCheck_((stat)); }
void cudnnErrCheck_(cudnnStatus_t stat) {
    if (stat != CUDNN_STATUS_SUCCESS) {
        mexPrintf( "cuDNN Error: %s\n", cudnnGetErrorString(stat));
        mexErrMsgTxt("cuDNN Error");
    }
}
__global__ void initGPUData_ker(float *data, int numElements, float value) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid < numElements) {
      data[tid] = value;
   }
}
void initGPUData(float *data, int numElements, float value) {
   dim3 gridDim;
   dim3 blockDim;
   
   blockDim.x = 1024;
   gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
   
   initGPUData_ker <<< gridDim, blockDim >>> (data, numElements, value);
}
// void GET_GPU_CONST_PTR(mxArray *arrayPtr,float const *dataPtr)
// {
//     dataPtr=(float const *)(mxGPUGetDataReadOnly (mxGPUCreateFromMxArray(arrayPtr)));
// }
// void GET_GPU_PTR(mxArray *arrayPtr,float *dataPtr)
// {
//     dataPtr=(float *)(mxGPUGetData(mxGPUCreateFromMxArray(arrayPtr)));
// }

EXPORTED_FUNCTION void MAT_CUDNN_test(void* x)
{
    int(*seqLength)=10;
    int (*inputSize)=128;
    int (*miniBatch)=64;
    cudnnHandle_t cudnnHandle;
    cudnnErrCheck(cudnnCreate(&cudnnHandle));
    cudaErrCheck(cudaMalloc((void**)&x,(*seqLength) * (*inputSize) * (*miniBatch) * sizeof(float)));
    cudnnDestroy(cudnnHandle);
    cudaFree(x);
}
EXPORTED_FUNCTION void MAT_CUDNN_RNN_LSTM_FF(mxArray const *ax,mxArray const *aw,mxArray *ah,mxArray *ac,int *hiddenSize,int *miniBatch，int *inputSize，int *seqLength,void *reserveSpace)
{
   //int(*seqLength)=20;
   int numLayers=1;
   //int hiddenSize=256;
   //int (*inputSize)=128;
   //int (*miniBatch)=64;
   float dropout=0.0;
   bool bidirectional=0;
   int mode=2;
   cudnnHandle_t cudnnHandle;   
   cudnnErrCheck(cudnnCreate(&cudnnHandle));
   float const *x=(float const *)mxGPUGetDataReadOnly(mxGPUCreateFromMxArray(ax));
   void *hx = NULL;
   void *cx = NULL;
   void *y;
   void *hy = NULL;
   void *cy = NULL;
   cudaErrCheck(cudaMalloc((void**)&hx, numLayers * (*hiddenSize) * (*miniBatch) * (bidirectional ? 2 : 1) * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&cx, numLayers * (*hiddenSize) * (*miniBatch) * (bidirectional ? 2 : 1) * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&y,(*seqLength) * (*hiddenSize) * (*miniBatch) * (bidirectional ? 2 : 1) * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&hy, numLayers * (*hiddenSize) * (*miniBatch) * (bidirectional ? 2 : 1) * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&cy, numLayers * (*hiddenSize) * (*miniBatch) * (bidirectional ? 2 : 1) * sizeof(float)));
   cudnnTensorDescriptor_t *xDesc, *yDesc;
   cudnnTensorDescriptor_t hxDesc, cxDesc;
   cudnnTensorDescriptor_t hyDesc, cyDesc;
   xDesc = (cudnnTensorDescriptor_t*)malloc((*seqLength) * sizeof(cudnnTensorDescriptor_t));
   yDesc = (cudnnTensorDescriptor_t*)malloc((*seqLength) * sizeof(cudnnTensorDescriptor_t));
   int dimA[3];
   int strideA[3];
   for (int i = 0; i <(*seqLength); i++) {
      cudnnErrCheck(cudnnCreateTensorDescriptor(&xDesc[i]));
      cudnnErrCheck(cudnnCreateTensorDescriptor(&yDesc[i]));
   
      dimA[0] = (*miniBatch);
      dimA[1] = (*inputSize);
      dimA[2] = 1;
     
      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;

      cudnnErrCheck(cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
      
      dimA[0] = (*miniBatch);
      dimA[1] = bidirectional ? (*hiddenSize) * 2 : (*hiddenSize);
      dimA[2] = 1;

      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;
      
      cudnnErrCheck(cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
   }
   
   
   dimA[0] = numLayers * (bidirectional ? 2 : 1);
   dimA[1] = (*miniBatch);
   dimA[2] = (*hiddenSize);
   
   strideA[0] = dimA[2] * dimA[1];
   strideA[1] = dimA[2];
   strideA[2] = 1;
   
   cudnnErrCheck(cudnnCreateTensorDescriptor(&hxDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&cxDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&hyDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&cyDesc));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(hxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(cxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(hyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(cyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   unsigned long long seed = 1337ull; // Pick a seed.
   
   cudnnDropoutDescriptor_t dropoutDesc;
   cudnnErrCheck(cudnnCreateDropoutDescriptor(&dropoutDesc));
   
   // How much memory does dropout need for states?
   // These states are used to generate random numbers internally
   // and should not be freed until the RNN descriptor is no longer used
   size_t stateSize;
   void *states;
   cudnnErrCheck(cudnnDropoutGetStatesSize(cudnnHandle, &stateSize));
   
   cudaErrCheck(cudaMalloc(&states, stateSize));
   
   cudnnErrCheck(cudnnSetDropoutDescriptor(dropoutDesc, 
                             cudnnHandle,
                             dropout, 
                             states, 
                             stateSize, 
                             seed));
                             
   // -------------------------   
   // Set up the RNN descriptor
   // -------------------------
   cudnnRNNDescriptor_t rnnDesc;
   cudnnRNNMode_t RNNMode;
   
   cudnnErrCheck(cudnnCreateRNNDescriptor(&rnnDesc));
   
   if      (mode == 0) RNNMode = CUDNN_RNN_RELU;
   else if (mode == 1) RNNMode = CUDNN_RNN_TANH;
   else if (mode == 2) RNNMode = CUDNN_LSTM;
   else if (mode == 3) RNNMode = CUDNN_GRU;
      
   cudnnErrCheck(cudnnSetRNNDescriptor(rnnDesc,
                                       (*hiddenSize), 
                                       numLayers, 
                                       dropoutDesc,
                                       CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
                                       bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, 
                                       RNNMode, 
                                       CUDNN_DATA_FLOAT));
   void *w;   
   cudnnFilterDescriptor_t wDesc;
   cudnnErrCheck(cudnnCreateFilterDescriptor(&wDesc));   
   size_t weightsSize;
   cudnnErrCheck(cudnnGetRNNParamsSize(cudnnHandle, rnnDesc, xDesc[0], &weightsSize, CUDNN_DATA_FLOAT));
   
   int dimW[3];   
   dimW[0] =  weightsSize / sizeof(float);
   dimW[1] = 1;
   dimW[2] = 1;
      
   cudnnErrCheck(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));   
   
   cudaErrCheck(cudaMalloc((void**)&w,  weightsSize));
   void *workspace;
   size_t workSize;
   size_t reserveSize;
   cudnnErrCheck(cudnnGetRNNWorkspaceSize(cudnnHandle, rnnDesc,(*seqLength), xDesc, &workSize));
   // Only needed in training, shouldn't be touched between passes.
   cudnnErrCheck(cudnnGetRNNTrainingReserveSize(cudnnHandle, rnnDesc,(*seqLength), xDesc, &reserveSize));
    
   cudaErrCheck(cudaMalloc((void**)&workspace, workSize));
   cudaErrCheck(cudaMalloc((void**)&reserveSpace, reserveSize));
   //if (hx != NULL) initGPUData((float*)hx, numLayers * hiddenSize * (*miniBatch) * (bidirectional ? 2 : 1), 1.f);
   //if (cx != NULL) initGPUData((float*)cx, numLayers * hiddenSize * (*miniBatch) * (bidirectional ? 2 : 1), 1.f);
   int numLinearLayers = 0;
   if (RNNMode == CUDNN_RNN_RELU || RNNMode == CUDNN_RNN_TANH) {
      numLinearLayers = 2;
   }
   else if (RNNMode == CUDNN_LSTM) {
      numLinearLayers = 8;
   }
   else if (RNNMode == CUDNN_GRU) {
      numLinearLayers = 6;
   }
   for (int layer = 0; layer < numLayers * (bidirectional ? 2 : 1); layer++) {
      for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
         cudnnFilterDescriptor_t linLayerMatDesc;
         cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerMatDesc));
         float *linLayerMat;
         
         cudnnErrCheck(cudnnGetRNNLinLayerMatrixParams( cudnnHandle,
                                                        rnnDesc,  
                                                        layer,
                                                        xDesc[0], 
                                                        wDesc, 
                                                        w,
                                                        linLayerID,  
                                                        linLayerMatDesc, 
                                                        (void**)&linLayerMat));
         
         cudnnDataType_t dataType;
         cudnnTensorFormat_t format;
         int nbDims;
         int filterDimA[3];
         cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerMatDesc,
                                                  3,
                                                  &dataType,
                                                  &format,
                                                  &nbDims,
                                                  filterDimA));
                                                  
         initGPUData(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f / (float)(filterDimA[0] * filterDimA[1] * filterDimA[2]));                                                 

         cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerMatDesc));         
         
         cudnnFilterDescriptor_t linLayerBiasDesc;
         cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
         float *linLayerBias;
         
         cudnnErrCheck(cudnnGetRNNLinLayerBiasParams( cudnnHandle,
                                                        rnnDesc,  
                                                        layer,
                                                        xDesc[0], 
                                                        wDesc, 
                                                        w,
                                                        linLayerID,  
                                                        linLayerBiasDesc, 
                                                        (void**)&linLayerBias));
         
         cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerBiasDesc,
                                                  3,
                                                  &dataType,
                                                  &format,
                                                  &nbDims,
                                                  filterDimA));
                                                  
         initGPUData(linLayerBias, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f);
                                                  
         cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
      }
   }
   cudaErrCheck(cudaDeviceSynchronize());
   
   cudnnErrCheck(cudnnRNNForwardTraining(cudnnHandle, 
                                         rnnDesc, 
                                        (*seqLength),                                          
                                         xDesc, 
                                         x, 
                                         hxDesc,
                                         hx, 
                                         cxDesc, 
                                         cx, 
                                         wDesc, 
                                         w, 
                                         yDesc,  
                                         y, 
                                         hyDesc, 
                                         hy, 
                                         cyDesc, 
                                         cy, 
                                         workspace, 
                                         workSize,
                                         reserveSpace, 
                                         reserveSize));



   cudaFree(hx);
   cudaFree(cx);
   cudaFree(y);
   cudaFree(hy);
   cudaFree(cy);
   cudaFree(workspace);
   cudaFree(reserveSpace);
   cudaFree(w);
   cudnnDestroy(cudnnHandle);

}
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray*prhs[] )
{
}
