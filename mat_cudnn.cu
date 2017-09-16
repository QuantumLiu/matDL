#include"mat_cudnn.h"
void GET_GPU_CONST_PTR(mxArray const *arrayPtr,float const *dataPtr)
{
    dataPtr=(float const *)(mxGPUGetData(mxGPUCreateFromMxArray(arrayPtr)));
}
void GET_GPU_PTR(mxArray const *arrayPtr,float *dataPtr)
{
    dataPtr=(float *)(mxGPUGetData(mxGPUCreateFromMxArray(arrayPtr)));
}
void MAT_CUDNN_LSTM_FF(mxArray const *x_array,mxArray const *w_array,void **reserveSpace,int* minibatch,int* hiddenSize,int* inputSize,int* seqLength )
{   // -------------------------   
   // Create cudnn context
   // -------------------------  
   mxInitGPU();
   cudnnHandle_t cudnnHandle;   
   cudnnErrCheck(cudnnCreate(&cudnnHandle));

   cudnnTensorDescriptor_t *xDesc, *yDesc, *dxDesc, *dyDesc;
   cudnnTensorDescriptor_t hxDesc, cxDesc;
   cudnnTensorDescriptor_t hyDesc, cyDesc;
   cudnnTensorDescriptor_t dhxDesc, dcxDesc;
   cudnnTensorDescriptor_t dhyDesc, dcyDesc;
   
   xDesc = (cudnnTensorDescriptor_t*)malloc(*seqLength * sizeof(cudnnTensorDescriptor_t));
   yDesc = (cudnnTensorDescriptor_t*)malloc(*seqLength * sizeof(cudnnTensorDescriptor_t));
   dxDesc = (cudnnTensorDescriptor_t*)malloc(*seqLength * sizeof(cudnnTensorDescriptor_t));
   dyDesc = (cudnnTensorDescriptor_t*)malloc(*seqLength * sizeof(cudnnTensorDescriptor_t));
   
   int dimA[3];
   int strideA[3];
   // In this example dimA[1] is constant across the whole sequence
   // This isn't required, all that is required is that it does not increase.
   for (int i = 0; i < *seqLength; i++) {
      cudnnErrCheck(cudnnCreateTensorDescriptor(&xDesc[i]));
      cudnnErrCheck(cudnnCreateTensorDescriptor(&yDesc[i]));
      cudnnErrCheck(cudnnCreateTensorDescriptor(&dxDesc[i]));
      cudnnErrCheck(cudnnCreateTensorDescriptor(&dyDesc[i]));
   
      dimA[0] = *miniBatch;
      dimA[1] = *inputSize;
      dimA[2] = 1;
     
      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;

      cudnnErrCheck(cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
      cudnnErrCheck(cudnnSetTensorNdDescriptor(dxDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
      
      dimA[0] = *miniBatch;
      dimA[1] = *hiddenSize;
      dimA[2] = 1;

      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;
      
      cudnnErrCheck(cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
      cudnnErrCheck(cudnnSetTensorNdDescriptor(dyDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
   }
   dimA[0] = 1;
   dimA[1] = *miniBatch;
   dimA[2] = *hiddenSize;
   
   strideA[0] = dimA[2] * dimA[1];
   strideA[1] = dimA[2];
   strideA[2] = 1;
   
   cudnnErrCheck(cudnnCreateTensorDescriptor(&hxDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&cxDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&hyDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&cyDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&dhxDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&dcxDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&dhyDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&dcyDesc));
   
   cudnnErrCheck(cudnnSetTensorNdDescriptor(hxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(cxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(hyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(cyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(dhxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(dcxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(dhyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(dcyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   // -------------------------
   // Set up the dropout descriptor (needed for the RNN descriptor)
   // -------------------------
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
   float dropout=0;
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
   
   RNNMode = CUDNN_LSTM;
      
   cudnnErrCheck(cudnnSetRNNDescriptor(rnnDesc,
                                       hiddenSize, 
                                       numLayers, 
                                       dropoutDesc,
                                       CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
                                       CUDNN_UNIDIRECTIONAL, 
                                       RNNMode, 
                                       CUDNN_DATA_FLOAT));
   // -------------------------
   // Set up parameters
   // -------------------------
   // This needs to be done after the rnn descriptor is set as otherwise
   // we don't know how many parameters we have to allocate
   void *w;   
   void *dw;   

   cudnnFilterDescriptor_t wDesc, dwDesc;
   
   cudnnErrCheck(cudnnCreateFilterDescriptor(&wDesc));
   cudnnErrCheck(cudnnCreateFilterDescriptor(&dwDesc));
   
   size_t weightsSize;
   cudnnErrCheck(cudnnGetRNNParamsSize(cudnnHandle, rnnDesc, xDesc[0], &weightsSize, CUDNN_DATA_FLOAT));
   
   int dimW[3];   
   dimW[0] =  weightsSize / sizeof(float);
   dimW[1] = 1;
   dimW[2] = 1;
      
   cudnnErrCheck(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));   
   cudnnErrCheck(cudnnSetFilterNdDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));   
   
   cudaErrCheck(cudaMalloc((void**)&w,  weightsSize));
   cudaErrCheck(cudaMalloc((void**)&dw, weightsSize));
   
   
   // -------------------------
   // Set up work space and reserved memory
   // -------------------------   
   void *workspace;
   
   size_t workSize;
   size_t reserveSize;

   // Need for every pass
   cudnnErrCheck(cudnnGetRNNWorkspaceSize(cudnnHandle, rnnDesc, seqLength, xDesc, &workSize));
   // Only needed in training, shouldn't be touched between passes.
   cudnnErrCheck(cudnnGetRNNTrainingReserveSize(cudnnHandle, rnnDesc, seqLength, xDesc, &reserveSize));
    
   cudaErrCheck(cudaMalloc((void**)&workspace, workSize));
   cudaErrCheck(cudaMalloc((void**)&reserveSpace, reserveSize));
   // Weights
   int numLinearLayers = 0;
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
   // *********************************************************************************************************
   // At this point all of the setup is done. We now need to pass through the RNN.
   // *********************************************************************************************************
   
  
   
   cudaErrCheck(cudaDeviceSynchronize());
   
   cudaEvent_t start, stop;
   float timeForward, timeBackward1, timeBackward2;
   cudaErrCheck(cudaEventCreate(&start));
   cudaErrCheck(cudaEventCreate(&stop));
   
   cudaErrCheck(cudaEventRecord(start));   

   // If we're not training we use this instead
   // cudnnErrCheck(cudnnRNNForwardInference(cudnnHandle, 
                                         // rnnDesc, 
                                         // xDesc, 
                                         // x, 
                                         // hxDesc,
                                         // hx, 
                                         // cxDesc, 
                                         // cx, 
                                         // wDesc, 
                                         // w, 
                                         // yDesc,  
                                         // y, 
                                         // hyDesc, 
                                         // hy, 
                                         // cyDesc, 
                                         // cy, 
                                         // workspace, 
                                         // workSize));

   cudnnErrCheck(cudnnRNNForwardTraining(cudnnHandle, 
                                         rnnDesc, 
                                         seqLength,                                          
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
   

}