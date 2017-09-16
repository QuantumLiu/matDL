#include "shrhelp.h"
#ifndef CUMEXHELP
#include <mex.h>
#endif
#ifdef __cplusplus
    #include <stddef.h>
    extern "C"
    {
#endif
    EXPORTED_FUNCTION void MAT_CUDNN_test(void* x);
    EXPORTED_FUNCTION void MAT_CUDNN_RNN_LSTM_FF(mxArray const *ax,void *reserveSpace);
#ifdef __cplusplus
    }
#endif