#define EXPORT_FCNS
#ifndef SHRHELP
    #include "shrhelp.h"
#endif

#include "cumexhelp.h"

#ifndef MATCUDNN
    #define MATCUDNN
#endif

#ifdef __cplusplus
    #include <stddef.h>
    extern "C"
    {
#endif

	EXPORTED_FUNCTION void MAT_CUDNN_test(void* x);

#ifdef __cplusplus
    }
#endif
