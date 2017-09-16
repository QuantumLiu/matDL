[e,n]=loadlibrary('C:\projects\mexcuda\matcudnn\mat_cudnn_test','mat_cudnn_test.h','includepath','C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include','addheader','cudnn.h','addheader','cuda_runtime.h');
ax=ones([128,256,20],'single','gpuArray');
reserve=libpointer('voidPtr');
tic;
for i=1:100
calllib('mat_cudnn_test','MAT_CUDNN_RNN_LSTM_FF',ax,reserve);
end
toc;
