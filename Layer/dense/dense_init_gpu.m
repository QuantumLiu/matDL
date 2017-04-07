function layer=dense_init_gpu(prelayer,hiddensize ,flag,loss)
%% Basic layer attributes
%Input tensor sahpe
layer.trainable=1;
layer.flag=flag;
if numel(prelayer.output_shape)>2
    layer.timedistributed=1;
else
    layer.timedistributed=0;
end
layer.input_shape=prelayer.output_shape;
dim=prelayer.output_shape(1);
batchsize=prelayer.output_shape(end);
layer.type='dense';
layer.prelayer_type=prelayer.type;
layer.output_shape=[hiddensize,layer.input_shape(2:end)];
layer.hiddensize=hiddensize;
layer.batchsize=batchsize;
layer.batch=1;
layer.epoch=1;
%% Dense layer attributes
%W contains weights bias
layer.weights_dim=dim+1;
layer.W=(rand([hiddensize,layer.weights_dim],'single','gpuArray')-0.5)./100;
layer.input=ones([layer.input_shape(1)+1,layer.input_shape(2:end)],'single','gpuArray');
layer.output=zeros(layer.output_shape,'single','gpuArray');
if ~strcmpi(layer.prelayer_type,'input')&&flag
    layer.dx=zeros(layer.input_shape,'single','gpuArray');
end
layer.e=layer.output;
if nargin>3&&flag
    [layer.loss_f,layer.loss_df]=loss_handle(loss);
    layer.loss=[];
end
layer.ff=@(layer,prelayer)dense_ff_gpu(layer,prelayer);
layer.bp=@(layer,next_layer)dense_bp_gpu(layer,next_layer);

layer.configs.type=layer.type;
layer.configs.input_shape=layer.input_shape;
layer.configs.output_shape=layer.output_shape;
layer.configs.hiddensize=layer.hiddensize;
layer.configs.W=size(layer.W);
end
