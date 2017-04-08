function layer=dropout_init_gpu(prelayer,drop_rate ,flag,loss)
%% Basic layer attributes
layer.trainable=0;
layer.flag=flag;
layer.input_shape=prelayer.output_shape;
batchsize=prelayer.output_shape(end);
layer.type='dropout';
layer.prelayer_type=prelayer.type;
layer.output_shape=layer.input_shape;
layer.batchsize=batchsize;
layer.batch=1;
layer.epoch=1;
%% Dropout layer attributes
layer.drop_rate=drop_rate;
if layer.flag
    layer.mask=ones(layer.output_shape,'single','gpuArray');
end
layer.output=zeros(layer.output_shape,'single','gpuArray');
if ~strcmpi(layer.prelayer_type,'input')&&flag
    layer.dx=zeros(layer.input_shape,'single','gpuArray');
end
if nargin>3&&flag
    [layer.loss_f,layer.loss_df]=loss_handle(loss);
    layer.loss=[];
end
layer.drop=@(mask,drop_rate)drop(mask,drop_rate);
layer.ff=@(layer,prelayer)dropout_ff_gpu(layer,prelayer);
layer.bp=@(layer,next_layer)dropout_bp_gpu(layer,next_layer);

layer.configs.type=layer.type;
layer.configs.input_shape=layer.input_shape;
layer.configs.output_shape=layer.output_shape;
layer.configs.drop_rate=layer.drop_rate;
end
