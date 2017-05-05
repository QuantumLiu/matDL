function layer=dense_init_cpu(prelayer,hiddensize ,flag,loss)
%% Basic layer attributes
%Input tensor sahpe
layer.trainable=1;
layer.flag=flag;
layer.input_shape=prelayer.output_shape;
if numel(prelayer.output_shape)>2
    layer.timedistributed=1;
    layer.output_shape=[layer.input_shape(1),hiddensize,layer.input_shape(end)];
else
    layer.timedistributed=0;
    layer.output_shape=[layer.input_shape(1),hiddensize];
end
dim=prelayer.output_shape(2);
batchsize=prelayer.output_shape(1);
layer.type='dense';
layer.prelayer_type=prelayer.type;
layer.hiddensize=hiddensize;
layer.batchsize=batchsize;
layer.batch=1;
layer.epoch=1;
%% Dense layer attributes
%W contains weights bias
layer.weights_dim=dim+1;
layer.W=(rand([layer.weights_dim,hiddensize],'single')-0.5)./100;
if layer.timedistributed
    layer.input=ones([layer.input_shape(1),layer.input_shape(2)+1,layer.input_shape(3)],'single');
else
    layer.input=ones([layer.input_shape(1),layer.input_shape(2)+1],'single');
end
layer.output=zeros(layer.output_shape,'single');
if ~strcmpi(layer.prelayer_type,'input')&&flag
    layer.dx=zeros(layer.input_shape,'single');
end
layer.e=layer.output;
if nargin>3&&flag
    [layer.loss_f,layer.loss_df]=loss_handle(loss);
    layer.loss=[];
end
layer.ff=@(layer,prelayer)dense_ff(layer,prelayer);
layer.bp=@(layer,next_layer)dense_bp(layer,next_layer);

layer.configs.type=layer.type;
layer.configs.input_shape=layer.input_shape;
layer.configs.output_shape=layer.output_shape;
layer.configs.hiddensize=layer.hiddensize;
layer.configs.W=size(layer.W);
end
