function layer= activation_init( prelayer,act_fun,flag,loss )
%% Basic layer attributes
layer.type='activation';
layer.trainable=0;
layer.flag=flag;
layer.prelayer_type=prelayer.type;

layer.batch=1;
layer.epoch=1;

layer.input_shape=prelayer.output_shape;
layer.output_shape=prelayer.output_shape;

% layer.input=prelayer.output;
layer.output=prelayer.output;

if ~strcmpi(layer.prelayer_type,'input')&&flag
    layer.dx=layer.output;
end
layer.e=layer.output;

if nargin>2
    [layer.loss_f,layer.loss_df]=loss_handle(loss);
    layer.loss=[];
end
layer.act=@(x)act(x,act_fun); 
layer.dact=@(x)dact(x,act_fun); 
layer.ff=@(layer,prelayer)activation_ff(layer,prelayer);
layer.bp=@(layer,next_layer)activation_bp(layer,next_layer);
layer.configs.type=layer.type;
layer.configs.input_shape=layer.input_shape;
layer.configs.output_shape=layer.output_shape;
layer.configs.act_fun=act_fun;
end

