function layer=lstm_init_cpu(prelayer,hiddensize,return_sequence,flag,loss)
%% Basic layer attributes
%Input tensor sahpe
layer.input_shape=prelayer.output_shape;
layer.trainable=1;
layer.flag=flag;

dim=prelayer.output_shape(2);
timestep=prelayer.output_shape(3);
batchsize=prelayer.output_shape(1);
if nargin<3
    return_sequence=1;
end
layer.return_sequence=return_sequence;
if return_sequence
%Output tensor shape
    layer.output_shape=[batchsize,hiddensize,timestep];
else
    layer.output_shape=[batchsize,hiddensize];        
end
%The type of the layer
layer.type='lstm';
%conected layer type
layer.prelayer_type=prelayer.type;
%The hiddensize of the layer
layer.hiddensize=hiddensize;

layer.batch=1;
layer.epoch=1;
%% lstm layer attributes
%Timestep 
layer.timestep=timestep;
layer.batchsize=batchsize;
%n is the number of unrolled timesteps in one batch
layer.n=batchsize*timestep;
%Put x(t) and h(t) in one array 
layer.xh=ones([batchsize,dim+1+hiddensize,timestep+1],'single');
%W is the weights of all four gates and bias
layer.weights_dim=dim+1+hiddensize;
layer.W=(rand([layer.weights_dim,4*hiddensize],'single')-0.5)./100;
%Compute the value of x_t*wx_t for all ts in one time
layer.maX=zeros([batchsize,4*hiddensize,timestep],'single');
%value before activited
layer.ma=layer.maX;
%value activited
layer.mb=layer.maX;
%sc:state of cell
layer.sc=zeros([batchsize,hiddensize,timestep],'single');
layer.bc=layer.sc;
%The output tensor and error
layer.output=zeros(layer.output_shape,'single');
layer.e=layer.sc;
if layer.flag
%diffs
layer.dW=zeros(size(layer.W),'single');
layer.dma=zeros([batchsize,4*hiddensize,timestep+1],'single');
layer.dmb=layer.dma;
layer.dsc=layer.sc;
layer.dh=layer.dsc;
end
if ~strcmpi(layer.prelayer_type,'input')&&layer.flag
    layer.dx=zeros(layer.input_shape,'single');
end
if nargin>4
    [layer.loss_f,layer.loss_df]=loss_handle(loss);
    layer.loss=[];
end
%% methods
layer.act_f =@(x)act(x,'sigmoid'); % active function for gate
layer.act_tc =@(x)act(x, 'tanh'); % active function for tc
layer.act_h = @(x)act(x, 'tanh');

layer.dact_f= @(x)dact(x,'sigmoid');
layer.dact_tc =@(x)dact(x, 'tanh'); % active function for tc
layer.dact_h = @(x)dact(x, 'tanh');
layer.ff=@(layer,prelayer)lstm_ff(layer,prelayer);
layer.bp=@(layer,next_layer)lstm_bp(layer,next_layer);

layer.configs.type=layer.type;
layer.configs.input_shape=layer.input_shape;
layer.configs.output_shape=layer.output_shape;
layer.configs.hiddensize=layer.hiddensize;
layer.configs.W=size(layer.W);
end