function layer=lstm_ff_gpu(layer,prelayer)
timestep=layer.timestep;
hiddensize=layer.hiddensize;
dim=layer.input_shape(1);
r_x=1:dim+1;%range of x and bias
r_h=dim+1+(1:hiddensize);%range of h
r_ifo=1:3*hiddensize;%range of forget,input and output gates
r_f=1:hiddensize;%range of forget gate
r_i=hiddensize+1:2*hiddensize;%~input gate
r_o=2*hiddensize+1:3*hiddensize;%~output gate
r_tc=3*hiddensize+1:4*hiddensize;%range of tilde c gate
%the xh is a 2d tensor contain x,bias,and h,((r_x)-1,1:end-1,:) is the area of x
%assign value from input tensor
if isequal(class(prelayer),'struct')
    if ~isequal(size(prelayer.output),layer.input_shape)
        error('Shape unmatched!')
    end
    layer.xh(r_x(1:end-1),1:end-1,:)=prelayer.output;
else
    layer.xh(r_x(1:end-1),1:end-1,:)=prelayer;
end
%compute all x(t)*W_x+bias in one time at first
layer.maX(:)=layer.W(:,r_x)*sq(layer.xh(r_x,1:end-1,:));

%% Feed forward
%t=1
layer.ma( :,1,:)=layer.maX( :,1,:);
layer.mb( r_ifo,1,:)=layer.act_f(layer.ma(r_ifo,1,:));
layer.mb( r_tc,1,:)=layer.act_tc(layer.ma(r_tc,1,:));
layer.sc( :,1,:)=layer.mb(r_i,1,:).*layer.mb(r_tc,1,:);
layer.bc( :,1,:)=layer.act_h(layer.sc( :,1,:));
layer.xh( r_h,2,:)=layer.bc(:,1,:).*layer.mb(r_o,1,:);
%t>1
for t=2:timestep
    % a(t) = W_x * x(t) + W_h * h(t-1)
    layer.ma( :,t,:)=sq(layer.maX( :,t,:))+layer.W(:,r_h)*sq(layer.xh( r_h,t,:));
    %b(t)=act(a(t))
    %The active functions of i,f,o gates are sigmoid,compute in one time
    layer.mb( r_ifo,t,:)=layer.act_f(layer.ma( r_ifo,t,:));
    %The active function of tc gate is tanh
    layer.mb( r_tc,t,:)=layer.act_tc(layer.ma( r_tc,t,:));
    % c(t) = f(t) * c(t-1) + i(t) * tc(t)
    layer.sc( :,t,:)=layer.sc( :,t-1,:).*layer.mb( r_f,t,:)+layer.mb( r_i,t,:).*layer.mb( r_tc,t,:);
    %tanh(c(t))
    layer.bc( :,t,:)=layer.act_h(layer.sc( :,t,:));
    % h(t) = o(t) * tanh(c(t))
    layer.xh( r_h,t+1,:)=layer.bc( :,t,:).*layer.mb( r_o,t,:);
    if layer.return_sequence
        layer.output=layer.xh(r_h,2:end,:);
    else
        layer.output=sq(layer.xh(r_h,1,:));
    end
end
end
function a=sq(a)
%% squeeze a 3d tensor (dim,timestep,batchsize) into £¨dim,timestep*batchsize),built_in function 'squeeze' has some redundant options
a=reshape(a,size(a,1),[]);
end