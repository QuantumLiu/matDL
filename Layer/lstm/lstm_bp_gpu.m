function layer=lstm_bp_gpu(layer,next_layer)
if isequal(class(next_layer),'struct')
    if ~isequal(size(next_layer.dx),layer.output_shape)
        error('Shape unmatched!')
    end
    if layer.return_sequence
        layer.e=next_layer.dx;
    else
        layer.e(:,end,:)=next_layer.dx;
    end
end
timestep=layer.timestep;
hiddensize=layer.hiddensize;
batchsize=layer.batchsize;
dim=layer.input_shape(1);
r_x=1:dim+1;
r_h=dim+1+(1:hiddensize);
r_ifo=1:3*hiddensize;
r_f=1:hiddensize;
r_i=hiddensize+1:2*hiddensize;
r_o=2*hiddensize+1:3*hiddensize;
r_tc=3*hiddensize+1:4*hiddensize;
%% Backpropagation through time
for t=timestep:-1:2
    % d_h(t) = e(t) + d_a(t+1)*W
    layer.dh(:,t,:)=sq(layer.e(:,t,:))+layer.W(:,r_h)'*sq(layer.dma(:,t+1,:));
    % d_c(t) = d_h(t) .* o(t) * tanh'(c(t))
    layer.dsc(:,t,:)=layer.dh(:,t,:).*layer.mb(r_o,t,:).*layer.dact_h(layer.sc(:,t,:));
    %db_o(t) = d_h(t) * bc(t)
    layer.dmb(r_o,t,:)=layer.dh(:,t,:).*layer.bc(:,t,:);
    % db_i(t) = d_c(t) .* tc(t)
    layer.dmb(r_i,t,:)=layer.dsc(:,t,:).*layer.mb(r_tc,t,:);
    % db_tc(t) = db_c(t) .* i(t)
    layer.dmb(r_tc,t,:)=layer.dsc(:,t,:).*layer.mb(r_i,t,:);
    % db_f(t) = db_c(t) .* c(t-1)
    layer.dmb(r_f,t,:)=layer.dsc(:,t,:).*layer.sc(:,t-1,:);
    %da=act'(b).*db
    layer.dma(r_ifo,t,:)=layer.dact_f(layer.mb(r_ifo,t,:)).*layer.dmb(r_ifo,t,:);
    layer.dma(r_tc,t,:)=layer.dact_tc(layer.mb(r_tc,t,:)).*layer.dmb(r_tc,t,:);
end
t=1;
layer.dh(:,t,:)=sq(layer.e(:,t,:))+layer.W(:,r_h)'*sq(layer.dma(:,t+1,:));
layer.dsc(:,t,:)=layer.dh(:,t,:).*layer.mb(r_o,t,:).*layer.dact_h(layer.sc(:,t,:));
layer.dmb(r_o,t,:)=layer.dh(:,t,:).*layer.bc(:,t,:);
layer.dmb(r_i,t,:)=layer.dsc(:,t,:).*layer.mb(r_tc,t,:);
layer.dmb(r_tc,t,:)=layer.dsc(:,t,:).*layer.mb(r_i,t,:);
layer.dma(r_ifo,t,:)=layer.dact_f(layer.mb(r_ifo,t,:)).*layer.dmb(r_ifo,t,:);
layer.dma(r_tc,t,:)=layer.dact_tc(layer.mb(r_tc,t,:)).*layer.dmb(r_tc,t,:);

layer.dma(r_f,2:end,:)=layer.dma(r_f,2:end,:)./(timestep-1);
layer.dma(hiddensize+1:end,:)=layer.dma(hiddensize+1:end,:)./timestep;
layer.dW=layer.dma(:,:)*layer.xh(:,:)'./batchsize;
if ~isequal(layer.prelayer_type,'input')
    layer.dx(:)=layer.W(:,r_x(1:end-1))'*sq(layer.dma(:,1:end-1,:))./batchsize;
end
end
function a=sq(a)
a=reshape(a,size(a,1),[]);
end