function layer=dense_ff(layer,prelayer)
if isequal(class(prelayer),'struct')
    if ~isequal(size(prelayer.output),layer.input_shape)
        error('Shape unmatched!')
    end
    if layer.timedistributed
        layer.input(:,1:end-1,:)=prelayer.output;
    else
        layer.input(:,1:end-1)=prelayer.output;
    end
else
    if layer.timedistributed
        layer.input(:,1:end-1,:)=prelayer;
    else
        layer.input(:,1:end-1)=prelayer;
    end
end
if layer.timedistributed
    layer.output(:)=mult_3d(layer.input,layer.W);
else
    layer.output=layer.input*layer.W;
end
end
function c=mult_3d(a,b)
shape=size(a);
timestep=shape(end);
dim=shape(2);
batchsize=shape(1);
c=permute(reshape((reshape(permute(a,[2,1,3]),dim,[])'*b)',[dim,batchsize,timestep]),[2,1,3]);
end