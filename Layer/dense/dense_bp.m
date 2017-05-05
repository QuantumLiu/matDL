function layer =dense_bp(layer,next_layer)
if isequal(class(next_layer),'struct')
    if ~isequal(size(next_layer.dx),layer.output_shape)
        error('Shape unmatched!')
    end
    layer.e=next_layer.dx;
end
if layer.timedistributed
    layer.dW=reshape(permute(layer.input,[2,1,3]),layer.weights_dim,[])*reshape(permute(layer.e,[2,1,3]),4*hiddensize,[])';
    if ~isequal(layer.prelayer_type,'input')
        layer.dx(:)=mult_3d(layer.e,layer.W(1:end-1,:)');
    end
else
    layer.dW=layer.input'*layer.e;
    if ~isequal(layer.prelayer_type,'input')
        layer.dx=layer.e*layer.W(1:end-1,:)';
    end
end
end
function a=sq(a)
a=reshape(a,size(a,1),[]);
end
function c=mult_3d(a,b)
shape=size(a);
timestep=shape(end);
dim=shape(2);
batchsize=shape(1);
c=permute(reshape((reshape(permute(a,[2,1,3]),dim,[])'*b)',[dim,batchsize,timestep]),[2,1,3]);
end