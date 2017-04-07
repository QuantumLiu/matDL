function layer =dense_bp_gpu(layer,next_layer)
if isequal(class(next_layer),'struct')
    if ~isequal(size(next_layer.dx),layer.output_shape)
        error('Shape unmatched!')
    end
    layer.e=next_layer.dx;
end
layer.dW=sq(layer.e)*sq(layer.input)';
if ~isequal(layer.prelayer_type,'input')
    if layer.timedistributed
        layer.dx(:)=layer.W(:,1:end-1)'*sq(layer.e);
    else
        layer.dx=layer.W(:,1:end-1)'*layer.e;
    end
end
end
function a=sq(a)
a=reshape(a,size(a,1),[]);
end
