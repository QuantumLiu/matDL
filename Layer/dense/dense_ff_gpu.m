function layer=dense_ff_gpu(layer,prelayer)
if isequal(class(prelayer),'struct')
    if ~isequal(size(prelayer.output),layer.input_shape)
        error('Shape unmatched!')
    end
    if layer.timedistributed
        layer.input(1:end-1,:,:)=prelayer.output;
    else
        layer.input(1:end-1,:)=prelayer.output;
    end
else
    if layer.timedistributed
        layer.input(1:end-1,:,:)=prelayer;
    else
        layer.input(1:end-1,:)=prelayer;
    end
end
if layer.timedistributed
    layer.output(:)=layer.W*sq(layer.input);
else
    layer.output=layer.W*layer.input;
end
end
function a=sq(a)
a=reshape(a,size(a,1),[]);
end
