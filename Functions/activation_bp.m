function layer=activation_bp(layer,next_layer)
if isequal(class(next_layer),'struct')
    if ~isequal(size(next_layer.dx),layer.output_shape)
        error('Shape unmatched!')
    end
    layer.e=next_layer.dx;
end
layer.dx=layer.e.*layer.dact(layer.output);
end
