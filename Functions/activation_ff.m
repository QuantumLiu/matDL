function layer=activation_ff(layer,prelayer)
if ~isequal(size(prelayer.output),layer.input_shape)
    error('Shape unmatched!')
end
layer.output=layer.act(prelayer.output);
end