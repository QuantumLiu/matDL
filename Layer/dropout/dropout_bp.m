function layer=dropout_bp(layer,next_layer)
layer.dx=next_layer.dx.*layer.mask;
end