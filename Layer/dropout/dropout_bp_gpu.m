function layer=dropout_bp_gpu(layer,next_layer)
layer.dx=next_layer.dx.*layer.mask;
end