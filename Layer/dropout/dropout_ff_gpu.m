function layer=dropout_ff_gpu(layer,prelayer)
if layer.flag
    [layer.mask,layer.mask_index]=layer.drop(layer.mask,layer.drop_rate);
    layer.output=prelayer.output.*layer.mask;
else
    layer.output=prelayer.output*layer.drop_rate;
end
end