function layer=layer_optimize(layer,pars,batch,epoch)
if nargin <2
    pars.opt='sgd';
end
switch pars.opt
    case 'sgd'
        if pars.momentum >0
            if batch==1
                layer.vW=pars.learningrate*layer.dW;
            else
                layer.vW=pars.momentum*layer.vW+pars.learningrate*layer.dW;
            end
            layer.W=layer.W-layer.vW;
        else
            layer.W=layer.W-pars.learningrate*layer.dW;
        end
end
layer.batch=batch;
layer.epoch=epoch;
end