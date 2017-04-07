function test_mlp(nb_batch,hiddensize,input_dim,batch_size,nb_epoch)
dense_ff=@(layer,prelayer)dense_ff_gpu(layer,prelayer);
dense_bp=@(layer,next_layer)dense_bp_gpu(layer,next_layer);
act_ff=@(layer,prelayer)activation_ff(layer,prelayer);
act_bp=@(layer,next_layer)activation_bp(layer,next_layer);
pars.learningrate=0.01;
pars.momentum=0;
pars.opt='sgd';
optimize=@(layer)layer_optimize(layer,pars);
x=ones(input_dim,batch_size*nb_batch,'single','gpuArray');
y=(zeros(hiddensize(end),batch_size*nb_batch,'single','gpuArray'));
y(1,:)=1;
inputlayer=tensor_init_gpu([input_dim,batch_size],'input');
denselayer1=dense_init_gpu(inputlayer,hiddensize(1));
denselayer2=dense_init_gpu(denselayer1,hiddensize(2));
denselayer3=dense_init_gpu(denselayer2,hiddensize(3));
outputlayer=activation_init(denselayer3,'softmax','categorical_cross_entropy');
profile on;
for epoch=1:nb_epoch
    tic;
    for i=1:nb_batch
        denselayer1=dense_ff(denselayer1,x(:,(i-1)*batch_size+1:i*batch_size));
        denselayer2=dense_ff(denselayer2,denselayer1);
        denselayer3=dense_ff(denselayer3,denselayer2);
        outputlayer=act_bp(eval_loss(act_ff(outputlayer,denselayer3),y(:,(i-1)*batch_size+1:i*batch_size)),[]);
        denselayer3=optimize(dense_bp(denselayer3,outputlayer));
        denselayer2=optimize(dense_bp(denselayer2,denselayer3));
        denselayer1=optimize(dense_bp(denselayer1,denselayer2));
    end
    toc;
end
profile report;
plot(outputlayer.loss);
end