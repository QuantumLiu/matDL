function test_lstm(nb_batch,hiddensize,input_dim,timestep,batch_size,nb_epoch)
optimizer.learningrate=0.01;
optimizer.momentum=0;
optimizer.opt='sgd';
x=rand(input_dim,timestep,batch_size*nb_batch,'single','gpuArray');
y=(zeros(hiddensize(end),timestep,batch_size*nb_batch,'single','gpuArray'));
y(1,:,:)=1;
input_shape=[input_dim,timestep,batch_size];
% inputlayer=tensor_init_gpu([input_dim,timestep,batch_size],'input');
% lstmlayer1=lstm_init_gpu(inputlayer,hiddensize(1),1);
% lstmlayer2=lstm_init_gpu(lstmlayer1,hiddensize(2),1);
% lstmlayer3=lstm_init_gpu(lstmlayer2,hiddensize(3),1);
% outputlayer=activation_init(lstmlayer3,'softmax','categorical_cross_entropy');
for l=1:length(hiddensize)
configs{l}.type='lstm';configs{l}.hiddensize=hiddensize(l);configs{l}.return_sequence=1;
end
configs{l+1}.type='activation';configs{l+1}.act_fun='softmax';configs{l+1}.loss='categorical_cross_entropy';
model=model_init(input_shape,configs,optimizer);
profile on;
model=model_train(model,x,y,nb_epoch,2);
profile report;
end