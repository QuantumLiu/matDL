function test_lstm(nb_batch,hiddensizes,input_dim,timestep,batch_size,nb_epoch)
optimizer.learningrate=0.01;
optimizer.momentum=0;
optimizer.opt='sgd';
x=rand(input_dim,timestep,batch_size*nb_batch,'single','gpuArray');
y=(zeros(hiddensizes(end),timestep,batch_size*nb_batch,'single','gpuArray'));
y(1,:,:)=1;
input_shape=[input_dim,timestep,batch_size];
for l=1:length(hiddensizes)
configs{l}.type='lstm';configs{l}.hiddensize=hiddensizes(l);configs{l}.return_sequence=1;
end
configs{l+1}.type='dropout';configs{l+1}.drop_rate=0.5;
configs{l+2}.type='activation';configs{l+2}.act_fun='softmax';configs{l+2}.loss='categorical_cross_entropy';
model=model_init(input_shape,configs,1,optimizer);
profile on;
model=model.train(x,y,nb_epoch,2,0);
loss=model.evaluate(x,y);
disp(loss);
profile report;
end