function test_mlp(nb_batch,hiddensize,input_dim,batch_size,nb_epoch)
optimizer.learningrate=0.01;
optimizer.momentum=0;
optimizer.opt='sgd';
x=rand(input_dim,batch_size*nb_batch,'single','gpuArray');
y=(zeros(hiddensize(end),batch_size*nb_batch,'single','gpuArray'));
y(1,:,:)=1;
input_shape=[input_dim,batch_size];
l=1;
for i=1:length(hiddensize)
configs{l}.type='dense';configs{l}.hiddensize=hiddensize(i);
l=l+1;
configs{l}.type='activation';configs{l}.act_fun='Relu';
l=l+1;
configs{l}.type='dropout';configs{l}.drop_rate=0.5;
end
configs{l+1}.type='activation';configs{l+1}.act_fun='softmax';configs{l+1}.loss='categorical_cross_entropy';
model=model_init(input_shape,configs,1,optimizer);
model=model.train(model,x,y,nb_epoch,3,0);%not save
loss=model.evaluate(model,x,y);
disp(loss);
end