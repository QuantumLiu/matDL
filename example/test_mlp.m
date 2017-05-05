function test_mlp(nb_batch,hiddensize,input_dim,batch_size,nb_epoch)
input_shape=[batch_size,input_dim];
l=1;
for i=1:length(hiddensize)
configs{l}.type='dense';configs{l}.hiddensize=hiddensize(i);
l=l+1;
configs{l}.type='activation';configs{l}.act_fun='Relu';
l=l+1;
configs{l}.type='dropout';configs{l}.drop_rate=0.5;
end
configs{l+1}.type='activation';configs{l+1}.act_fun='softmax';configs{l+1}.loss='categorical_cross_entropy';
optimizer.learningrate=0.01;
optimizer.momentum=0.5;
optimizer.opt='sgd';
model=model_init(input_shape,configs,1,optimizer);
x=rand(batch_size*nb_batch,input_dim);
y=(zeros(batch_size*nb_batch,hiddensize(end)));
y(:,1,:)=1;
model=model.train(model,x,y,nb_epoch,3,0);%not save
loss=model.evaluate(model,x,y);
disp(loss);
y_pred=model.predict(model,x);
end