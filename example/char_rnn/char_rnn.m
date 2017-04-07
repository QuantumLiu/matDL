function char_rnn(data_filename,hiddensize,timestep,batch_size,nb_epoch)
load(data_filename,'x');
load(data_filename,'y');
x=reshape(x(:,1:(timestep*batch_size)*floor(length(x)/(timestep*batch_size))),size(x,1),timestep,[]);
y=reshape(y(:,1:(timestep*batch_size)*floor(length(y)/(timestep*batch_size))),size(y,1),timestep,[]);
y=squeeze(y(:,end,:));
input_shape=[size(x,1),timestep,batch_size];
for l=1:length(hiddensize)-2
    configs{l}.type='lstm';configs{l}.hiddensize=hiddensize(l);configs{l}.return_sequence=1;
end
l=l+1;
configs{l}.type='lstm';configs{l}.hiddensize=hiddensize(l);configs{l}.return_sequence=0;
configs{l+1}.type='dense';configs{l+1}.hiddensize=hiddensize(l+1);
configs{l+2}.type='activation';configs{l+2}.act_fun='softmax';configs{l+2}.loss='categorical_cross_entropy';
optimizer.learningrate=0.001;
optimizer.momentum=0;
optimizer.opt='sgd';
model=model_init(input_shape,configs,1,optimizer);
profile on;
model=model.train(x,y,nb_epoch,3,'example/minimodel_f.mat');
profile report;
end