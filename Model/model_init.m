function model=model_init(input_shape,configs ,flag,optimizer)
if nargin<3
    flag=0;
end
model.flag=flag;
if nargin<4&&flag
    optimizer.type='sgd';
    optimizer.momentum=0;
    optimizer.learningrate=0.01;
end
model.layers=cell(1,length(configs)+1);
model.layers{1}=tensor_init_gpu(input_shape,'input');
for l=2:length(model.layers)
    model.layers{l}=layer_init(model.layers{l-1},configs{l-1},flag);
end
model.layers=[model.layers,0];
for l=1:length(model.layers)-1
    disp(['layer ' ,num2str(l),' :']);
    disp(model.layers{l}.configs);
end

model.input_shape=model.layers{1}.input_shape(1:end-1);
model.output_shape=model.layers{end-1}.output_shape(1:end-1);
model.batchsize=input_shape(end);
model.loss=[];
model.configs=configs;
if flag
    model.optimizer=optimizer;
    model.optimize=@(layer,optimizer,batch,epoch)layer_optimize(layer,optimizer,batch,epoch);
end
model.eval_loss=@(outputlayer,y_true,flag)eval_loss(outputlayer,y_true,flag);
model.predict=@(model,x)model_predict(model,x);
model.save=@(model,filename)model_save(model,filename);
model.evaluate=@(model,x,y_true)model_evaluate(model,x,y_true);
if flag
    model.train=@(model,x,y,nb_epoch,verbose,filename)model_train(model,x,y,nb_epoch,verbose,filename);
end
end
function layer=layer_init(prelayer,config,flag)
switch config.type
    case 'lstm'
        if isfield(config,'loss')
            layer=lstm_init_gpu(prelayer,config.hiddensize,config.return_sequence,flag,config.loss);
        else
            layer=lstm_init_gpu(prelayer,config.hiddensize,config.return_sequence,flag);
        end
    case 'dense'
        if isfield(config,'loss')
            layer=dense_init_gpu(prelayer,config.hiddensize,flag,config.loss);
        else
            layer=dense_init_gpu(prelayer,config.hiddensize,flag);
        end
    case 'activation'
        if isfield(config,'loss')
            layer=activation_init(prelayer,config.act_fun,flag,config.loss);
        else
            layer=activation_init(prelayer,config.act_fun,flag);
        end
    case 'dropout'
        if isfield(config,'loss')
            layer=dropout_init_gpu(prelayer,config.drop_rate,flag,config.loss);
        else
            layer=dropout_init_gpu(prelayer,config.drop_rate,flag);
        end
end
end