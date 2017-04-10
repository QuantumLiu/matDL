function model=model_load(minimodel,batch_size,flag,optimizer)
if nargin<2
    batch_size=32;
end
if nargin<3
    flag=0;
end
if nargin<4&&flag
    optimizer.type='sgd';
    optimizer.momentum=0;
    optimizer.learningrate=0.01;
elseif ~flag
    optimizer=[];
end
if isequal(class(minimodel),'char')
    load(minimodel);
end
model=model_init([minimodel.input_shape,batch_size],minimodel.configs,flag,optimizer);
end