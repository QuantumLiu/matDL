function y_pred=model_predict(model,x)
batchsize=model.batchsize;
shape_x=size(x);
nb_batch=floor(shape_x(1)/batchsize);
m=mod(shape_x(1),batchsize);
y_pred=zeros([shape_x(1),model.output_shape],'single');
for batch=1:nb_batch
    %% ff
    if numel(shape_x)==2
        model.layers{1}=x((batch-1)*batchsize+1:batch*batchsize,:);
    elseif numel(shape_x)==3
        model.layers{1}=x((batch-1)*batchsize+1:batch*batchsize,:,:);
    else
        error('The number of dims of input data must be 2/3');
    end
    for l=2:length(model.layers)-1
        model.layers{l}=model.layers{l}.ff(model.layers{l},model.layers{l-1});
    end
    if numel(size(y_pred))>2
        y_pred((batch-1)*batchsize+1:batch*batchsize,:,:)=gather(model.layers{end-1}.output);
    else
        y_pred((batch-1)*batchsize+1:batch*batchsize,:)=gather(model.layers{end-1}.output);
    end
end
if m
    if numel(shape_x)==2
        model.layers{1}=x(end-batchsize+1:end,:);
    elseif numel(shape_x)==3
        model.layers{1}=x(end-batchsize+1:end,:,:);
    else
        error('The number of dims of input data must be 2/3');
    end
    for l=2:length(model.layers)-1
        model.layers{l}=model.layers{l}.ff(model.layers{l},model.layers{l-1});
    end
    if numel(size(y_pred))>2
        y_pred(end-batchsize+1:end,:,:)=gather(model.layers{end-1}.output);
    else
        y_pred(end-batchsize+1:end,:,:)=gather(model.layers{end-1}.output);
    end
end
end