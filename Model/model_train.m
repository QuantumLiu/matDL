function model=model_train(model,x,y,nb_epoch,verbose,filename)
if nargin<5
    verbose=0;
end
if nargin<6
    filename=0;
end
batchsize=model.batchsize;
shape_x=size(x);
shape_y=size(y);
g_batch=1;
nb_batch=floor(shape_x(end)/batchsize)*nb_epoch;
if verbose
    h = waitbar(g_batch/nb_batch,'Training model');
end
model.epoch_loss=[];
model.batch_loss=[];
if verbose>=2
    f_epoch=figure('Name',' epochs loss');
    f_batch=figure('Name',' batches loss');
end
for epoch=1:nb_epoch
    batch=1;
    tic;
    epoch_batch_loss=[];
    while batch*batchsize<=shape_x(end)
        %% ff
        if numel(shape_x)==2
            model.layers{1}=x(:,(batch-1)*batchsize+1:batch*batchsize);
        elseif numel(shape_x)==3
            model.layers{1}=x(:,:,(batch-1)*batchsize+1:batch*batchsize);
        else
            error('The number of dims of input data must be 2/3');
        end
        for l=2:length(model.layers)-1
            model.layers{l}=model.layers{l}.ff(model.layers{l},model.layers{l-1});
        end
        %% eval
        if numel(shape_y)==2
            model.layers{end-1}=model.eval_loss(model.layers{end-1},y(:,(batch-1)*batchsize+1:batch*batchsize),model.flag);
        elseif numel(shape_y)==3
            model.layers{end-1}=model.eval_loss(model.layers{end-1},y(:,:,(batch-1)*batchsize+1:batch*batchsize),model.flag);
        else
            error('The number of dims of output data must be 2/3');
        end
        epoch_batch_loss=[epoch_batch_loss,model.layers{end-1}.loss(end)];
        cu_epoch_loss=mean(epoch_batch_loss(:));
        model.batch_loss=model.layers{end-1}.loss;
        if verbose>=3
            set(0,'CurrentFigure',f_batch);
            plot(model.batch_loss,'r-');hold off;
        end
        if verbose
            pro=num2str(100*g_batch/nb_batch);
            message=['Training model ','Epoch: ',num2str(epoch),'/',num2str(nb_epoch), ' Progress: ',pro,'%',' loss: ',num2str(cu_epoch_loss)];
            waitbar(g_batch/nb_batch,h,message);
        end
        %% bp
        for l=length(model.layers)-1:-1:2
            if model.layers{l}.trainable
                model.layers{l}=model.optimize(model.layers{l}.bp(model.layers{l},model.layers{l+1}),model.optimizer,batch,epoch);
            else
                model.layers{l}=model.layers{l}.bp(model.layers{l},model.layers{l+1});
            end
        end
        batch=batch+1;
        g_batch=g_batch+1;
    end
    toc
    model.epoch_loss=[model.epoch_loss,cu_epoch_loss];
    if verbose>=2
        set(0,'CurrentFigure',f_epoch);
        plot(model.epoch_loss,'r-');
        set(0,'CurrentFigure',f_batch);
        plot(model.batch_loss,'r-');
    end
end
if filename
    model.save(model,filename);
end
delete(h);
end