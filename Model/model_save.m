function model_save(model,filename)
% if nargin<3
%     batchsize=model.input_shape(end);
% end
minimodel.input_shape=model.input_shape;
minimodel.output_shape=model.output_shape;
minimodel.configs=model.configs;
for l=2:length(model.layers)-1
    if  model.layers{l}.trainable
    minimodel.Ws{l}=gather(model.layers{l}.W);
    end
end
save(filename,'minimodel','-v7.3');
end

