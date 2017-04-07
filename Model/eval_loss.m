function outputlayer=eval_loss(outputlayer,y_true)
dim=size(y_true,1);
loss=dim*feval(@(x)mean(x(:)),outputlayer.loss_f(single(y_true),outputlayer.output));
outputlayer.loss=[outputlayer.loss,loss];
if isequal(outputlayer.type,'lstm')&& ~outputlayer.return_sequence
    outputlayer.e(:,end,:)=outputlayer.loss_df(y_true,outputlayer.output);
else
    outputlayer.e=outputlayer.loss_df(single(y_true),outputlayer.output);
end
end
