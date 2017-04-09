function mean_loss=model_evaluate(model,x,y_true)
y_pred=model.predict(model,x);
dim=size(y_true,1);
mean_loss=dim*feval(@(x)mean(x(:)),model.layers{end-1}.loss_f(single(y_true),y_pred));
end