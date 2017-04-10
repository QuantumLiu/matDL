function text=textgenerate(model,dic,term)
seed=double('In the beginning Monster created the plate and the earth.Ge1:2 And the earth was without form, and void; and darkness was upon the face of the deep.');
seed=seed(1:50)
end
function index=sample(pred,temp)
pred=exp(log(double(pred))./temp);
[~,index]=max(pred,[],1);
end