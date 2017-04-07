function [f,df]=loss_handle(type)
syms y_true y_pred num
switch type
    case 'mse'
        symsf(y_true,y_pred)=(y_true-y_pred).^2;
        f=matlabFunction(symsf);
        df=matlabFunction(diff(symsf,y_pred));
        return
    case 'cross_entropy'
        symsf(y_true,y_pred)=-1.*sum(y_true.*(y_pred)+(1-y_true).*log(1-y_pred));
        f=matlabFunction(symsf);
        df=matlabFunction(diff(symsf,y_pred));
        return   
    case 'categorical_cross_entropy'
        symsf(y_true,y_pred)=-1.*y_true.*log(y_pred);
        f=matlabFunction(symsf);
        df=@(y_true,y_pre)y_pre-y_true;
end
end
