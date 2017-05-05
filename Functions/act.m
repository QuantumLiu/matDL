function y=act(x,fun)
switch fun
    case 'sigmoid'
        y = 1./(1+exp(-x));
        return
    case 'tanh'
        y=tanh(x);
        return
    case 'softmax'
        E=exp(x- max(x,[],2));
        y =  E./ sum(E,2) ;
        return
    case 'Relu'
        y=x.*(x>0);
        return
    case 'linear'
        y=x;
        return
end
end