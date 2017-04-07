function [x,y,dic]=txt2seq(text,threshold)
if nargin<2
    threshold=50000;
end
if exist(text,'file')
    text=cell2mat(importdata(text)');
end
[dic,~,index]=unique(double(text));
for i=1:length(dic)
    if numel(find(index==i))<=length(text)/threshold
        text(index==i)=',';
    end
end
[dic,~,index]=unique(double(text));
seq=zeros(length(dic),length(index),'int8');
for i=1:length(index)
    seq(index(i),i)=1;
end
x=seq(:,1:end-1);
y=seq(:,2:end);
end
    
    