function [mask,mask_index]=drop(mask,drop_rate)
mask_index=randperm(numel(mask),floor(numel(mask)*drop_rate));
mask(mask_index)=0;
end