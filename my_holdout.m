function [train_data,train_label,test_data,test_label] = my_holdout(data,labels,train_percent)

nSamples = size(data,2);
array_split = randperm(nSamples);
nSplit = ceil(train_percent*nSamples);

train_data  = data(:,array_split(1:nSplit));
train_label = labels(array_split(1:nSplit))';
test_data   = data(:,array_split(nSplit+1:end));
test_label  = labels(array_split(nSplit+1:end))';

end

