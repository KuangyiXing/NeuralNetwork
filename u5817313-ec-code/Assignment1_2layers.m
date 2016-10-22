function y = Assignment1_2layers(x) %treate the network as a function 
data = csvread('fertility.csv')   %read the csv, can be get rid of 
inputs = data(:,1:end-1)'   
outputs = data(:,end)'
CVO = cvpartition (outputs(1,:), 'k', 10);   % 10th cross-validation
err = zeros(CVO.NumTestSets,1);     
for i = 1:CVO.NumTestSets
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    train_inputs = inputs(:,trIdx);
    train_outputs = outputs(:,trIdx);
    test_inputs = inputs(:,teIdx);
    test_outputs = outputs(:,teIdx);
    train_outputs = bsxfun(@eq,train_outputs(:),[0,1])'; %create two class classification
    test_outputs = bsxfun(@eq,test_outputs(:),[0,1])';   %create two class claasification 
    net = network(1,2,[1;1],[1;0],[0 0;1 0],[0 1]);   %the network has one input, two layers, the first and second layer has bias, each layer is only fully connected to following layer
    net.layers{1}.size = x;  %the variable
    net.layers{2}.size = 2;   %the output layer neuron number is 2
    net.inputs{1}.size = 9;   %the input neuron number is 9, equivelant to number of input attributes
    net.layers{1}.transferFcn = 'logsig';  %the first layer transferring function is logsig.
    net.layers{2}.transferFcn = 'logsig';  %the second layer transferring fuction is logsig.
    net.trainFcn = 'trainlm';   %the network training training function is trainlm
    net = train(net, train_inputs, train_outputs);%train the network
    y = net(test_inputs);   %y is the actual output
    y = vec2ind(y);   %transfer the actual output martrix into indix
    test_outputs = vec2ind(test_outputs);   %tranfer the desired output matrix into indix
    err(i) = sum(y~=test_outputs)/length(test_outputs);   %compare the difference between actual output and desired output and calculate the error
end

y = sum(err)/CVO.NumTestSets; %summary all the error after ten iterations.


