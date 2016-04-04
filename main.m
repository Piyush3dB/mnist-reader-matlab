close all;
clear;
clc;
format compact

addpath('/home/piyush/Downloads/GitHub/mxnet/matlab');

MNIST_DIR = '/home/piyush/Downloads/GitHub/mxnet/example/image-classification/mnist';


IMG_FILE = fullfile(MNIST_DIR, 'train-images-idx3-ubyte');
LAB_FILE = fullfile(MNIST_DIR, 'train-labels-idx1-ubyte');


%images = loadMNISTImages(IMG_FILE);
%labels = loadMNISTLabels(LAB_FILE);

%% Load the model

clear model1;
clear model2;
model1 = mxnet.model;
model2 = mxnet.model;

model1.load('/home/piyush/Downloads/GitHub/mxnet/example/image-classification/MLP-0'  , 9);
model2.load('/home/piyush/Downloads/GitHub/mxnet/example/image-classification/LENet-0', 9);
for i = 0:100
    [images, labels] = readMNIST(IMG_FILE, LAB_FILE, 1, i);
    
    
    
    
    %% Run prediction
    pred1 = model1.forward(reshape(images, 28*28, 1));
    pred2 = model2.forward(images);
    
    [p, i1] = max(pred1);
    [p, i2] = max(pred2);
    fprintf('True=%d. MLP=%d. LENET=%d\n', labels, i1-1, i2-1);

    
    figure(1);
    clf;
    plot([0:9], pred1, '-+b');
    hold on;
    plot([0:9], pred2, '-r');
    plot(labels, 1, '*');
    
    pause(0.1);
    
    continue
    
    %% find the predict label
    [p, i] = max(pred);
    fprintf('the best result is %2d [%2d], with probability %f\n', i-1, labels, p)
    
    %% Print the last 10 layers in the symbol
    
    sym = model.parse_symbol();
    layers = {};
    for i = 1 : length(sym.nodes)
        if ~strcmp(sym.nodes{i}.op, 'null')
            layers{end+1} = sym.nodes{i}.name;
            fprintf('layer name: %s\n', layers{end})
        end
    end
    
    
    
end

