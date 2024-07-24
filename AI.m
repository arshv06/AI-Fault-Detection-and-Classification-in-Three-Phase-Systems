% Load the training dataset
trainingData = readtable('C:\Users\Esty\Downloads\CAPSTONE\Training\ClassData.xlsx');

% Extract the inputs and outputs
inputs = trainingData{:, 5:10}'; % Columns Ia, Ib, Ic, Va, Vb, Vc
targets = trainingData{:, 1:4}'; % Columns G, C, B, A

% Normalize inputs
inputsMean = mean(inputs, 2);
inputsStd = std(inputs, 0, 2);
inputs = (inputs - inputsMean) ./ inputsStd;

% Define fault types
faultTypes = {'AB', 'BC', 'CA', 'AG', 'BG', 'CG', 'ABC', 'ABG', 'ACG', 'BCG', 'ABCG'};
faultCombinations = [
    0 1 1 0; % AB
    0 1 0 1; % BC
    0 0 1 1; % CA
    1 1 0 0; % AG
    1 0 1 0; % BG
    1 0 0 1; % CG
    0 1 1 1; % ABC
    1 1 1 0; % ABG
    1 1 0 1; % ACG
    1 0 1 1; % BCG
    1 1 1 1; % ABCG
];

% Create and configure the neural network
hiddenLayerSize = [10, 30]; % Adjusted hidden layer size
net = patternnet(hiddenLayerSize); % Use Scaled Conjugate Gradient
net.trainParam.epochs = 4000; % Number of training epochs
net.trainParam.min_grad = 1e-18; % Lower minimum gradient threshold
net.trainParam.lr = 0.01; % Lower learning rate
net.layers{end}.transferFcn = 'logsig'; % Set activation function for the output layer
net.divideParam.trainRatio = 0.9; % 90% of data for training
net.divideParam.valRatio = 0.35; % 35% of data for validation
net.divideParam.testRatio = 0.55; % 55% of data for testing

% Train the neural network
net = train(net, inputs, targets);

% Load the test dataset
testData = readtable('C:\Users\Esty\Downloads\CAPSTONE\Test\Test.xlsx');
testInputs = testData{:, :}';

% Normalize test inputs
testInputs = (testInputs - inputsMean) ./ inputsStd;

% Predict faults in the test data
predictions = net(testInputs);

% Apply a threshold to get binary fault presence indicators
threshold = 0.5;
binaryPredictions = predictions > threshold;

% Initialize fault type classifications
trainingFaultClassification = zeros(size(trainingData, 1), 1);
testingFaultClassification = zeros(size(testData, 1), 1);

% Classify faults in the training dataset
for i = 1:size(trainingData, 1)
    for j = 1:length(faultTypes)
        if isequal(trainingData{i, 1:4}, faultCombinations(j, :))
            trainingFaultClassification(i) = j;
            break;
        end
    end
end

% Classify faults in the testing dataset
for i = 1:size(binaryPredictions, 2)
    for j = 1:length(faultTypes)
        if isequal(binaryPredictions(:, i)', faultCombinations(j, :))
            testingFaultClassification(i) = j;
            break;
        end
    end
end

% Check and correct indices for faultTypes assignment
validIndices = testingFaultClassification > 0 & testingFaultClassification <= length(faultTypes);
if ~all(validIndices)
    error('testingFaultClassification contains invalid indices.');
end

% Add fault type classification to the test data
testData.FaultType = cell(size(testingFaultClassification));
for i = 1:length(testingFaultClassification)
    if testingFaultClassification(i) > 0
        testData.FaultType{i} = faultTypes{testingFaultClassification(i)};
    else
        testData.FaultType{i} = 'None';
    end
end

% Write the updated test data to a new file
writetable(testData, 'C:\Users\Esty\Downloads\CAPSTONE\Test\TestWithFaults.xlsx');

% Plot the number of each fault type in the training data
figure;
bar(categorical(faultTypes), histcounts(trainingFaultClassification, 1:length(faultTypes) + 1));
xlabel('Fault Type');
ylabel('Count');
title('Number of Faults in Training Data');

% Plot the number of each fault type in the testing data
figure;
bar(categorical(faultTypes), histcounts(testingFaultClassification, 1:length(faultTypes) + 1));
xlabel('Fault Type');
ylabel('Count');
title('Number of Faults in Test Data');

% Calculate and plot MSE
mseTrain = perform(net, targets, net(inputs));
mseTest = perform(net, binaryPredictions, binaryPredictions);
figure;
bar(categorical({'Training MSE', 'Test MSE'}), [mseTrain, mseTest]);
xlabel('Dataset');
ylabel('MSE');
title('Mean Squared Error');

% Plot the currents and voltages in training and test data
figure;
subplot(2, 1, 1);
plot(inputs');
xlabel('Sample');
ylabel('Value');
title('Currents and Voltages in Training Data');
legend({'Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'});

subplot(2, 1, 2);
plot(testInputs');
xlabel('Sample');
ylabel('Value');
title('Currents and Voltages in Test Data');
legend({'Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'});

% Calculate and plot the number and types of faults in training data
faultCounts = sum(trainingData{:, 1:4}, 1);
figure;
bar(categorical({'G', 'C', 'B', 'A'}), faultCounts);
xlabel('Fault Type');
ylabel('Count');
title('Number of Faults in Training Data');

% Calculate and plot the number and types of faults in test data
testFaultCounts = sum(binaryPredictions, 2);
figure;
bar(categorical({'G', 'C', 'B', 'A'}), testFaultCounts);
xlabel('Fault Type');
ylabel('Count');
title('Number of Faults in Test Data');

hold off;
