% uplink_massive_MIMO_detection.m
% QAM modulation and demodulation
% Source https://www.mathworks.com/help/comm/gs/examine-16-qam-using-matlab.html
% The original code is changed to support uplink massive MIMO detection


clc
clear all
close all

% Generate Random Binary Data Stream
M = 16; % Modulation order (alphabet size or number of points in signal constellation)
K = 16; % Number of users
N = 128; % Number of antennas at the BTS
k = log2(M); % Number of bits per symbol
n = 3000*k*K; % Number of bits to process

SNR = 10;

rng default;
dataIn = randi([0 1],n,1); % Generate vector of binary data


stem(dataIn(1:40),'filled');
title('Random Bits');
xlabel('Bit Index');
ylabel('Binary Value');

% Convert Binary Signal to Integer-Valued Signal
dataInMatrix = reshape(dataIn,length(dataIn)/k,k);
dataSymbolsIn = bi2de(dataInMatrix);

% Plot the first 10 symbols in a stem plot.
figure; % Create new figure window.
stem(dataSymbolsIn(1:10));
title('Random Symbols');
xlabel('Symbol Index');
ylabel('Integer Value');


% Modulate Using 16-QAM
dataMod = qammod(dataSymbolsIn,M,'bin'); % Binary coding with phase offset of zero
dataModG = qammod(dataSymbolsIn,M); % Gray coding with phase offset of zero

dataModK = reshape(dataMod,K,length(dataMod)/K); % Binary coding with phase offset of zero
dataModGK = reshape(dataModG,K,length(dataMod)/K); % Gray coding with phase offset of zero


% randon i.i.d channel generation H
H = randn(N,K)+0*1i*randn(N,K);
receivedSignalK = H*dataModK;
receivedSignalGK = H*dataModGK;

% Add White Gaussian Noise

snr = SNR+10*log10(k);
receivedSignalK = awgn(receivedSignalK,snr,'measured');
receivedSignalGK = awgn(receivedSignalGK,snr,'measured');

% Massive MIMO detection
% Matched filtering
A = H'*H;  
Diag_A = diag(diag(A));
receivedSignalK_MF = inv(Diag_A)*H'*receivedSignalK;
receivedSignalGK_MF = inv(Diag_A)*H'*receivedSignalGK;
% % Create Constellation Diagram after matched filtering
% sPlotFig = scatterplot(reshape(receivedSignalK_MF,length(dataMod),1),1,0,'g.');
% hold on
% scatterplot(dataMod,1,0,'k*',sPlotFig)

% receivedSignalK_ZF = inv(A)*H'*receivedSignalK;
% receivedSignalGK_ZF = inv(A)*H'*receivedSignalGK;
% % % Create Constellation Diagram after matched filtering
% sPlotFig = scatterplot(reshape(receivedSignalK_ZF,length(dataMod),1),1,0,'g.');
% hold on
% scatterplot(dataMod,1,0,'k*',sPlotFig)

% NSE
% Direct (exact) matrix inversion
Ainv=inv(A);
% Isolate diagonal and off-diagonal parts of A
D=diag(diag(A));
E=A-D;
Dinv=inv(D);
% Computing Neumann series approximation of order 2, 3 and 4
Ainv2=Dinv-Dinv*E*Dinv;
Ainv3=Dinv-Dinv*E*Dinv+Dinv*E*Dinv*E*Dinv;
Ainv4=Dinv-Dinv*E*Dinv+Dinv*E*Dinv*E*Dinv-Dinv*E*Dinv*E*Dinv*E*Dinv;

receivedSignalK_NSE = Ainv*H'*receivedSignalK;
receivedSignalGK_NSE = Ainv*H'*receivedSignalGK;

% % Create Constellation Diagram after matched filtering
sPlotFig = scatterplot(reshape(receivedSignalK_NSE,length(dataMod),1),1,0,'g.');
hold on
scatterplot(dataMod,1,0,'k*',sPlotFig)

% Demodulate 16-QAM
dataSymbolsOut = qamdemod(reshape(receivedSignalK_NSE,length(dataMod),1),M,'bin');
dataSymbolsOutG = qamdemod(reshape(receivedSignalGK_NSE,length(dataMod),1),M);

% Convert Integer-Valued Signal to Binary Signal
dataOutMatrix = de2bi(dataSymbolsOut,k);
dataOut = dataOutMatrix(:); % Return data in column vector
dataOutMatrixG = de2bi(dataSymbolsOutG,k);
dataOutG = dataOutMatrixG(:); % Return data in column vector

% Compute System BER

[numErrors,ber] = biterr(dataIn,dataOut);
fprintf('\nThe binary coding bit error rate is %5.2e, based on %d errors.\n', ber,numErrors)

[numErrorsG,berG] = biterr(dataIn,dataOutG);
fprintf('\nThe Gray coding bit error rate is %5.2e, based on %d errors.\n', berG,numErrorsG)

% Compute System EVM

evm = comm.EVM('MaximumEVMOutputPort',true,...
               'XPercentileEVMOutputPort',true, 'XPercentileValue',90,...
               'SymbolCountOutputPort',true);
[rmsEVM,maxEVM,pctEVM,numSym] = evm(reshape(dataModK,length(dataMod),1),reshape(receivedSignalK_NSE,length(dataMod),1));

fprintf('\nThe RMS error vector magnetude (EVM) is %3.2e percent, and the maximum EVM is %3.2e percent.\n', rmsEVM,maxEVM)






