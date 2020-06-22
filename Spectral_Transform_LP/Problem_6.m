clear
[audio,fs]=audioread('Train_1_Example_1.wav');
%for 'Train_0_Example_1.wav'
%voised = audio(7305:((fs*25/1000)+7305-1));
%Unvoised = audio(10616:((fs*25/1000)+10616-1));

%for 'Train_0_Example_2.wav'
% voised = audio(8404:((fs*25/1000)+8404-1));
% Unvoised = audio(12211:((fs*25/1000)+12211-1));

%for 'Train_1_Example_1.wav'
voised = audio(3974:((fs*25/1000)+3974-1));
Unvoised = audio(5441:((fs*25/1000)+5441-1));

%for 'Train_1_Example_2.wav'
%voised = audio(5918:((fs*25/1000)+5918-1));
%Unvoised = audio(8204:((fs*25/1000)+8204-1));

figure()
plot_STLP(voised,fs);
title('Voised')
figure()
plot_STLP(Unvoised,fs);
title('Unvoised')

function []=plot_STLP(signal,fs)
    win = hamming(fs*25/1000);
    [Sxx,w] = periodogram(signal,win);
    w = (w.*fs)/(2*pi);
    q = 3;
    Sq = Sxx.^(1/q);
    R = ifft(Sq);
    N = 70;
    [a,e] = levinson(R,N);
    sqp = abs(freqz(1,a,w,fs/2));
    sqp = sqp.^2;
    sqp = sqp.*e;
    sqp = sqp.^(q);
    x = 1:length(w);
    x = x./length(w);
    plot(x,Sxx)
    xlabel('Normalised Frequency')
    ylabel('PSD')
    hold
    plot(x,sqp)
    legend('Sxx','Predicted Sxx')
end

