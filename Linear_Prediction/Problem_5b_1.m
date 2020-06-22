clear
[audio,fs] = audioread('Train_0_Example_1.wav');
N=70;
win = hamming(fs*25/1000);
[STFT,f,t] = stft(audio,fs,'Window',win,'overlaplength',(fs*15/1000));
n = ceil((length(audio)-length(win))/(fs*10/1000));
audio_m = zeros((n*(fs*10/1000)+length(win)),1);
audio_m(1:length(audio),1) = audio;
FFT = zeros(length(f),int8(n));
for i=1:n
    if(i == 1)
        s=1;
        e=s+length(win)-1;
    else
        s=s+160;
        e=e+160;
    end
    ws = win.*audio_m(s:e);
    [a,g] = lpc(ws,N);
    Syy = abs(freqz(1,a,f,fs));
    Syy = Syy.^2;
    Syy = g*Syy;
    FFT(:,i) = Syy.^(1/2);
end
pstft = FFT.*(cos(angle(STFT))+(sin(angle(STFT))*j));
synthetic_signal = istft(pstft,fs,'window',win,'overlaplength',(fs*15/1000),'ConjugateSymmetric',true);
%To increase amplitude of signal
synthetic_signal = synthetic_signal*10;
figure()
subplot(2,1,1)
plot(audio)
title('Original Signal')
ylabel('Amplitude')
xlabel('Sample')
subplot(2,1,2)
plot(synthetic_signal)
title('White noise base Synthetic Signal')
ylabel('Amplitude')
xlabel('Sample')
audiowrite('synthetic_white.wav',synthetic_signal,fs)