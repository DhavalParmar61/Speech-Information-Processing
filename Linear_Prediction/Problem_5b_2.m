clear
[audio,fs] = audioread('Train_0_Example_1.wav');
N=70;
win = hamming(fs*25/1000);
[STFT,f,t] = stft(audio,fs,'Window',win,'OverlapLength',(fs*15/1000));
n = ceil((length(audio)-length(win))/(fs*10/1000));
audio_e = zeros((n*(fs*10/1000)+length(win)),1);
audio_e(1:length(audio),1) = audio;
FFT = zeros(length(f),int8(n));
for i=1:n
    if(i == 1)
        s=1;
        e=s+length(win)-1;
    else
        s=s+160;
        e=e+160;
    end
    sp = audio_e(s:e);
    ws = win.*audio_e(s:e);
    [a,g] = lpc(ws,N);
    Syy = abs(freqz(1,a,f,fs));
    Syy = Syy.^2;
    fft_sp = fft(sp);
    pow = fft_sp.*conj(fft_sp);
    pow =sum(pow);
    if(pow > 5)
        sp = audio_e(s:(e+450));%extending window to match minimum length needed for pitch function 
        p = pitch(sp, fs);
        T_impulse = round(fs/p);
        impulse = zeros(length(win),1);
        for k=1:round(length(win)/T_impulse)
            impulse(((k-1)*T_impulse)+1)= 1;
        end
        psd_x = periodogram(impulse,win,f,fs);
        Syy = psd_x.*Syy;
    else
        Syy = g*Syy;
    end
    FFT(:,i) = Syy.^(1/2);
end
pstft = FFT.*(cos(angle(STFT))+(sin(angle(STFT))*j));
synthetic_signal = istft(pstft,fs,'window',win,'overlaplength',(fs*15/1000),'ConjugateSymmetric',true);
%To increase amplitude of signal
synthetic_signal = synthetic_signal*20;
figure()
subplot(2,1,1)
plot(audio)
title('Original Signal')
ylabel('Amplitude')
xlabel('Sample')
subplot(2,1,2)
plot(synthetic_signal)
title('Impulse Base Synthetic signal')
ylabel('Amplitude')
xlabel('Sample')
audiowrite('synthetic_impulse.wav',synthetic_signal,fs)