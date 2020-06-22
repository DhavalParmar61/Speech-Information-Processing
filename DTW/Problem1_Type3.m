clear
[audio1_o,fs1] = audioread('./Data_processed/Train_1_Example_1.wav');
[audio2_o,fs2] = audioread('./Data_processed/Test_Sample_j.wav');

l1 = length(audio1_o);
l2 = length(audio2_o);
for i=200:l1 %starting from 200 frame to avoid initial noise
    if(abs(audio1_o(i))>(10^-3)) %removing silence portion
        s=i;
        break;
    end
end
for i=(l1-200):-1:1 %starting from 200 frame to avoid initial noise
    if(abs(audio1_o(i))>(10^-3)) %removing silence portion
        e=i;
        break;
    end
end
audio1 = audio1_o(s:e);

for i=200:l2%starting from 200 frame to avoid initial noise
    if(abs(audio2_o(i))>(10^-3)) %removing silence portion
        s=i;
        break;
    end
end
for i=(l2-200):-1:1 %starting from 200 frame to avoid initial noise
    if(abs(audio2_o(i))>(10^-3)) %removing silence portion
        e=i;
        break;
    end
end
audio2 = audio2_o(s:e);

global mfcc1
global mfcc2
mfcc1 = mfcc(audio1,fs1);
mfcc2 = mfcc(audio2,fs2);
max = 99999999;
t1 = length(mfcc1);
t2 = length(mfcc2);
D(1:t1,1:t2) = 999;
D(1,1) = d(1,1);
for i=2:t1
    if(2<=i && i<((2*t2-t1+2)/3))
        s = round(((i-1)/2)+1);
        e = round(2*(i-1)+1);
    elseif(((2*t2-t1+2)/3)<=i && i<((4*t1-2*t2+1)/3))
        s = round(((i-1)/2)+1);
        e = round(((i-t1)/2)+t2);
    else
        s = 2*(i-t1)+t2;
        e = round(((i-t1)/2)+t2);
    end
    for j=s:e
        if(i>2)
            o1 = D(i-2,j-1)+(3*d(i,j));
        else
            o1 = max;
        end
        o2 = D(i-1,j-1)+ (2*d(i,j));
        if(j>2)
            o3 = D(i-1,j-2)+(3*d(i,j));
        else
            o3 = max;
        end
        D(i,j)= min([o1,o2,o3]);
    end
end
distance = D(t1,t2)/(t1+t2);


function x=d(i,j)
    global mfcc1
    global mfcc2
    x = 1-((mfcc1(i)*mfcc2(j)')/(norm(mfcc1(i))*norm(mfcc2(j))));
end