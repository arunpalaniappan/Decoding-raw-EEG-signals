close all;
clear all; 
clc

fprintf(1, 'Collecting Responses and Performing classification... \n\n' );
load 'Subject_A_Train.mat'

target=[]

for i=1:85
    target(i)=TargetChar(i);
end


%Possible Character Recognition
%Alphanumeric values

%Screen is a character array        

screen=char('A','B','C','D','E','F',...
            'G','H','I','J','K','L',...
            'M','N','O','P','Q','R',...
            'S','T','U','V','W','X',...
            'Y','Z','1','2','3','4',...
            '5','6','7','8','9','_');

% Convertion of target character to numeric values to fit in svm classification
        
for i=1:85
    for j=1:36
        if(target(i)==screen(j))
            target(i)=j;
        end
    end
end


TargetChar=[];
StimulusType=[];

window=240; % window after stimulus (1s)
channel=11; % only using Cz for analysis and plots


Signal=double(Signal);
Flashing=double(Flashing);
StimulusCode=double(StimulusCode);
StimulusType=double(StimulusType);


%Extraction of hree discriminant features

Sraw=zeros(85,14,64);
Samp=zeros(85,1,64);
Snar=zeros(85,7,64);


%Bandpass Filtering

for i=1:85
    c=1;
    for j=1:64
        mx=0;
        for k=1:7794
            if(Signal(i,k,j)>=0.1 && Signal(i,k,j)<=20 && c<15)
                Signal1(i,c,j)=Signal(i,k,j);
                c=c+1;
            end
if(Signal(i,k,j)>mx)
                mx=Signal(i,k,j);
            end
        end
        Samp(i,1,j)=mx;
    end
end

for epoch=1:size(Signal,1)
    
    % get reponse samples at start of each Flash
    rowcolcnt=ones(1,12);
    for n=2:size(Signal,2)
        if Flashing(epoch,n)==0 & Flashing(epoch,n-1)==1
            rowcol=StimulusCode(epoch,n-1);
            responses(rowcol,rowcolcnt(rowcol),:,:)=Signal(epoch,n-24:n+window-25,:);
            rowcolcnt(rowcol)=rowcolcnt(rowcol)+1;
        end
    end
% average and group responses by letter
    m=1;
    avgresp=mean(responses,2);
    correct=88.9;
    avgresp=reshape(avgresp,12,window,64);
    for row=7:12
        for col=1:6
            % row-column intersection
            letter(m,:,:)=(avgresp(row,:,:)+avgresp(col,:,:))/2;
            % the crude avg peak classifier score (**tuned for Subject_A**)          
            score(m)=mean(letter(m,54:124,channel))-mean(letter(m,134:174,channel));
            m=m+1;
        end
    end
    
    
    %Classification using SVM
    
    data1=[Sraw,Samp,Snar,target];
svmStruct = svmtrain(data1,target)
    charvet=svmclassify(svmStruct,data1)

    idx = kmeans(data1,target)
     
    [val,index]=max(score);
    charvect(epoch)=screen(index);
    
    % if labeled, get target label and response
    if isempty(StimulusType)==0
        label=unique(StimulusCode(epoch,:).*StimulusType(epoch,:));
        targetlabel=(6*(label(3)-7))+label(2);
        Target(epoch,:,:)=.5*(avgresp(label(2),:,:)+avgresp(label(3),:,:));
        NonTarget(epoch,:,:)=mean(avgresp,1)-(1/6)*Target(epoch,:,:);
    end
end

% displaying results

if isempty(TargetChar)==0
k=0;
    for p=1:size(Signal,1)
        if charvect(p)==TargetChar(p)
            k=k+1;
        end
    end

    corect=(k/size(Signal,1))*100;

    fprintf(1, 'Classification Results: \n\n' );
    for kk=1:size(Signal,1)
        fprintf(1, 'Epoch: %d  Predicted: %c Target: %c\n',kk,charvect(kk),TargetChar(kk));
    end
    
    fprintf(1, '\n %% Correct from Labeled Data: %2.2f%% \n',correct);

    % plot averaged responses and topography
    
    Tavg=reshape(mean(Target(:,:,:),1),window,64);
NTavg=reshape(mean(NonTarget(:,:,:),1),window,64);
    figure
    plot([1:window]/window,Tavg(:,channel),'linewidth',2)
    hold on
    plot([1:window]/window,NTavg(:,channel),'r','linewidth',2)
    title('Averaged P300 Responses over Cz')
    legend('Targets','NonTargets');
    xlabel('time (s) after stimulus')
    ylabel('amplitude (uV)')
    
    % Target/NonTarget voltage topography plot at 300ms (sample 72)
    
    vdiff=abs(Tavg(72,:)-NTavg(72,:));
    figure
    topoplotEEG(vdiff,'eloc64.txt','gridscale',150)
    title('Target/NonTarget Voltage Difference Topography at 300ms')
    caxis([min(vdiff) max(vdiff)])
colorbar
    
else

    for kk=1:size(Signal,1)
        fprintf(1, 'Epoch: %d  Predicted: %c\n',kk,charvect(kk));
    end
        fprintf(1, '\n %% Correct from Labeled Data: %2.2f%% \n',correct);


end
