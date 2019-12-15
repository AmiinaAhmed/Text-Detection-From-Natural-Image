function  [labels]= TextDetector(~)

[filename, ~] = imgetfile ;
info=imfinfo(filename);
fileinfo=info.Filename;
gray=rgb2gray(imread(fileinfo));
figure('Name','Original Image');
imshow(gray);

[rows,cols]=size(gray);
res=zeros(rows,cols);
afterbgsuppress=gray;

for i=1:8:rows
    for j=1:8:cols
        res(i:min(rows,i+8),j:min(j+8,cols))=dct2(gray(i:min(rows,i+8),j:min(j+8,cols)));
        res(i,j)=0;
    end
end
for i=1:8:rows
    for j=1:8:cols
        afterbgsuppress(i:min(rows,i+8),j:min(j+8,cols))=idct2(res(i:min(rows,i+8),j:min(j+8,cols)));        
    end
end
afterbgsuppress=uint8(afterbgsuppress);

figure('Name','After background supression');
imshow(afterbgsuppress);

[~,D]=featurextract(afterbgsuppress);

thresh1=0;
thresh2=0;
blocks=size(D,1);
for i=1:blocks
    thresh1=thresh1+D(i,12)+D(i,6)+D(i,8)+D(i,10);
    thresh2=thresh2+D(i,11)+D(i,5)+D(i,7)+D(i,9);
end
thresh1=1.2*thresh1/(4*blocks);
thresh2=1.2*thresh2/(4*blocks);
classes=zeros(blocks,1);

for i=1:blocks
    if D(i,5)<=thresh2 && D(i,7)<=thresh2  && D(i,9)<=thresh2   && D(i,11)<=thresh2 && D(i,12)>=thresh1&& D(i,6)>=thresh1&& D(i,8)>=thresh1&& D(i,10)>=thresh1
        classes(i)=1;
    end
end

block=0;
for i=1:blocks
    if classes(i)==1
        block=block+1;
        D1(block,:)=D(i,:);
    end
end

blocks=size(D1,1);
classifyImage=zeros(rows,cols);
for i=1:blocks
    classifyImage(D1(i,1):D1(i,2),D1(i,3):D1(i,4))=gray(D1(i,1):D1(i,2),D1(i,3):D1(i,4));
end
  figure('Name','After classification');
  imshow(uint8(classifyImage));

D2=mergeBlocks(D1);

blocks=size(D,1);
block=size(D2,1);
labels=zeros(1,blocks);
for i=1:blocks
    for j=1:block
        if D(i,1)>=D2(j,1) && D(i,3)>=D2(j,3) && D(i,2)<=D2(j,2) && D(i,4)<=D2(j,4)
            labels(i)=1;
            break;
        end
    end
end

blocks=size(D2,1);
finalImage=zeros(rows,cols);
for i=1:blocks
    finalImage(D2(i,1):D2(i,2),D2(i,3):D2(i,4))=gray(D2(i,1):D2(i,2),D2(i,3):D2(i,4));
end
  figure('Name','final image');
   imshow(uint8(finalImage));
end

function [blocks,D]=featurextract(afterbgsuppress)
blocks=0;
[rows,cols]=size(afterbgsuppress);
for i=1:50:rows
    for j=1:50:cols
        blocks=blocks+1;
        [h,c]=homo_con(afterbgsuppress(i:min(rows,i+50),j:min(j+50,cols)));
        D(blocks,:)=[i,min(rows,i+50),j,min(j+50,cols),h(1),c(1),h(2),c(2),h(3),c(3),h(4),c(4),0];
    end
end
end

function [homogenity,contrast]=homo_con(arr)
% offsets=[0 1; -1 1;-1 0;-1 -1];
% P=graycomatrix(arr,'offset',offsets);
P(1,:,:)=graycomatrix(arr,'offset', [0 1]);
P(2,:,:)=graycomatrix(arr,'offset', [-1 1]);
P(3,:,:)=graycomatrix(arr,'offset', [-1 0]);
P(4,:,:)=graycomatrix(arr,'offset', [-1 -1]);
[~,rows,cols]=size(P);
R(1,1:4)=0;
for i=1:rows
    for j=1:cols
        for k=1:4
            R(k)=R(k)+P(k,i,j);
        end
    end
end
homogenity(1,1:4)=0;
contrast(1,1:4)=0;
for i=1:rows
    for j=1:cols
        for k=1:4
            homogenity(k)=homogenity(k)+(P(k,i,j)/R(k))^2;
            contrast(k)=contrast(k)+(abs(i-j))^2*P(k,i,j)/R(k);
        end
    end
end
end

function [merged_blocks]=mergeBlocks(D)
D(:,1:4)
blocks=size(D,1);
block=0;
lookup=zeros(blocks,1);
for i=1:blocks
    lookup(i)=i;
end
for i=1:blocks
    ind=lookup(i);
    if ind==i 
        tmp=D(ind,:);
        lookup(i)=block+1;
    else
        tmp=merged_blocks(ind,:);
    end
    for j=1:blocks
        if i==j
            continue;
        end
        ind2=lookup(j);
        if ind2==j && j~=1
            tmp2=D(ind2,:);
        else
            tmp2=merged_blocks(ind2,:);
        end
        if   rowequivalent(tmp,tmp2) || colequivalent(tmp,tmp2)  rowequivalent(D(i,:),D(j,:)) || colequivalent(D(i,:),D(j,:)) 
            tmp(1)=min(tmp(1),tmp2(1));
            tmp(2)=max(tmp(2),tmp2(2));
            tmp(3)=min(tmp(3),tmp2(3));
            tmp(4)=max(tmp(4),tmp2(4));
            lookup(j)=lookup(i);
        end            
    end
    if ind==i 
        block=block+1;
        merged_blocks(block,:)=tmp;
    else
        merged_blocks(ind,:)=tmp;
    end
end
merged_blocks=removeDuplicates(merged_blocks);
end

function [flag]=rowequivalent(D1,D2)
rmin1=D1(1);
rmin2=D2(1);
rmax1=D1(2);
rmax2=D2(2);
cmin1=D1(3);
cmin2=D2(3);
cmax1=D1(4);
cmax2=D2(4);
if (rmin2<=rmin1 && rmin1<=rmax2 && cmin2<=cmin1 && cmin1<=cmax2) || (rmin2<=rmin1 && rmin1<=rmax2 && cmin2<=cmax1 && cmax1<=cmax2) || (rmax1>=rmin2 && rmax1<=rmax2 && cmin2<=cmin1 && cmin1<=cmax2) || (rmax1>=rmin2 && rmax1<=rmax2 && cmin2<=cmax1 && cmax1<=cmax2)
    flag=1;
else
    flag=0;
end
end
function [flag]=colequivalent(D1,D2)
rmin1=D1(1);
rmin2=D2(1);
rmax1=D1(2);
rmax2=D2(2);
cmin1=D1(3);
cmin2=D2(3);
cmax1=D1(4);
cmax2=D2(4);
if (cmin2<=cmin1 && cmin1<=cmax2 && rmin2<=rmin1 && rmin1<=rmax2) || (cmin2<=cmin1 && cmin1<=cmax2 && rmin2<=rmax1 && rmax1<=rmax2) || (cmax1>=cmin2 && cmax1<=cmax2 && rmin2<=rmin1 && rmin1<=rmax2) || (cmax1>=cmin2 && cmax1<=cmax2 && rmin2<=rmax1 && rmax1<=rmax2)
    flag=1;
else
    flag=0;
end
end

function [D1]=removeDuplicates(D)
blocks=size(D,1);
block=0;
for i=1:blocks
    flag=0;
    for j=i+1:blocks
        if i==j
            continue;
        end
        if D(i,1)>=D(j,1) && D(i,2)<=D(j,2) && D(i,3)>=D(j,3) && D(i,4)<=D(j,4)
            flag=1;
            break;
        end
    end
    if flag==0 || i==blocks
        block=block+1;
        D1(block,:)=D(i,:);
    end
end
end