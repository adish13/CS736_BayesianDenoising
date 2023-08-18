A = importdata('assignmentSegmentBrain.mat');
mask = A.imageMask;
orig_image = A.imageData;
[row , col] = find(mask); 
[filt_image, new_mask] = filterimage(orig_image , mask);
[rows,cols] = size(filt_image);
imshow(filt_image);
k = 3;
q = 2.45;
series_objective = []; 
weights = fspecial('gaussian' , 9 , 5);
h = imshow(weights , []);
saveas(h , 'Weights.png');
bias = 0.5*ones([rows,cols]);
prev_objective = inf;
%%%%%% (Remove comments to use proposed solution )lambda_jk = 0.001*ones(rows , cols , k);
% initialise
[init_class_labels , init_class_means , class_membership] = initialise(filt_image , rows, cols , k , mask);
class_means = init_class_means;
d_kjs = zeros(rows,cols,k);
prev_objective = inf;
flag = true;
while(flag)
 w_ij_b_i_2 = conv2(bias.^2,weights, 'same');
 w_ij_b_i = conv2(bias,weights, 'same'); 

 %updating d_kj and memberships keeping everything else constant
 d_kjs = repmat(filt_image.^2 , [1,1,k])+reshape(class_means.^2 , 1,1,[]).*repmat(w_ij_b_i_2 , [1,1,k])-2*reshape(class_means , 1,1,[]).*repmat(filt_image.*w_ij_b_i , [1,1,k]);
 class_membership = calc_uj(d_kjs , new_mask, k , q);
 %%%%%%%%(Remove Comments to use proposed solution) class_membership = modify_calc_uj(d_kjs , lambda_jk, new_mask  ,bias , k , q);
 %updating class_means keeping everything else constant
 class_means = calc_ck(class_membership , filt_image , w_ij_b_i ,w_ij_b_i_2, k , q);

 %updating bias keepings everything else constant
 bias = calc_bi(class_membership , q , class_means , filt_image ,weights );
 %%%%%%%%(Remove comments to use proposed solution) bias = modified_calc_bi(class_membership , q , class_means , filt_image ,weights , lambda_jk );
 %evaluating current scores
 partial_obj=sum(class_membership.^q.*d_kjs , [1,2]);
 objective = sum(partial_obj, 'all');
 if abs(objective - prev_objective)/abs(prev_objective)< 1e-3  % a percentage change of less than 0.1%
    flag=false; 
 end
    series_objective = [series_objective , objective ];
    prev_objective = objective;
end

%calculating final results
 [bias_removed_image , residual_image] = calc_res(class_membership , class_means,filt_image , bias);

%save all images
saveall(mask ,orig_image, bias_removed_image , class_membership , bias,residual_image);

disp(series_objective);

%%%%%%% Functions %%%%%%%%

function [init_class_labels , init_class_means , class_membership] = initialise(filt_image , rows,cols,k , mask)
[mask_rows,mask_cols] = find(mask);
[init_class_labels , class_means] = kmeans(reshape(filt_image , [rows*cols , 1]) , k+1, "MaxIter",200);
init_class_labels = reshape(init_class_labels,[rows , cols]); 
initial_segment=zeros(rows,cols,3);
j=0;
for i=1:(k+1)
    if i~=find(class_means<0.1)
        j=j+1;
        initial_segment(:,:,j)=(init_class_labels==i)*class_means(i);
    end
end
initial_segment_orig = zeros(256,256,3);
initial_segment_orig(min(mask_rows): max(mask_rows) , min(mask_cols):max(mask_cols), :) = initial_segment; 
h = imshow(cat(3,initial_segment_orig(:,:,1), zeros(256,256,2)));
saveas(h , "Initial_Segment_Class_1.png");
h = imshow(cat(3,zeros(256,256),initial_segment_orig(:,:,2), zeros(256,256)));
saveas(h , "Initial_Segment_Class_2.png");
h = imshow(cat(3,zeros(256,256,2),initial_segment_orig(:,:,3)));
saveas(h , "Initial_Segment_Class_3.png");
i_max = max(filt_image , [],'all');
i_min = min(filt_image , [],'all');
range = i_max-i_min;
class_membership = zeros(rows,cols,k);
for j=1:3
        class_membership(:,:,j) = (filt_image >= i_min + range* (j-1)/k).*(filt_image < i_min + range * j/k);
end
class_means = class_means(class_means> 0.1);
init_class_means = class_means;
end

function class_membership = calc_uj(d_kjs , new_mask , k , q)
     class_membership = d_kjs.^(1/1-q);  
     class_membership(isnan(class_membership)) = inf;
     class_membership = class_membership ./repmat(sum(class_membership , 3),[1,1,k]);   
     class_membership(isnan(class_membership)) = 1;
     class_membership= class_membership.*repmat(new_mask , [1,1,k]);
end

function class_means = calc_ck(class_membership , filt_image , w_ij_b_i ,w_ij_b_i_2 ,k , q)
     num =  sum(class_membership.^q.*repmat(filt_image.*w_ij_b_i , [1,1,k]), [1,2]);
     denom = sum(class_membership.^q.*repmat(w_ij_b_i_2 , [1,1,k]) , [1,2]);
     class_means = num./denom;
end

function saveall(mask ,orig_image, bias_removed_image , class_membership , bias,residual_image)
[mask_rows,mask_cols] = find(mask);
h = imshow(orig_image);
saveas(h , "Original_image.png");
bias_removed_orig_image = zeros(256,256);
bias_removed_orig_image(min(mask_rows):max(mask_rows), min(mask_cols):max(mask_cols)) = bias_removed_image;
h = imshow(bias_removed_orig_image );
saveas(h , "Bias_Removed.png")
class_memberships = zeros(256,256,3);
class_memberships(min(mask_rows):max(mask_rows), min(mask_cols):max(mask_cols),:) = class_membership;
h = imshow(class_memberships);
saveas(h , "Computed_membership.png");
bias_orig = zeros(256,256);
bias_orig(min(mask_rows):max(mask_rows) , min(mask_cols):max(mask_cols)) = bias;
h = imshow(bias_orig);
saveas(h , "Computed_Bias.png")
residual_image_orig = zeros(256,256);
residual_image_orig(min(mask_rows):max(mask_rows) , min(mask_cols):max(mask_cols)) = residual_image;
h = imshow(1-abs(residual_image_orig));
saveas(h , "Neg_Residual_image.png")
h = imshow(abs(residual_image_orig));
saveas(h , "Residual_image.png");
end

function bias = calc_bi(class_membership , q , class_means , filt_image ,weights )
 bias_num = sum(class_membership(:,:,:).^q.*reshape(class_means(:),1,1,[]), 3);
 bias_denom = sum(class_membership(:,:,:).^q.*reshape(class_means(:).^2 , 1,1,[]) , 3);
 bias = conv2(filt_image.*bias_num,weights , 'same')./conv2(bias_denom,weights, 'same');
 bias(isnan(bias))= 0;
end

function [bias_removed_image , residual_image] = calc_res(class_membership , class_means,filt_image , bias)
bias_removed_image = sum(class_membership.*reshape(class_means , 1,1,[]) , 3);
residual_image = filt_image - bias_removed_image.*bias;
end

%%%%% Proposed changed functions %%%%%
function class_membership = modify_calc_uj(d_kjs , lambda_jk, new_mask  ,bias , k , q)
     d_kjs_modified = d_kjs + lambda_jk.*(repmat(bias.^2 , [1,1,k]));
     class_membership = d_kjs_modified.^(1/1-q);  
     class_membership(isnan(class_membership)) = inf;
     class_membership = class_membership ./repmat(sum(class_membership , 3),[1,1,k]);   
     class_membership(isnan(class_membership)) = 1;
     class_membership= class_membership.*repmat(new_mask , [1,1,k]);
end

function bias = modified_calc_bi(class_membership , q , class_means , filt_image ,weights , lambda_jk )
 bias_num = sum(class_membership(:,:,:).^q.*reshape(class_means(:),1,1,[]), 3);
 bias_denom = sum(class_membership(:,:,:).^q.*reshape(class_means(:).^2 , 1,1,[])+ class_membership(:,:,:).^q.*lambda_jk , 3);
 bias = conv2(filt_image.*bias_num,weights , 'same')./conv2(bias_denom,weights, 'same');
 bias(isnan(bias))= 0;
end