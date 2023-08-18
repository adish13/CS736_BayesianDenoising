load('assignmentSegmentBrainGmmEmMrf.mat');

K = 3; % num of gaussians

%% Part a) Initialize MRF params

% Note: a 4-neighborhood system is used with potential function nonzero on
% 2-cliques

% Generating valid maps for neighbors

vMapLeft = circshift(imageMask,1,2);
vMapRight = circshift(imageMask,-1,2);
vMapTop = circshift(imageMask,1,1);
vMapBottom = circshift(imageMask,-1,1);

beta1 = 0.35;
beta2 = 0;  % No MRF prior on labels

priorFunction1 = @(candidate_label,current_labels) evalLabelPriors(...
    candidate_label,current_labels,beta1,vMapLeft,vMapRight,...
    vMapTop,vMapBottom,imageMask);

priorFunction2 = @(candidate_label,current_labels) evalLabelPriors(...
    candidate_label,current_labels,beta2,vMapLeft,vMapRight,...
    vMapTop,vMapBottom,imageMask);

%% Part b) Label initialization

% Using k-means for label initialization
% Motivation is that is gives quick division of the values into 3 classes

validImage = imageData(logical(imageMask));
[idx,C] = kmeans(validImage,K);

labelMap = zeros(size(imageData));

labelMap(logical(imageMask)) = idx;

%% Part c) Gaussian params initialization

% Using label initialization to get means and variances
% We're using the initial class means obtained using kmeans and the initial
% standard deviation estimates are the average distances of any pixel value
% from the respective class means. This gives us a quick and good estimate
% of the optimal standard deviations

means_init = C; % kmeans centroids

sigmas_init = zeros(K,1);

for i=1:K
    clusterVals = validImage(idx==i);
    sigmas_init(i) = sqrt(sumsqr(clusterVals - means_init(i))/length(clusterVals));
end

%% Part d) Perform Segmentation
xInit = labelMap;

fprintf('*** Starting modified ICM with beta = %f ***\n',beta1);
[x1,means1,sigmas1,iters1] = do_segmentation(xInit,imageData,means_init,...
    sigmas_init,20,imageMask,priorFunction1);
fprintf('\n*** Starting modified ICM with beta = %f ***\n',beta2);
[x2,means2,sigmas2,iters2] = do_segmentation(xInit,imageData,means_init,...
    sigmas_init,20,imageMask,priorFunction2);

%% Viewing results
showImage(imageData,'Corrupted Image.png')
showImage(xInit,'Initial estimate for label image.png');

%% Show images 

% With MRF
est1_mrf = zeros(size(imageData));
est2_mrf = zeros(size(imageData));
est3_mrf = zeros(size(imageData));

est1_mrf(x1==1) = imageData(x1==1);
est2_mrf(x1==2) = imageData(x1==2);
est3_mrf(x1==3) = imageData(x1==3);

showImage(est1_mrf,'Optimal membership estimate1 beta=0.35.png');
showImage(est2_mrf,'Optimal membership estimate2 beta=0.35.png');
showImage(est3_mrf,'Optimal membership estimate3 beta=0.35.png');
showImage(x1,'Optimal label image estimate for beta=0.35.png');

% Without MRF
est1_no_mrf = zeros(size(imageData));
est2_no_mrf = zeros(size(imageData));
est3_no_mrf = zeros(size(imageData));

est1_no_mrf(x2==1) = imageData(x2==1);
est2_no_mrf(x2==2) = imageData(x2==2);
est3_no_mrf(x2==3) = imageData(x2==3);

showImage(est1_no_mrf,'Optimal membership estimate1 beta=0.png');
showImage(est2_no_mrf,'Optimal membership estimate2 beta=0.png');
showImage(est3_no_mrf,'Optimal membership estimate3 beta=0.png');
showImage(x2,'Optimal label image estimate for beta=0.png');

%% Report Optimal Estimate
fprintf('\nChosen value of beta = %f',beta1);
fprintf('\nThe optimal estimates for the class means are [%f %f %f] for beta = 0.35\n',means1(1),means1(2),means1(3));


%% Functions 

function [means,sigmas] = getGaussianParameters( y,mem,vMap )
    %GetGaussianParams Evaluates params of GMM using observed image and memberships
    % Input arguments:
    % y - oberserved image
    % mem - memberships (0 for invalid pixels)
    
    K = size(mem,3); % number of gaussians
    
    means = zeros(1,K);
    sigmas = zeros(1,K);
    den = sum(mem , [1,2]);
    means = (sum(mem.*y , [1,2]))./den;
    sigmas = sqrt(sum(mem.*(repmat(reshape(y , [size(y,1),size(y,2),1]) , [ 1,1,3] ) ...
        - repmat(reshape(means , [1,1,K]) , [size(y,1),size(y,2) ,1])).^2.*repmat(vMap , [1,1,3]), [1,2])./den);
    %repmat(reshape(y , [size(y,1),size(y,2),1]) , [ 1,1,3] ) - repmat(reshape(means , [1,1,k]) , [1,1 ,3])
end

function mem = getMemberships( y,means,sigmas,x,vMap,priorFunction)
    %GetMemberships Calculates the membership as per the E-step of soft segmentation
    % Input arguments
    % y - observed image
    % means - the mean of gaussians of GMM
    % vars - variance of gaussians of GMM
    % x - the current labelling
    % vMap - the map indicating valid portions of the image
    % priorFunction - the function handle for MRF with beta and vMaps already
    % initialized

    K = length(means);

    % calculate the likelihood and prior term for each class

    likelihood = zeros(size(y,1),size(y,2),3);
    prior = zeros(size(y,1),size(y,2),3);
    for i=1:K
        likelihood(:,:,i) = ((1/(sigmas(i)*sqrt(2*pi)))*exp(-(y-means(i)).^2/(2*sigmas(i)^2))).*vMap;
        prior(:,:,i) = priorFunction(i,x);    
    end

    % normalizing prior term of each pixel
    norms = sum(prior,3);
    for i=1:size(y,1)
        for j=1:size(y,2)
            prior(i,j,:) = prior(i,j,:)/norms(i,j);
        end
    end

    mem = likelihood.*prior;
    % normalizing the memberships 
    norms = sum(mem,3);
    mem = mem./repmat(norms , [1,1,K]);


    % Setting memberships of invalid regions to be zero
    memberships = zeros(size(x,1),size(x,2));
    for i=1:K
        memberships = mem(:,:,i);
        memberships(~logical(vMap)) = 0;
        mem(:,:,i) = memberships;
    end

end

function posterior = getPosterior( x,y, means,sigmas, vMap, priorFunction )
    %GetPosterior Summary of this function goes here
    %   Detailed explanation goes here
    
    idx1 = find(x==1);
    idx2 = find(x==2);
    idx3 = find(x==3);
    
    likelihood = zeros(size(x));
    
    likelihood(idx1) = (1/(sigmas(1)*sqrt(2*pi)))*exp(-(y(idx1)-means(1)).^2/(2*(sigmas(1)^2)));
    likelihood(idx2) = (1/(sigmas(2)*sqrt(2*pi)))*exp(-(y(idx2)-means(2)).^2/(2*(sigmas(2)^2)));
    likelihood(idx3) = (1/(sigmas(3)*sqrt(2*pi)))*exp(-(y(idx3)-means(3)).^2/(2*(sigmas(3)^2)));
    
    prior = priorFunction(x,x);
    
    posterior = likelihood.*prior.*vMap;

end


function p = evalLabelPriors( candidate_label,x,beta,vMapLeft,...
    vMapRight,vMapTop,vMapBottom,vMap)
    %EvaluateLabelPriors Evaluates prior on labels
    %   Input arguments
    %   candidate_label - the candidate for the whole image (evaluated at once for
    %   speedup)
    %   x - labels
    %   beta - penalty for dissimilar labels
    %   vMap - valid pixels

    % Evaluating on 4-neighborhood system


    if length(candidate_label)==1
        candidate_img = candidate_label*ones(size(x));
    else
        candidate_img = candidate_label;
    end

    topArray = ((candidate_img-circshift(x,1,1)).*vMapTop)~=0;
    bottomArray = ((candidate_img-circshift(x,-1,1)).*vMapBottom)~=0;
    leftArray = ((candidate_img-circshift(x,1,2)).*vMapLeft)~=0;
    rightArray = ((candidate_img-circshift(x,-1,2)).*vMapRight)~=0;


    penalty = (topArray+bottomArray+leftArray+rightArray).*beta;

    p  = exp(-penalty).*vMap;

end

function [x,means,sigmas,max_iterations] = do_segmentation( x,y,means,sigmas,max_iterations,vMap,priorFunction)
    % Perform segmentation using modified ICM
    % Input arguments:
    % x - initial labels
    % y - oberved image
    % means - initial means of Gaussians
    % sigmas - intial s.d of Gaussians
    % max_iterations - maximum number of iterations
    % vMap - the map indicating the valid regions
    % a = size(x,1)
    % b = size(x,2)
    mem = zeros(size(x,1),size(x,2),length(means));
    xNew = zeros(size(x));
    posterior = zeros(size(x));
    
    % Log Posterior before and after
    lp_before = 0;
    lp_after = 0;
    
    for i=1:max_iterations
        % get posterior
        posterior = getPosterior(x,y,means,sigmas,vMap,priorFunction);
        lp_before = sum(log(posterior(logical(vMap))));
        fprintf('Iteration %d: Log posterior before = %f\n',i,lp_before);
        
        % getting memberships
        mem = getMemberships(y,means,sigmas,x,vMap,priorFunction);
        
        % generating the new label maps
        [~,xNew] = max(mem,[],3);
        xNew = xNew.*vMap;
        
        % get posterior after update
        posterior = getPosterior(xNew,y,means,sigmas,vMap,priorFunction);
        lp_after = sum(log(posterior(logical(vMap))));
        fprintf('Iteration %d: Log posterior after = %f\n',i,lp_after);
        
        % Terminate if "log posterior after" value does not increase
        if lp_after < lp_before
            break;
        end
        
        % Getting new params
        [means,sigmas] = getGaussianParameters(y,mem,vMap);
        
        diff = any(x~=xNew);
        if ~diff
            break;
        end 
        x = xNew;

    end
end

function showImage( image, image_title )
    figure;
        my_num_of_colors = 200;
        my_color_scale = [ [0:1/(my_num_of_colors-1):1]' , ...
            [0:1/(my_num_of_colors-1):1]' , [0:1/(my_num_of_colors-1):1]' ];
        h = imagesc (single (image));
        title(image_title);
        colormap (my_color_scale);
        colormap gray;
        daspect ([1 1 1]);
        axis tight;
        colorbar;
        saveas(h,image_title);
end