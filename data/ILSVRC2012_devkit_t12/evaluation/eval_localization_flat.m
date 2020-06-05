function labels_error = eval_localization_flat ( predict_file, gtruth_file, gtruth_dir, meta_file, max_num_pred_per_image )
% Evaluate flat localization error
% predict_file: each line is the predicted labels ( must be positive
% integers, seperated by white spaces ) followed by the detected location
% of that object for one image, sorted by confidence in descending order. 
% <label(1)> <xmin(1)> <ymin(1)> <xmax(1)> <ymax(1)>  <label(2)> <xmin(2)> <ymin(2)> <xmax(2)> <ymax(2)> ....
% The number of labels per line can vary, but not
% more than max_num_pred_per_image ( extra labels are ignored ).
% gtruth_file: each line is the ground truth labels, in the same format.

pred = dlmread(predict_file);
gt_labels = dlmread(gtruth_file);
gt = dir(sprintf('%s/*.xml',gtruth_dir));
load (meta_file);
hash = make_hash(synsets);
n = size(gt_labels,2);
%% extra labels are ignored
if size(pred,2) > max_num_pred_per_image*5
    pred = pred(:,1:max_num_pred_per_image*5);
end
assert(size(pred,1)==size(gt_labels,1));
%assert(size(pred,1)==size(gt,1));
num_guesses = size(pred,2)/5;

pred_labels = []; %%zeros(size(pred,1), num_guesses);
pred_bbox = zeros(size(pred,1), num_guesses, 4);

for i=1:5:size(pred,2)
	pred_labels = [ pred_labels, pred(:,i) ];
	for j=1:size(pred,1)
		pred_bbox(j,ceil(i/5),:) = pred(j,i+1:i+4);
	end
end

e = zeros(size(pred_labels,1),1);
for i=1:size(pred,1)				
    filename = gt(i).name;
	rec = VOCreadrecxml(sprintf('%s/%s',gtruth_dir,filename),hash);
	e(i) = 0;
	for k=1:n						%% sum
		for j=1:num_guesses			%% min
			d_jk = (gt_labels(i,k) ~= pred_labels(i,j));
			if d_jk == 0
				ov_vector = compute_overlap(pred_bbox(i,j,:),rec,gt_labels(i,k));	
				f_j = ( ov_vector < 0.50 );
			else
				f_j = 1;
			end
			d_jk = ones(1,numel(f_j)) * d_jk;
			d(i,j) = min( max([f_j;d_jk]) );
		end		
		e(i) = e(i) + min(d(i,:));	%% min over j
	end
end
labels_error = sum(e./n)/size(pred_labels,1);
