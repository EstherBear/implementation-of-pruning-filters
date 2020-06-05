%this script demos the usage of evaluation routines
% the result file 'demo.val.pred.txt' on validation data is evaluated
% against the ground truth

meta_file = '../data/meta.mat';
pred_file='demo.val.pred.txt'
ground_truth_file='../data/ILSVRC2012_validation_ground_truth.txt'
num_predictions_per_image=5;

load(meta_file);

%%% Task 1
error_flat=zeros(num_predictions_per_image,1);

for i=1:num_predictions_per_image
    error_flat(i) = eval_flat(pred_file,ground_truth_file, i);
end

disp('Task 1: # guesses  vs flat error');
disp([(1:num_predictions_per_image)',error_flat]);

%% Task 2
pred_localization_file='demo.val.pred.det.txt'
ground_truth_file='../data/ILSVRC2012_validation_ground_truth.txt'
num_val_files = -1;

while num_val_files ~= 50000
	if num_val_files ~= -1 
		fprintf('That does not seem to be the correct directory. Please try again\n');
	end
	localization_ground_truth_dir=input('Please enter the path to the Validation bounding box annotations directory: ', 's')
	val_files = dir(sprintf('%s/*.xml',localization_ground_truth_dir));
	num_val_files = numel(val_files);
end

error_localization_flat=zeros(num_predictions_per_image,1);

for i=1:num_predictions_per_image
    error_localization_flat(i) = eval_localization_flat(pred_localization_file,ground_truth_file,localization_ground_truth_dir,meta_file,i);
end

disp('Task 2: # guesses  vs flat error');
disp([(1:num_predictions_per_image)',error_localization_flat]);

