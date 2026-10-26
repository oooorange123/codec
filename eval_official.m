function metrics = eval_official(dataset_dir, results_dir)
% Run the pinned VisDrone DET toolkit without changing its matching logic.
% dataset_dir has images/ and annotations/; results_dir has 8-column TXT files.
  root = fileparts(mfilename('fullpath'));
  addpath(fullfile(root, 'vendor', 'VisDrone2018-DET-toolkit', 'utils'));
  if exist('OCTAVE_VERSION', 'builtin')
    pkg load image;
  end
  gt_dir = fullfile(dataset_dir, 'annotations');
  image_dir = fullfile(dataset_dir, 'images');
  names = findImageList(gt_dir);
  [all_gt, all_det] = saveAnnoRes(gt_dir, results_dir, image_dir, numel(names), names);
  [ap, ap50, ap75, ar1, ar10, ar100, ar500] = calcAccuracy(numel(names), all_gt, all_det);
  class_ap = per_class_ap(numel(names), all_gt, all_det);
  metrics = struct('AP', ap, 'AP50', ap50, 'AP75', ap75, ...
                   'AR1', ar1, 'AR10', ar10, 'AR100', ar100, 'AR500', ar500, ...
                   'class_AP_supplemental', class_ap);
  fprintf('Official VisDrone DET (percent): AP=%.6f AP50=%.6f AP75=%.6f\n', ap, ap50, ap75);
end

function per_class = per_class_ap(num_images, all_gt, all_det)
% Supplemental breakdown using the toolkit's evalRes/VOCap, maxDets=500.
  per_class = nan(1, 10);
  for category = 1:10
    AP = zeros(1, 10);
    total_gt = 0;
    for threshold_index = 1:10
      threshold = 0.5 + (threshold_index - 1) * 0.05;
      matches = [];
      num_gt = 0;
      for image_index = 1:num_images
        gt = all_gt{image_index};
        det = all_det{image_index};
        this_gt = gt(gt(:,6) == category, 1:5);
        det = det(1:min(size(det,1),500), :);
        this_det = det(det(:,6) == category, 1:5);
        [gt_out, det_out] = evalRes(this_gt, this_det, threshold);
        num_gt = num_gt + size(gt_out, 1);
        matches = [matches; det_out(:,5:6)];
      end
      total_gt = max(total_gt, num_gt);
      if ~isempty(matches)
        [~, order] = sort(-matches(:,1));
        tp = cumsum(matches(order,2) == 1);
        fp = cumsum(matches(order,2) == 0);
        AP(threshold_index) = VOCap(tp / max(1,num_gt), tp ./ max(1,tp+fp)) * 100;
      end
    end
    if total_gt > 0
      per_class(category) = mean(AP);
    end
  end
end
