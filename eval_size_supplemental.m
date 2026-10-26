function metrics = eval_size_supplemental(dataset_dir, results_dir)
% Supplemental original-pixel area AP. Official AP is evaluated separately.
  root = fileparts(mfilename('fullpath'));
  project = fullfile(root, '..', 'visdrone_semantic_v2');
  addpath(fullfile(project, 'vendor', 'VisDrone2018-DET-toolkit', 'utils'));
  if exist('OCTAVE_VERSION', 'builtin'), pkg load image; end
  if nargin == 1 && strcmp(dataset_dir, 'selftest')
    gt = {[1,1,10,10,0,1; 100,100,120,120,0,1]};
    det = {[1,1,10,10,0.9,1; 100,100,120,120,0.8,1]};
    assert(sized_ap(gt,det,0,32^2),100,1e-8);
    assert(sized_ap(gt,det,96^2,inf),100,1e-8);
    metrics = struct('selftest_pass', true);
    return;
  end
  gt_dir = fullfile(dataset_dir, 'annotations');
  names = findImageList(gt_dir);
  [gt, det] = saveAnnoRes(gt_dir, results_dir, fullfile(dataset_dir, 'images'), numel(names), names);
  metrics = struct('AP_small_supplemental', sized_ap(gt, det, 0, 32^2), ...
                   'AP_medium_supplemental', sized_ap(gt, det, 32^2, 96^2), ...
                   'AP_large_supplemental', sized_ap(gt, det, 96^2, inf));
end

function ap = sized_ap(all_gt, all_det, low_area, high_area)
  class_ap = nan(10, 10);
  for cls = 1:10
    for t = 1:10
      threshold = 0.5 + (t - 1) * 0.05;
      matches = zeros(0, 2);
      positives = 0;
      for image_index = 1:numel(all_gt)
        gt = all_gt{image_index};
        det = all_det{image_index};
        current = gt(gt(:,6) == cls, 1:5);
        area = current(:,3) .* current(:,4);
        outside = area < low_area | area >= high_area;
        current(outside, 5) = 1; % Other-size GT absorbs detections as ignore.
        det = det(1:min(size(det,1),500), :);
        [matched_gt, matched_det] = evalRes(current, det(det(:,6) == cls, 1:5), threshold);
        positives = positives + sum(matched_gt(:,5) ~= -1);
        matches = [matches; matched_det(:,5:6)];
      end
      if positives > 0
        [~, order] = sort(-matches(:,1));
        tp = cumsum(matches(order,2) == 1);
        fp = cumsum(matches(order,2) == 0);
        class_ap(cls,t) = 100 * VOCap(tp / positives, tp ./ max(1,tp + fp));
      end
    end
  end
  ap = mean(class_ap(~isnan(class_ap)));
end

