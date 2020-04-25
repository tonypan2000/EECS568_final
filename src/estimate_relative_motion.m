function [orient, loc, inlierIdx, status] = estimate_relative_motion(matchedPoints1, matchedPoints2, cameraParams)
[F, inlierIdx] = estimateFundamentalMatrix(matchedPoints1, matchedPoints2, 'Method', 'RANSAC');

inlierPoints1 = matchedPoints1(inlierIdx, :);
inlierPoints2 = matchedPoints2(inlierIdx, :);

for i = 1:100
    [orient, loc, validPointsFraction] = relativeCameraPose(...
        F, cameraParams, inlierPoints1, inlierPoints2);
    
    if validPointsFraction > 0.8
        status = 0;
        return
    end
end

orient = eye(3);
loc = zeros(1, 3);
status = 1;

end