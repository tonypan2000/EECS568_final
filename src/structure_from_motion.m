function features_curr = structure_from_motion(frame_prev, frame_curr,k,odometry,nav_method)

global Map
global State
global Params
global Debug

[features_prev, validPoints_prev] = extract_features(frame_prev);
[features_curr, validPoints_curr] = extract_features(frame_curr);

matchedIdx = matchFeatures(features_prev, features_curr, 'Unique', true, ...
    'Method', 'Approximate', 'MatchThreshold', .8);

matchedPoints1 = validPoints_prev(matchedIdx(:, 1));
matchedPoints2 = validPoints_curr(matchedIdx(:, 2));

[relativeOrient, relativeLoc, inlierIdx, status] = estimate_relative_motion(...
    matchedPoints1, matchedPoints2, Params.cameraParams);

bow = calc_bow_repr(features_curr, Params.kdtree, Params.numCodewords);
%k = Map.covisibilityGraph.NumViews + 1;
Map.covisibilityGraph = addView(Map.covisibilityGraph, k, ...
    features_curr, validPoints_curr, ...
    bow, 'Points', validPoints_curr);

pose_km1 = poses(Map.covisibilityGraph, k - 1);
orient_km1 = pose_km1.Orientation{1};
loc_km1 = pose_km1.Location{1};

if nav_method == true
    orientation = eul2rotm([odometry(3) odometry(2) odometry(1)]);
    location = [odometry(7) odometry(8) odometry(9)];

    relativeOrient = orientation*orient_km1';
    relativeLoc =  (location - loc_km1)*orient_km1';
end

if status == 1 && Map.covisibilityGraph.NumViews > 2
    pose_km2 = poses(Map.covisibilityGraph, k - 2);
    orient_km2 = pose_km2.Orientation{1};
    loc_km2 = pose_km2.Location{1};
    
    relativeOrient = orient_km1 * orient_km2';
    relativeLoc = (loc_km1 - loc_km2) * orient_km2';
end



Map.covisibilityGraph = addConnection(Map.covisibilityGraph, k - 1, k, ...
    'Matches', matchedIdx(inlierIdx,:), ...
    'Orientation', relativeOrient, ...
    'Location', relativeLoc);


if nav_method == false
    orientation = relativeOrient * orient_km1;
    location = loc_km1 + relativeLoc * orient_km1;
end


[U, ~, V] = svd(orientation);
orientation = U * V';

Map.covisibilityGraph = updateView(Map.covisibilityGraph, k, ...
    'Orientation', orientation, 'Location', location);

% Connect every past view to the current view
for i = max(k - Params.numViewsToLookBack, 1):k-2
    try
        connect_views(i, k, Params.minMatchesForConnection)
    catch
        % warning('Could not find enough inliers between view %d and %d.', i, k)
    end
end

% local BA
%{
viewIds = max(k - 10, 1):k;
tracks = findTracks(Map.covisibilityGraph, viewIds);

camPoses = poses(Map.covisibilityGraph, viewIds);

xyzPoints = triangulateMultiview(tracks, camPoses, Params.cameraParams);

[xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(xyzPoints, ...
	tracks, camPoses, Params.cameraParams, 'FixedViewId', 1, ...
	'PointsUndistorted', true);

Map.covisibilityGraph = updateView(Map.covisibilityGraph, camPoses);
%}
end

function connect_views(viewIdx1, viewIdx2, minNumMatches)

global Map
global State
global Params
global Debug

if nargin < 3
    minNumMatches = 0;
end

features1 = Map.covisibilityGraph.Descriptors{viewIdx1};
points1 = Map.covisibilityGraph.Points{viewIdx1};
features2 = Map.covisibilityGraph.Descriptors{viewIdx2};
points2 = Map.covisibilityGraph.Points{viewIdx2};

matchedIdx = matchFeatures(features1, features2, 'Unique', true, ...
    'Method', 'Approximate', 'MatchThreshold', .8);

matchedPoints1 = points1(matchedIdx(:, 1));
matchedPoints2 = points2(matchedIdx(:, 2));


[relativeOrient, relativeLoc, inlierIdx, status] = estimate_relative_motion(...
    matchedPoints1, matchedPoints2, Params.cameraParams);


if size(matchedIdx,1) >= minNumMatches && status == 0
    Map.covisibilityGraph = addConnection(Map.covisibilityGraph, ...
        viewIdx1, viewIdx2, ...
        'Matches', matchedIdx(inlierIdx,:), ...
        'Orientation', relativeOrient, ...
        'Location', relativeLoc);
    
end
end
