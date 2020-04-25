% SEE Line 746 in ourViewSet.m in src/structs/


% SETUP
clear all
keyFrame1 = 3; 
keyFrame2 = 6;
keyFrame_delete = 2;
load saved_map


% For brevity
vs = Map.covisibilityGraph;


% For convenience
vs = vs.deleteKeyFrame(keyFrame_delete)
matches = vs.getMatches(keyFrame1, keyFrame2)
connections = vs.getAllConnections()


% Get n vector for each feature in an image, where n the opposite of the
% average viewing angle from all views where the feature is observed
ns = vs.getNForAllFeaturesInKeyFrame(keyFrame1);
a_particular_n = ns{1}


% Access MapPoints
[descriptors_all, points_all] = vs.getAllMapPoints(keyFrame1)
[descriptors_noMatch,  points_noMatch, idx_noMatch]   = vs.getMapPointsWithNoMatches(keyFrame1)
[descriptors_mt1Match, points_mt1Match, idx_mt1Match] = vs.getMapPointsWithGreaterThanOrEqualToNMatches(keyFrame1, 1)
[descriptors_2matches, points_2matches, idx_2matches] = vs.getMapPointsWithExactlyNMatches(keyFrame1, 2)
% More general
% [descriptors, points, idx] = vs.getMapPointsWithMatches(keyFrame1, 1, @gt);


% Access spanning tree
[spanningTree, connections_spantree, views_spantree] = vs.getSpanningTree()