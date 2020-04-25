function loop_closing()

global Map
global State
global Params
global Debug

i = Map.covisibilityGraph.NumViews;
features1 = Map.covisibilityGraph.Descriptors{i};
points1 = Map.covisibilityGraph.Points{i};
if i > Params.numFramesApart
    % Candidates detection
    bow1 = Map.covisibilityGraph.bow(1:(i - Params.numFramesApart));
    bow2 = Map.covisibilityGraph.bow(i);
    d2 = hist_diff(cell2mat(bow1'), cell2mat(bow2'));
    [~, j] = min(d2);

    features2 = Map.covisibilityGraph.Descriptors{j};
    points2 = Map.covisibilityGraph.Points{j};

    matchedIdx = matchFeatures(features1, features2, 'unique', true);

    ni = length(features1);
    nj = length(features2);
    matchRatio = numel(matchedIdx) / (ni + nj);

    if matchRatio > Params.minMatchRatioRatio
        try
            matchedPoints1 = points1(matchedIdx(:, 1));
            matchedPoints2 = points2(matchedIdx(:, 2));

            [orient, loc, inlierIdx, status] = estimate_relative_motion(...
                matchedPoints1, matchedPoints2, Params.cameraParams);

            if status == 0
                Map.covisibilityGraph = addConnection(Map.covisibilityGraph, ...
                    i, j, ...
                    'Matches', matchedIdx(inlierIdx, :), ...
                    'Orientation', orient, ...
                    'Location', loc);

                fprintf('[!] Found a loop closure between %d and %d.\n', i, j);
            end
        catch
            fprintf('[!] Loop closure proposal (%d, %d) rejected.\n', i, j);
        end
    end

end

% Compute Sim3
% Loop fusion
% Optimize essential graph

end

%%
function d2 = hist_diff(h1, h2)
d2 = sum((h1 - h2).^2 ./ (h1 + h2 + 1e-6), 2);
end
