function surf_slam(frame_curr,k,liekf_results,nav_method)

global Map
global State
global Params
global Debug

persistent frame_prev
if ~isempty(frame_prev)
    

    
    structure_from_motion(frame_prev, frame_curr,k,liekf_results,nav_method);

    % disp('Before keyframe culling')
    % disp(Map.covisibilityGraph.NumViews);
% 	if mod(k, Params.cullingSkip) == 0
%         local_mapping();
%     end
    % disp('After keyframe culling')
    % disp(Map.covisibilityGraph.NumViews);
	loop_closing();
else
	% Initialize
	[descriptors, points] = extract_features(frame_curr);

    bow = calc_bow_repr(descriptors, Params.kdtree, Params.numCodewords);

	Map.covisibilityGraph = addView(Map.covisibilityGraph, 1,...
        descriptors, points, bow, 'Points', points, ...
		'Orientation', eye(3), 'Location', zeros(1, 3));

end

	frame_prev = frame_curr;
end
