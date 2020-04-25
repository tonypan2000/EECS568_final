function local_mapping()

global Map
global State
global Params
global Debug

% KeyFrame insertion
%   Update covisability graph: 
%       add keyframe as a node
%       update edge weights that node is connected
%   Update spanning tree graph
%       Find frame with highest number of matching features
%   Compute bag of words representation using precomputed vocabulary
% Recent MapPoints culling
%   Pass conditions
%   QUESTION: 25% of predicted poses means what?
% New points creation
%   Look for points that are not matched
%   Check that those points satisfy 
%       Epipolar constraints
%       Positive depth
%       Reprojection error
%       Scale consistency
%    Add to Map.mapPoints
% Local BA
%   Wrap matlabs implementation
% Local KeyFrames culling
%   For all keyframes
%       Check if 90% of map points are present in at least 3 other key
%       frames at the same or finer scale

vs = Map.covisibilityGraph;
i=1;
k = 1;
%check_map_point=[];
%check_match_map_point=[];
no_deleted =0;

while(k<=vs.NumViews)
    keyframe1=vs.Views.ViewId(i);
    if ~isempty(find(Params.deletedframes ==keyframe1))
        i = i+1;
        k =k+1;
        continue 
    end
    num_key_frames=vs.NumViews;
    [descriptors_all, points_all] = vs.getAllMapPoints(i);
    map_points_keyframe1=points_all.Count;
    %check_map_point=[check_map_point; [map_points_keyframe1, keyframe1]];
    count=0;
    for j=1:num_key_frames
        keyframe2=vs.Views.ViewId(j);
        if ~isempty(find(Params.deletedframes ==keyframe2))
            continue
        end
        %connections = vs.getAllConnections();
        %[conn_row, conn_col]=size(connections);
        %check_conn=repmat([keyframe1,keyframe2],conn_row,1)
        %Lia = ismember(connections,check_conn,'rows')
        Lia = vs.hasConnection(keyframe1,keyframe2);
        %if(~nnz(Lia))
        if Lia == 0
            continue
        end
        [matches] = vs.getMatches(keyframe1, keyframe2);
        [match_map_points,~]=size(matches);


        %check_match_map_point=[check_match_map_point; [match_map_points,keyframe1,keyframe2]];

       
        if (match_map_points >= Params.cullingThreshold*map_points_keyframe1)
            count=count+1;
        end
        flag_del=0;
        if(count==3)
            vs = vs.deleteKeyFrame(keyframe1);
            Params.deletedframes = [Params.deletedframes keyframe1];
            flag_del=1;
            fprintf('\n [!] Delete frame %d', keyframe1)
            no_deleted = no_deleted +1;
            break;
        end
    end
    
    k= k+1;
    if (flag_del==0)
        i=i+1;
    end
    %vs.NumViews
%     if (k>=vs.NumViews)
%         
%         break;
%     end
end

Map.covisibilityGraph=vs;
no_deleted
end

