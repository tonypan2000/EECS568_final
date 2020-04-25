% This is the same as Matlab's viewset with the addition of tracking
% descriptors.  




% viewSet Object for managing data for structure-from-motion and visual odometry
%   Use this object to store view attributes, such as feature points and 
%   absolute camera poses, and pairwise connections between views, such as 
%   point matches and relative camera poses. Also use this object to find 
%   point tracks used by triangulateMultiview and bundleAdjustment 
%   functions.
%
%   vSet = viewSet() returns an empty viewSet object
%
%   viewSet properties:
%
%   NumViews    - the number of views (read only)   
%   Views       - a table containing view attributes (read only)
%   Connections - a table containing pairwise relationships between views (read only)
%
%   viewSet methods:
%
%   addView          - add a new view
%   updateView       - modify an existing view
%   deleteView       - delete an existing view
%   hasView          - check if a view exists
%   addConnection    - add a new connection between a pair of views
%   updateConnection - modify an existing connection
%   deleteConnection - delete and existing connection
%   hasConnection    - check if a connection between two views exists
%   findTracks       - find matched points across multiple views
%   poses            - return a table containing camera poses
%
%   Example 1: Find point tracks across a sequence of images
%   ----------------------------------------------------------
%   % Load images
%   imageDir = fullfile(toolboxdir('vision'), 'visiondata', ...
%    'structureFromMotion');
%   images = imageDatastore(imageDir);
%
%   % Compute features for the first image
%   I = rgb2gray(readimage(images, 1));
%   pointsPrev = detectSURFFeatures(I);
%   [featuresPrev, pointsPrev] = extractFeatures(I, pointsPrev);
%
%   % Create a viewSet object
%   vSet = viewSet;
%   vSet = addView(vSet, 1, 'Points', pointsPrev);
%
%   % Compute features and matches for the rest of the images
%   for i = 2:numel(images.Files)
%     I = rgb2gray(readimage(images, i));
%     points = detectSURFFeatures(I);
%     [features, points] = extractFeatures(I, points);
%     vSet = addView(vSet, i, 'Points', points);
%     pairsIdx = matchFeatures(featuresPrev, features);
%     vSet = addConnection(vSet, i-1, i, 'Matches', pairsIdx);
%     featuresPrev = features;
%   end
%
%   % Find point tracks
%   tracks = findTracks(vSet);
%
%   Example 2: Structure from motion from multiple views
%   ----------------------------------------------------
%   % This example shows you how to estimate the poses of a calibrated 
%   % camera from a sequence of views, and reconstruct the 3-D structure of
%   % the scene up to an unknown scale factor.
%   % <a href="matlab:web(fullfile(matlabroot,'toolbox','vision','visiondemos','html','StructureFromMotionFromMultipleViewsExample.html'))">View example</a>
%
%   See also detectSURFFeatures, detectHarrisFeatures, detectMinEigenFeatures,
%     detectFASTFeatures, detectBRISKFeatures, detectMSERFeatures,
%     pointTrack, matchFeatures, bundleAdjustment, triangulateMultiview,
%     table

classdef ourViewSet
    properties(SetAccess=private)
        % Views A table containing view attributes (read only)
        %   The table contains the following columns: 'ViewId', 
        %   'Points', 'Orientation', and 'Location'.
        Views = table();
        
        % Connections A table containing connections between views (read only) 
        %   The table contains the following columns: 'ViewId1', 'ViewId2',
        %   'Matches', 'RelativeOrientation', and 'RelativeLocation'.
        Connections = table();
		
		% ADDITION
		Points = {};     % Added because Matlab typecasts view.points
		Descriptors = {};
        Orientation = {};
        Location = {};
        bow = {};
% 		MapPoints = containers.Map();
% 		KeyFrames = containers.Map();
    end
    
    properties(SetAccess=private, Dependent)
        % NumViews The number of views in the view set
        NumViews;
    end
    
    properties(GetAccess=private, SetAccess=private)
        FeatureGraph;
        ShouldRecreateGraph = false;
    end
    
    methods
        %------------------------------------------------------------------
        function numViews = get.NumViews(this)
            numViews = height(this.Views);
        end
                
        %------------------------------------------------------------------
        function this = addView(this, view, descriptors, points_, bow_, varargin)
            % addView Add a new view to a viewSet object
            %   vSet = addView(vSet, viewId, Name, Value, ...) adds a new view
            %   denoted by viewId, an integer, to a viewSet object.
            %
            %   View attributes can be specified using additional name-value pairs
            %   described below:
            %
            %   'Points'       image points, specified as an M-by-2 matrix of [x,y]
            %                  coordinates, or an object of any <a href="matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'pointfeaturetypes')">point feature type</a>
            %
            %   'Orientation'  3-by-3 matrix containing absolute camera orientation
            %
            %   'Location'     3-element vector containing absolute camera location
            %
            %   vSet = addView(vSet, views) adds a new view or a set of views
            %   specified as a table. The table must contain a column named 'ViewId',
            %   and one or more columns named 'Points', 'Orientation',
            %   'Location'.
            %
            %   Example
            %   -------
            %   % Create an empty viewSet object
            %   vSet = viewSet;
            %
            %   % Detect interest points in the image
            %   imageDir = fullfile(toolboxdir('vision'), 'visiondata', ...
            %      'structureFromMotion');
            %   I = imread(fullfile(imageDir, 'image1.jpg'));
            %   points = detectSURFFeatures(rgb2gray(I));
            %
            %   % Add the points to the object
            %   vSet = addView(vSet, 1, 'Points', points, 'Orientation', eye(3), ...
            %       'Location', [0,0,0]);
            %
            %   See also detectSURFFeatures, detectHarrisFeatures, detectMinEigenFeatures,
            %     detectFASTFeatures, detectBRISKFeatures, detectMSERFeatures, table
            if istable(view)
                view = checkViewTable(view);
			else
                [ViewId, Points, Orientation, Location] = ...
                    parseViewInputs(view, varargin{:});
                view = table(ViewId, Points, Orientation, Location);
            end
            
            % Check if any of new views already exist
            ids = view.ViewId;
            viewIdx = hasView(this, ids);
            if any(viewIdx)
                existingViews = ids(viewIdx);
                error(message('vision:viewSet:viewIdAlreadyExists', existingViews(1)));
            end
            
            if isempty(this.Views)
                this.Views = view;
            else
                this.Views = [this.Views; view];
            end
            
            if ~this.ShouldRecreateGraph
                if isempty(this.FeatureGraph)
                    this.FeatureGraph = graph;
                end
                
                for i = 1:height(view)
                    if ~ismember('Points', view.Properties.VariableNames)
                        continue;
                    end
                    points = view.Points{i};
                    if isempty(points)
                        continue;
                    end
                    numPoints = size(points, 1);
                    viewId = repmat(view.ViewId(i), [numPoints, 1]);
                    pointIdx = (1:numPoints)';
                    this.FeatureGraph = addnode(this.FeatureGraph, table(viewId, pointIdx, points));
                end
			end
			
			% ADDITION
			% Update descriptors, keyframes and mappoints
			this.Descriptors{end+1} = descriptors;
			this.Points{end+1} = points_;
            this.bow{end+1} = bow_;
        end
        
        %------------------------------------------------------------------
        function this = updateView(this, view, varargin)
            % updateView Modify an existing view in a viewSet object
            %   vSet = updateView(vSet, viewId, Name, Value, ...) modifies an
            %   existing view denoted by viewId, and integer.
            %
            %   View attributes can be specified using additional name-value pairs
            %   described below:
            %
            %   'Points'       image points, specified as an M-by-2 matrix of [x,y]
            %                  coordinates, or an object of any <a href="matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'pointfeaturetypes')">point feature type</a>
            %
            %   'Orientation'  3-by-3 matrix containing absolute camera orientation
            %
            %   'Location'     3-element vector containing absolute camera location
            %
            %   vSet = updateView(vSet, views) modifies an existing view or a set of
            %   views specified as a table. The table must contain a column named
            %   'ViewId', and one or more columns named 'Points', 'Orientation',
            %   'Location'.
            %
            %   Example
            %   -------
            %   % Create an empty viewSet object
            %   vSet = viewSet;
            %
            %   % Detect interest points in the image
            %   imageDir = fullfile(toolboxdir('vision'), 'visiondata', ...
            %      'structureFromMotion');
            %   I = imread(fullfile(imageDir, 'image1.jpg'));
            %   points = detectSURFFeatures(rgb2gray(I));
            %
            %   % Add the points to the object
            %   vSet = addView(vSet, 1, 'Points', points);
            %
            %   % Specify the camera pose
            %   vSet = updateView(vSet, 1, 'Orientation', eye(3), ...
            %       'Location', [0,0,0]);
            %
            %   See also detectSURFFeatures, detectHarrisFeatures, detectMinEigenFeatures,
            %     detectFASTFeatures, detectBRISKFeatures, detectMSERFeatures,
            %     table
            if istable(view)
                view = checkViewTable(view);
                checkIfViewIsMissing(this, view.ViewId);
                
                % check if there are new columns
                newCols = setdiff(view.Properties.VariableNames, ...
                    this.Views.Properties.VariableNames);
                if ~isempty(newCols)
                    this.Views{:, newCols} = cell(height(this.Views), numel(newCols));
                end
                
                viewIds1 = this.Views.ViewId;
                viewIds2 = view.ViewId;
                [viewIdsToUpdate, ia] = intersect(viewIds1, viewIds2);
                this.Views(ia, view.Properties.VariableNames) = view;
                
                % If the points of the views are modified, delete matches
                % in all the view's connections.
                if ismember('Points', view.Properties.VariableNames)
                    this = wipeMatches(this, viewIdsToUpdate);
                    this.ShouldRecreateGraph = true;
                end
            else
                [ViewId, Points, Orientation, Location, unsetColumns] = ...
                    parseViewInputs(view, varargin{:});
                checkIfViewIsMissing(this, ViewId);
                
                % ViewId must be a scalar. Limit find() to stop after
                % finding the first instance.
                idx = find(this.Views.ViewId == ViewId, 1, 'first');
                
                if ~ismember('Points', unsetColumns)
                    this.Views{idx, 'Points'} = Points;
                    this = wipeMatches(this, ViewId);
                    this.ShouldRecreateGraph = true;
                end
                
                if ~ismember('Orientation', unsetColumns)
                    this.Views{idx, 'Orientation'} = Orientation;
                end
                
                if ~ismember('Location', unsetColumns)
                    this.Views{idx, 'Location'} = Location;
                end
                
            end
        end
        
        %------------------------------------------------------------------
        function this = deleteView(this, viewId)
        % deleteView Delete an existing view from a viewSet object
        %   vSet = deleteView(vSet, viewId) deletes an existing view or a
        %   set of views. viewId is either an integer or a vector
        %   containing the ids of the views to be deleted.
        %
        %   Example
        %   -------
        %   % Create an empty viewSet object
        %   vSet = viewSet;
        %
        %   % Detect interest points in the image
        %   imageDir = fullfile(toolboxdir('vision'), 'visiondata', ...
        %      'structureFromMotion');
        %   I = imread(fullfile(imageDir, 'image1.jpg'));
        %   points = detectSURFFeatures(rgb2gray(I));
        %
        %   % Add a new view
        %   vSet = addView(vSet, 1, 'Points', points);
        %
        %   % Delete the view
        %   vSet = deleteView(vSet, 1);
        %
        %   See also detectSURFFeatures, detectHarrisFeatures, detectMinEigenFeatures,
        %     detectFASTFeatures, detectBRISKFeatures, detectMSERFeatures, table                        
            viewIds = checkViewIds(viewId);
                        
            for viewId = viewIds'                
                checkIfViewIsMissing(this, viewId);
                
                viewIdx = getViewIndex(this, viewId);
                if ~isempty(this.Views.Points{viewIdx})
                    this.ShouldRecreateGraph = true;
                end
                this.Views(viewIdx, :) = [];
				
				% ADDITION
				this.Descriptors(viewIdx) = [];
				this.Points(viewIdx) = [];
				% Update map points by removing the index of the keyframe
				% being removed
				% If the mappoint no longer has the index of any keyframes,
				% remove the mappoint
% 				strView = num2str(viewId);
% 				mapPoints = this.KeyFrames(strView);
% 				for i = 1:length(mapPoints)
% 					mpKey = num2str(mapPoints(i));
% 					frameArray = this.MapPoints(mpKey);
% 					frameArray(frameArray == viewId) = [];
% 					this.MapPoints(mpKey) = frameArray;
% 					if isempty(this.MapPoints(mpKey))
% 						this.MapPoints.remove(mpKey)
% 					end
% 				end
% 				% Remove the keyframe
% 				this.KeyFrames.remove(strView);
                
                if ~isempty(this.Connections)
                    connIdx = getConnectionIndexToAndFrom(this, viewId);
                    this.Connections(connIdx, :) = [];
                end
            end
        end
        
        %------------------------------------------------------------------
        function tf = hasView(this, viewId)
        % hasView Check if a view with a given viewId exists
        %   tf = hasView(vSet, viewId) returns true if the view with the
        %   given viewId exists in the view set, and false otherwise.
        %   viewId can be a scalar or a vector of view ids. If viewId is a
        %   vector, then tf is a logical vector of the same length.
        %
        %   Example
        %   -------
        %   % Create an empty viewSet object
        %   vSet = viewSet;
        %
        %   % Detect interest points in the image
        %   imageDir = fullfile(toolboxdir('vision'), 'visiondata', ...
        %      'structureFromMotion');
        %   I = imread(fullfile(imageDir, 'image1.jpg'));
        %   points = detectSURFFeatures(rgb2gray(I));
        %
        %   % Add a new view
        %   vSet = addView(vSet, 1, 'Points', points);
        %
        %   % Check if view with id 1 exists
        %   tf = hasView(vSet, 1);
        %
        %   See also viewSet
            viewId = checkViewIds(viewId);
            if isempty(this.Views)
                tf = false;
            else
                ids = this.Views.ViewId;
                tf = ismember(viewId, ids);
            end
        end
        
        %------------------------------------------------------------------
        function tf = hasConnection(this, viewId1, viewId2)
        % hasConnection Check if a connection between two views exists
        %   tf = hasConnection(vSet, viewId1, viewId2) returns true if 
        %   both views exist and there is a connection between them. 
        %   Otherwise, the method returns false.
        %
        %   Example
        %   -------
        %   % Create an empty viewSet object
        %   vSet = viewSet;
        %
        %   % Add a pair of views
        %   vSet = addView(vSet, 1);
        %   vSet = addView(vSet, 2);
        %
        %   % Add a connection
        %   vSet = addConnection(vSet, 1, 2);
        %   
        %   % Check if the connection exists
        %   tf = hasConnection(vSet, 1, 2);
        %
        %   See also viewSet
            viewId1 = checkViewId(viewId1);
            viewId2 = checkViewId(viewId2);
            if hasView(this, viewId1) && hasView(this, viewId2)
                idx = getConnectionIndex(this, viewId1, viewId2);
                tf = any(idx);
            else
                tf = false;
            end
        end
        
        %------------------------------------------------------------------
        function this = addConnection(this, viewId1, viewId2, varargin)
        % addConnection Add a connection between two views in a viewSet object
        %   vSet = addConnection(vSet, viewId1, viewId2, Name, Value, ...)
        %   adds a connection between two views denoted by integer scalars
        %   viewId1 and viewId2.
        %
        %   Connection attributes can be specified using additional name-value
        %   pairs described below:
        %
        %   'Matches'      M-by-2 matrix containing indices of matched
        %                  points between the two views
        %
        %   'Orientation'  3-by-3 matrix containing orientation of the second
        %                  camera relative to the first
        %
        %   'Location'     3-element vector containing location of the second
        %                  camera relative to the first
        %
        %   Example
        %   -------
        %   % Create an empty viewSet object
        %   vSet = viewSet;
        %
        %   % Read a pair of images
        %   imageDir = fullfile(toolboxdir('vision'), 'visiondata', ...
        %     'structureFromMotion');
        %   I1 = rgb2gray(imread(fullfile(imageDir, 'image1.jpg')));
        %   I2 = rgb2gray(imread(fullfile(imageDir, 'image2.jpg')));
        %
        %   % Detect interest points in the two images
        %   points1 = detectSURFFeatures(I1);
        %   points2 = detectSURFFeatures(I2);
        %
        %   % Add the points to the viewSet object
        %   vSet = addView(vSet, 1, 'Points', points1);
        %   vSet = addView(vSet, 2, 'Points', points2);
        %
        %   % Extract feature descriptors
        %   features1 = extractFeatures(I1, points1);
        %   features2 = extractFeatures(I2, points2);
        %
        %   % Match features and store the matches
        %   indexPairs = matchFeatures(features1, features2);
        %   vSet = addConnection(vSet, 1, 2, 'Matches', indexPairs);
        %
        %   See also detectSURFFeatures, detectHarrisFeatures, detectMinEigenFeatures,
        %     detectFASTFeatures, detectMSERFeatures, detectBRISKFeatures, matchFeatures,
        %    bundleAdjustment, triangulateMultiview
            [ViewId1, ViewId2, Matches, RelativeOrientation, RelativeLocation] = ...
                parseConnectionInputs(this, viewId1, viewId2, varargin{:});
            
            if hasConnection(this, ViewId1, ViewId2)
                error(message('vision:viewSet:connectionAlreadyExists', ...
                    ViewId1, ViewId2));
            end
            
            conn = table(ViewId1, ViewId2, Matches, RelativeOrientation, ...
                RelativeLocation);
            
            if isempty(this.Connections)
                this.Connections = conn;
            else
                this.Connections = [this.Connections; conn];
            end
            
            if ~this.ShouldRecreateGraph
                matches = Matches{1};
                viewId1 = ViewId1;
                viewId2 = ViewId2;
                if ~isempty(matches)
                    offsetS = find(this.FeatureGraph.Nodes.viewId == viewId1, 1, 'first') - 1;
                    s = matches(:, 1) + offsetS;
                    
                    offsetT = find(this.FeatureGraph.Nodes.viewId == viewId2, 1, 'first') - 1;
                    t = matches(:, 2) + offsetT;
                    this.FeatureGraph = addedge(this.FeatureGraph, s, t);
                end
            end
        end
                
        %------------------------------------------------------------------
        function this = updateConnection(this, viewId1, viewId2, varargin)
        % updateConnection Modify a connection between two views in a viewSet object
        %   vSet = updateConnection(vSet, viewId1, viewId2, Name, Value, ...)
        %   modifies a connection between two views denoted by integer scalars
        %   viewId1 and viewId2.
        %
        %   Connection attributes to be modified can be specified using additional
        %   name-value pairs described below:
        %
        %   'Matches'      M-by-2 matrix containing indices of matched 
        %                  points between the two views
        %
        %   'Orientation'  3-by-3 matrix containing orientation of the second
        %                  camera relative to the first
        %
        %   'Location'     3-element vector containing location of the second
        %                  camera relative to the first
        %
        %   Example
        %   -------
        %   % Create an empty viewSet object
        %   vSet = viewSet;
        %
        %   % Read a pair of images
        %   imageDir = fullfile(toolboxdir('vision'), 'visiondata', ...
        %     'structureFromMotion');
        %   I1 = rgb2gray(imread(fullfile(imageDir, 'image1.jpg')));
        %   I2 = rgb2gray(imread(fullfile(imageDir, 'image2.jpg')));
        %
        %   % Detect interest points in the two images
        %   points1 = detectSURFFeatures(I1);
        %   points2 = detectSURFFeatures(I2);
        %
        %   % Add the points to the viewSet object
        %   vSet = addView(vSet, 1, 'Points', points1);
        %   vSet = addView(vSet, 2, 'Points', points2);
        %
        %   % Extract feature descriptors
        %   features1 = extractFeatures(I1, points1);
        %   features2 = extractFeatures(I2, points2);
        %
        %   % Match features and store the matches
        %   indexPairs = matchFeatures(features1, features2);
        %   vSet = addConnection(vSet, 1, 2, 'Matches', indexPairs);
        %
        %   % Store a relative pose between the views
        %   vSet = updateConnection(vSet, 1, 2, 'Orientation', eye(3), ...
        %     'Location', [1 0 0]);
        %
        %   See also detectSURFFeatures, detectHarrisFeatures, detectMinEigenFeatures,
        %     detectFASTFeatures, detectMSERFeatures, detectBRISKFeatures, matchFeatures,
        %     bundleAdjustment, triangulateMultiview       
        
            [viewId1, viewId2, matches, orientation, location, unsetColumns] = ...
                parseConnectionInputs(this, viewId1, viewId2, varargin{:});
            
            checkIfConnectionIsMissing(this, viewId1, viewId2);
            
            idx = getConnectionIndex(this, viewId1, viewId2);
            
            if ~ismember('Matches', unsetColumns)
                this.Connections{idx, 'Matches'} = matches;
                this.ShouldRecreateGraph = true;
            end
            
            if ~ismember('Orientation', unsetColumns)
                this.Connections{idx, 'RelativeOrientation'} = orientation;
            end
            
            if ~ismember('Location', unsetColumns)
                this.Connections{idx, 'RelativeLocation'} = location;
            end            
                   
        end
                
        %------------------------------------------------------------------
        function this = deleteConnection(this, viewId1, viewId2)
        % deleteConnection Delete a connection between two views in a viewSet object
        %   vSet = deleteConnection(vSet, viewId1, viewId2) deletes a connection
        %   between two views denoted by integer scalars viewId1 and viewId2.
        %
        %   Example
        %   -------
        %   % Create an empty viewSet object
        %   vSet = viewSet;
        %
        %   % Read a pair of images
        %   imageDir = fullfile(toolboxdir('vision'), 'visiondata', ...
        %     'structureFromMotion');
        %   I1 = rgb2gray(imread(fullfile(imageDir, 'image1.jpg')));
        %   I2 = rgb2gray(imread(fullfile(imageDir, 'image2.jpg')));
        %
        %   % Detect interest points in the two images
        %   points1 = detectSURFFeatures(I1);
        %   points2 = detectSURFFeatures(I2);
        %
        %   % Add the points to the viewSet object
        %   vSet = addView(vSet, 1, 'Points', points1);
        %   vSet = addView(vSet, 2, 'Points', points2);
        %
        %   % Extract feature descriptors
        %   features1 = extractFeatures(I1, points1);
        %   features2 = extractFeatures(I2, points2);
        %
        %   % Match features and store the matches
        %   indexPairs = matchFeatures(features1, features2);
        %   vSet = addConnection(vSet, 1, 2, 'Matches', indexPairs);
        %
        %   % Delete the connection between the views
        %   vSet = deleteConnection(vSet, 1, 2);
        %
        %   See also detectSURFFeatures, detectHarrisFeatures, detectMinEigenFeatures,
        %     detectFASTFeatures, detectMSERFeatures, detectBRISKFeatures, matchFeatures,
        %     bundleAdjustment, triangulateMultiview        
            checkIfConnectionIsMissing(this, viewId1, viewId2);
            idx = getConnectionIndex(this, viewId1, viewId2);
            if ~isempty(this.Connections{idx, 'Matches'})                     
                this.ShouldRecreateGraph = true;
            end
            this.Connections(idx, :) = [];
        end
        
        %------------------------------------------------------------------
        function camPoses = poses(this, viewIds)
        % poses returns camera poses associated with views
        %   camPoses = poses(vSet) returns the camera poses
        %   associated with the views contained in the viewSet.
        %   camPoses is a table containing three columns: 'ViewId',
        %   'Orientation' and 'Location'.
        %
        %   camPoses = poses(vSet, viewIds) returns the camera
        %   poses associated with a subset of views specified by viewIds,
        %   a vector of integers.
        %
        %   Example
        %   -------
        %   % Create an empty viewSet object
        %   vSet = viewSet;
        %
        %   % Add views
        %   vSet = addView(vSet, 1, 'Orientation', eye(3), ...
        %       'Location', [0,0,0]);
        %   vSet = addView(vSet, 2, 'Orientation', eye(3), ...
        %       'Location', [1,0,0]);
        %
        %   % Retrieve and display the camera poses
        %   camPoses = poses(vSet);
        %   plotCamera(camPoses, 'Size', 0.2);
        %
        %   See also triangulateMultiview, bundleAdjustment, table
            if nargin > 1
                viewIds = checkViewIds(viewIds);
            end
            
            camPosesColumnNames = {'ViewId', 'Orientation', ...
                    'Location'};
            if isempty(this.Views)
                camPoses = table();
                return;
            end
            
            if nargin < 2
                camPoses = this.Views(:, camPosesColumnNames);
            else                
                ids = this.Views.ViewId;
                viewIds = viewIds(:);
                [~, idx] = intersect(ids, viewIds);
                camPoses = this.Views(idx, camPosesColumnNames);
            end
        end
        
        %------------------------------------------------------------------
        function tracks = findTracks(this, viewIds)
        % findTracks find matched points across multiple views
        %   tracks = findTracks(vSet) finds point tracks across
        %   multiple views. Each track contains 2-D projections of the
        %   same 3-D world point. tracks is an array of pointTrack objects
        %
        %   tracks = findTracks(vSet, viewIds) finds point tracks
        %   across a subset of views, viewIds, specified as a vector
        %   of integers.
        %
        %   Example - Find point tracks across a sequence of images
        %   -----------------------------------------------------------
        %   % Load images
        %   imageDir = fullfile(toolboxdir('vision'), 'visiondata', ...
        %    'structureFromMotion');
        %   images = imageDatastore(imageDir);
        %
        %   % Compute features for the first image
        %   I = rgb2gray(readimage(images, 1));
        %   pointsPrev = detectSURFFeatures(I);
        %   [featuresPrev, pointsPrev] = extractFeatures(I, pointsPrev);
        %
        %   % Create a viewSet object
        %   vSet = viewSet;
        %   vSet = addView(vSet, 1, 'Points', pointsPrev);
        %
        %   % Compute features and matches for the rest of the images
        %   for i = 2:numel(images.Files)
        %    I = rgb2gray(readimage(images, i));
        %    points = detectSURFFeatures(I);
        %    [features, points] = extractFeatures(I, points);
        %    vSet = addView(vSet, i, 'Points', points);
        %    pairsIdx = matchFeatures(featuresPrev, features);
        %    vSet = addConnection(vSet, i-1, i, 'Matches', pairsIdx);
        %    featuresPrev = features;
        %  end
        %
        %  % Find point tracks
        %  tracks = findTracks(vSet);
        %
        %  See also detectSURFFeatures, detectHarrisFeatures, detectMinEigenFeatures,
        %    detectFASTFeatures, detectBRISKFeatures, detectMSERFeatures,
        %    pointTrack, matchFeatures, bundleAdjustment, triangulateMultiview            
            if nargin > 1
                viewIds = checkViewIds(viewIds);      
            end
            
            if isempty(this.Connections)
            tracks = pointTrack.empty();    
                return;
            end
            
            if this.ShouldRecreateGraph
                this.FeatureGraph = addFeatureNodes(this);
                this.FeatureGraph = addFeatureEdges(this);                
            end
                        
            if nargin > 1
                subgraphIdx = false(numnodes(this.FeatureGraph), 1);
                for i = viewIds'
                    subgraphIdx = subgraphIdx | (this.FeatureGraph.Nodes.viewId == i);
                end
                % This is ok, because viewSet is a value object. 
                % The change to FeatureGraph will not be visible to the
                % caller.
                this.FeatureGraph = subgraph(this.FeatureGraph, find(subgraphIdx));
            end   
            
            tracks = createTracks(this);
        end
	end
	
	methods % ADDITION OUR METHODS
		
		% An alias for deleteView
		function this = deleteKeyFrame(this, idx)
			this = this.deleteView(idx);
		end
		
		function connections = getConnectionsWithGreaterThanOrEqualToNMatches(this, N)
			connections = getAllConnections(this);
			for i = size(connections,1):-1:1
				idx1 = connections(i,1);
				idx2 = connections(i,2);
				connectionIdx = this.getConnectionIndex(idx1, idx2);
				if ~(size(this.Connections.Matches{connectionIdx},1) >= N)
					connections(i,:) = [];
				end
			end
		end
		
		function connections = getConnectionsWithLessThanNMatches(this, N)
			connections = getAllConnections(this);
			for i = size(connections,1):-1:1
				idx1 = connections(i,1);
				idx2 = connections(i,2);
				connectionIdx = this.getConnectionIndex(idx1, idx2);
				if ~(size(this.Connections.Matches{connectionIdx},1) < N)
					connections(i,:) = [];
				end
			end
		end
		
		% Not sure if spanning tree is in the right form
		function [spanningTree, connections, views] = getSpanningTree(this)
			views = this.Views.ViewId;
			connections = zeros(length(views)-1, 2);
			for i = 1:length(views)-1
				connections(i,1) = views(i);
				connections(i,2) = views(i+1);
			end
			spanningTree = zeros(length(connections));
			for i = 1:length(connections)
				idx1 = connections(i,1);
				idx2 = connections(i,2);
				spanningTree(idx1, idx2) = 1;
			end
		end
		
		function matches = getMatches(this, viewIdx1, viewIdx2)
			connectionIdx = this.getConnectionIndex(viewIdx1, viewIdx2);
			matches = this.Connections.Matches{connectionIdx};
		end
		
		function connections = getAllConnections(this)
			connections = nchoosek(this.Views.ViewId, 2);
			for i = size(connections,1):-1:1
				idx1 = connections(i,1);
				idx2 = connections(i,2);
				% If no connection exists delete from potential connections
				if ~this.hasConnection(idx1, idx2)
					connections(i,:) = [];
				end
			end
		end
		
		function [descriptors, points] = getAllMapPoints(this, viewIdx)
			descriptors = this.Descriptors{viewIdx};
			points = this.Points{viewIdx};
		end
		
		function [descriptors, points, idx] = getMapPointsWithMatches(this, viewIdx, numMatches, comparrison)
			if nargin < 4
				comparrison = @ge;
			end
			if nargin < 3
				numMatches = 1;
			end
			
			featuresWithMatches = getViewsThatHaveMatchingFeatures(this, viewIdx);
			checkLength = @(v) comparrison(length(v), numMatches);
			
			satisfyCheckLength = cellfun(checkLength, featuresWithMatches);
			[descriptors_all, points_all] = getAllMapPoints(this, viewIdx);
			descriptors = descriptors_all(satisfyCheckLength,:);
			points = points_all(satisfyCheckLength);
			idx = find(satisfyCheckLength);
		end
		
		function [descriptors, points, idx] = getMapPointsWithGreaterThanOrEqualToNMatches(this, viewIdx, N)
			[descriptors, points, idx] = getMapPointsWithMatches(this, viewIdx, N, @gt);
		end
		
		function [descriptors, points, idx] = getMapPointsWithExactlyNMatches(this, viewIdx, N)
			[descriptors, points, idx] = getMapPointsWithMatches(this, viewIdx, N, @eq);
		end
		
		function [descriptors, points, idx] = getMapPointsWithNoMatches(this, viewIdx)
			[descriptors, points, idx] = getMapPointsWithMatches(this, viewIdx, 0, @eq);
		end
		
		function n = getNForAllFeaturesInKeyFrame(this, viewIdx)
			matches = this.getViewsThatHaveMatchingFeatures(viewIdx);
			n = cell([1 length(matches)]);
			n_sum = cell([1 length(matches)]);
			n_sum(1,:) = {-this.Views.Orientation{viewIdx}(3,:)};
			for i = 1:length(matches)
				match = matches{i};
				for j = 1:length(match)
					n_sum{i} = n_sum{i} - this.Views.Orientation{j}(3,:);
				end
				n{i} = n_sum{i}./norm(n_sum{i}); % +1 accounts for viewIdx frame
			end
		end
		
		function matches = getViewsThatHaveMatchingFeatures(this, viewIdx)
			matches = cell([1 length(this.Points{viewIdx})]);
			otherViewIdxs = unique([this.Connections.ViewId1; this.Connections.ViewId2]);
			otherViewIdxs(viewIdx) = [];
			for i = 1:length(otherViewIdxs)
				otherViewIdx = otherViewIdxs(i);
				if this.hasConnection(viewIdx, otherViewIdx)
					connectionIdx = getConnectionIndex(this, viewIdx, otherViewIdx);
					matchList = this.Connections.Matches{connectionIdx}(:,1);
				elseif this.hasConnection(otherViewIdx, viewIdx)
					connectionIdx = getConnectionIndex(this, otherViewIdx, viewIdx);
					matchList = this.Connections.Matches{connectionIdx}(:,2);
				else
					continue
				end
				
				for j = 1:length(matchList)
					match = matchList(j);
					matches{match} = [matches{match} otherViewIdx];
				end
			end
		end
	end
                
    methods(Access=private)
        %------------------------------------------------------------------
        function featureGraph = addFeatureNodes(this)
            featureGraph = graph;
            for i = 1:height(this.Views)
                points = this.Views.Points{i};
                if isempty(points)
                    continue;
                end
                numPoints = size(points, 1);
                viewId = repmat(this.Views.ViewId(i), [numPoints, 1]);
                pointIdx = (1:numPoints)';
                featureGraph = addnode(featureGraph, table(viewId, pointIdx, points));
            end
        end
        
        %------------------------------------------------------------------
        function featureGraph = addFeatureEdges(this)
            for i = 1:height(this.Connections)
                viewId1 = this.Connections.ViewId1(i);
                viewId2 = this.Connections.ViewId2(i);
                matches = this.Connections.Matches{i};
                
                if isempty(matches)
                    continue;
                end
                
                offsetS = find(this.FeatureGraph.Nodes.viewId == viewId1, 1, 'first') - 1;
                s = matches(:, 1) + offsetS;
                
                offsetT = find(this.FeatureGraph.Nodes.viewId == viewId2, 1, 'first') - 1;
                t = matches(:, 2) + offsetT;
                this.FeatureGraph = addedge(this.FeatureGraph, s, t);
            end
            featureGraph = this.FeatureGraph;
        end
            
        %------------------------------------------------------------------
        function tracks = createTracks(this)
            % find connected components in the feature graph
            bins = conncomp(this.FeatureGraph);            
            allViewIds = this.FeatureGraph.Nodes.viewId;
            allPoints = this.FeatureGraph.Nodes.points;
            
            % sort bins, so all connected nodes are grouped.
            [binsSorted,sortInx] = sort(bins'); %#ok<TRSRT>
            
            % find bin transitions and remove tracks of length 1
            diffBins=[1;binsSorted(2:end) - binsSorted(1:end-1)];
            nonSingleBins= ~[diffBins(1:end-1)>0 & diffBins(2:end)>0; ...
                             diffBins(end)>0];
                         
            binsSorted=binsSorted(nonSingleBins);
            
            % if there are no tracks of length greater than 1, return empty
            % pointTrack array.
            if isempty(binsSorted)
                tracks = pointTrack.empty();    
                return;
            end
            
            sortInx=sortInx(nonSingleBins);
            diffBins=diffBins(nonSingleBins);
                                   
            % specify bin transitions as incluve index pairs
            binInx = find(diffBins);
            binInx = [binInx(1:end-1) (binInx(2:end)-1);...
                      binInx(end) length(binsSorted)];
            
            % create all tracks first to avoid reallocating memory on each
            % iteration
            numTracks = size(binInx, 1);
            tracks = repmat(pointTrack(0,[0 0]),1, numTracks);
            for i = 1:numTracks
                idx = sortInx(binInx(i,1):binInx(i,2));
                viewIds = allViewIds(idx);
                [~, uidx] = unique(viewIds, 'first');
                tracks(i) = pointTrack(viewIds(uidx), ...
                    allPoints(idx(uidx), :));
            end
        end

    
        %------------------------------------------------------------------
        function checkIfViewIsMissing(this, viewId)
            missingViewIdx = ~hasView(this, viewId);
            if any(missingViewIdx)
                missingViewIds = viewId(missingViewIdx);
                error(message('vision:viewSet:missingViewId', ...
                    missingViewIds(1)));
            end
        end
        
        %------------------------------------------------------------------
        function checkIfConnectionIsMissing(this, viewId1, viewId2)
            if ~hasConnection(this, viewId1, viewId2)
                error(message('vision:viewSet:missingConnection', ...
                    viewId1, viewId2));
            end
        end
        
        %------------------------------------------------------------------
        function view = getView(this, viewId)
            view = this.Views(getViewIndex(this, viewId), :);
        end
        
        %------------------------------------------------------------------
        function viewIdx = getViewIndex(this, viewId)
            viewIds = this.Views.ViewId;
            viewIdx = viewIds == viewId;
        end
        
        %------------------------------------------------------------------
        function conn = getConnections(this, viewId)
            viewIds = this.Connections.ViewId1;
            conn = this.Connections(viewIds == viewId, :);
        end
        
        %------------------------------------------------------------------
        function connIdx = getConnectionIndex(this, viewId1, viewId2)
            if isempty(this.Connections)
                connIdx = [];
            else
                viewIds1 = this.Connections.ViewId1;
                viewIds2 = this.Connections.ViewId2;
                connIdx = (viewIds1 == viewId1 & viewIds2 == viewId2);
            end
        end
        
        %------------------------------------------------------------------
        function connIdx = getConnectionIndexToAndFrom(this, viewId)
            viewIds1 = this.Connections.ViewId1;
            viewIds2 = this.Connections.ViewId2;
            connIdx = (viewIds1 == viewId | viewIds2 == viewId);
        end
        
        %--------------------------------------------------------------------------
        function [ViewId1, ViewId2, Matches, RelativeOrientation, RelativeLocation, ...
                unsetColumns] = parseConnectionInputs(this, viewId1, viewId2, varargin)
            
            ViewId1 = checkViewId(viewId1);
            ViewId2 = checkViewId(viewId2);
            
            if ~hasView(this, ViewId1)
                error(message('vision:viewSet:missingViewId', ViewId1));
            end
            
            if ~hasView(this, ViewId2)
                error(message('vision:viewSet:missingViewId', ViewId2));
            end            
            
            parser = inputParser;
            parser.addParameter('Matches', zeros(0, 2, 'uint32'), @checkMatches);
            parser.addParameter('Orientation', [], @checkOrientation);
            parser.addParameter('Location', [], @checkLocation);
            
            parser.parse(varargin{:});
            
            Matches = {parser.Results.Matches};
            RelativeOrientation = {parser.Results.Orientation};
            RelativeLocation = {parser.Results.Location};
            
            unsetColumns = parser.UsingDefaults;
            
            if ~ismember('Matches', unsetColumns)
                if isempty(Matches{1})
                    Matches{1} = zeros(0, 2, 'uint32');
                else
                    checkMatchesOutOfBounds(this, Matches{1}, ViewId1, ViewId2);
                end
            end
        end
        
        %------------------------------------------------------------------
        function checkMatchesOutOfBounds(this, matches, viewId1, viewId2)
            % Check that match indices are valid.
            view1 = getView(this, viewId1);
                view2 = getView(this, viewId2);
                
                points1 = view1.Points{1};
                points2 = view2.Points{1};
                
                areMatchesOutOfBounds = ...
                max(matches(:, 1)) > size(points1, 1) || ...
                max(matches(:, 2)) > size(points2, 1);
                if areMatchesOutOfBounds
                    error(message('vision:viewSet:matchIdxOutOfBounds'));
                end
            end
        
        %------------------------------------------------------------------
        function this = wipeMatches(this, viewIds)            
            if isempty(this.Connections)
                return;
            end
            
            % Force re-computing the point tracks from scratch.
            this.ShouldRecreateGraph = true;
            
            displayWarning = false;
            for id = viewIds(:)'
                idx = getConnectionIndexToAndFrom(this, id);
                % idx is a logical index. Use find to convert it to numeric
                % index.
                for j = find(idx')
                    if ~isempty(this.Connections.Matches{j})
                        displayWarning = true;
                    end
                    this.Connections.Matches{j} = [];                    
                end
            end
            
            if displayWarning                
                warning(message('vision:viewSet:deletingMatches'));
            end
        end
    end
    
%     methods(Hidden)
%         %------------------------------------------------------------------
%         function that = saveobj(this)
%             that.Views = this.Views;
%             that.Connections = this.Connections;
%             that.FeatureGraph = this.FeatureGraph;
%             that.ShouldRecreateGraph = this.ShouldRecreateGraph;
%         end
%     end
%     
%     methods(Static, Hidden)
%         %------------------------------------------------------------------
%         function this = loadobj(that)
%             this = viewSet;
%             this.Views = that.Views;
%             this.Connections = that.Connections;
%             this.FeatureGraph = that.FeatureGraph;
%             this.ShouldRecreateGraph = that.ShouldRecreateGraph;
%         end
%     end
end

%--------------------------------------------------------------------------
function [ViewId, Points, Orientation, Location, unsetColumns] = ...
    parseViewInputs(view, varargin)

ViewId = checkViewId(view);
parser = inputParser;
parser.addParameter('Points', zeros(0, 2), @checkPoints);
parser.addParameter('Orientation', [], @checkOrientation);
parser.addParameter('Location', [], @checkLocation);

parser.parse(varargin{:});

% Putting Points, Orientation and Location into cells,
% to make it easier to construct a table from them.
Points = {parser.Results.Points};
if ~isnumeric(Points{1})
    Points = {Points{1}.Location};
end

Orientation = {parser.Results.Orientation};
Location = {parser.Results.Location};

% Returning usingDefaults to be used in updateView to determine which
% columns should not be updated.
unsetColumns = parser.UsingDefaults;
end

%--------------------------------------------------------------------------
function view = checkViewTable(view)
validator = vision.internal.inputValidation.TableValidator;
validator.MinRows = 1;
validator.RequiredVariableNames = {'ViewId'};
validator.OptionalVariableNames = {'Points', 'Orientation', 'Location'};

validator.validate(view, mfilename, 'views');

view.ViewId = checkViewIds(view.ViewId);

for i = height(view)
    if ismember('Points', view.Properties.VariableNames)
        points = view{i, 'Points'}{1};
        if isempty(points)
            view{i, 'Points'} = {zeros(0, 2)};
        else
            view{i, 'Points'} = ...
                {vision.internal.inputValidation.checkAndConvertPoints(...
                points, mfilename, 'Points', false)};
        end
    end
    
    if ismember('Orientation', view.Properties.VariableNames)
        R = view{i, 'Orientation'}{1};
        checkOrientation(R);
    end
    
    if ismember('Location', view.Properties.VariableNames)
        t = view{i, 'Location'}{1};
        checkLocation(t);
    end
            
end
end

%--------------------------------------------------------------------------
function viewId = checkViewId(viewId)
validateattributes(viewId, {'numeric'}, ...
    {'nonsparse', 'scalar', 'integer', 'nonnegative'}, mfilename);
viewId = uint32(viewId);
end

%--------------------------------------------------------------------------
function viewIds = checkViewIds(viewIds)
validateattributes(viewIds, {'numeric'}, ...
    {'nonsparse', 'vector', 'integer'}, mfilename);
viewIds = uint32(viewIds(:));

if numel(unique(viewIds)) ~= numel(viewIds)
    error(message('vision:viewSet:duplicateViewIds'));
end

end

%--------------------------------------------------------------------------
function tf = checkPoints(points)
if ~isempty(points)
    vision.internal.inputValidation.checkPoints(points, 'viewSet', 'Points');
end
tf = true;
end

%--------------------------------------------------------------------------
function tf = checkOrientation(R)
if ~isempty(R)
    vision.internal.inputValidation.validateRotationMatrix(R, 'viewSet',...
        'Orientation');
end
tf = true;
end

%--------------------------------------------------------------------------
function tf = checkLocation(t)
if ~isempty(t)
    vision.internal.inputValidation.validateTranslationVector(t, 'viewSet', ...
        'Location');
end
tf = true;
end

%--------------------------------------------------------------------------
function tf = checkMatches(matches)
if ~isempty(matches)
validateattributes(matches, {'numeric'}, ...
    {'nonsparse', '2d', 'ncols', 2, 'integer', 'positive'}, mfilename);
end
tf = true;
end
        
        