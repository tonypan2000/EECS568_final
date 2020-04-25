function [features, validPoints] = extract_features(frame)

	global Debug

	if isColor(frame)
		frame = rgb2gray(frame);
	end
	points = detectSURFFeatures(frame);
	[features, validPoints] = extractFeatures(frame, points);
	
	if Debug.displayFeaturesOnImages
		figure; imshow(frame); hold on; plot(validPoints); hold off;
	end
end

function result = isColor(image)
	result = size(image, 3) == 3;
end