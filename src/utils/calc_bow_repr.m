function repr = calc_bow_repr(features, kdtree_mdl, numCodewords)
idx = knnsearch(kdtree_mdl, features);

repr = histcounts(idx, 1:(numCodewords + 1));
repr = repr / sum(repr);
end