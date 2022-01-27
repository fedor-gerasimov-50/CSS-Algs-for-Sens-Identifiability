# CSS-Algs-for-Sens-Identifiability


Main Subset Selection Routine: SubsetSelect.m
  Requires: n x p sensitivity matrix, tol (optional), k (optional), algorithm numbers
  Each algorithm (B1, B4, B3, srrqr) corresponds to an input 1:4, respectively, to PSS.m
  PSS.m can accept rank tolerance (tol, e.g. 1e-8) or numerical rank (k < p)--if both inputted, defaults to k; if none, defaults to tol = 1e-8 to find k
  
  Then calls column subset selection algorithms based on alg numbers
  Within each call to a CSS algorithm (PSS_X.m), SuccessCheck.m computes the success criteria (2.4) and (2.5) as well as cond(S) and cond(S_1) after CSS
  
