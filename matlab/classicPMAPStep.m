function st = classicPMAPStep(st)
%PREPCLASSICPMAPSTATE does all pre-computations for classic PMAP
%  classic PMAP has no multi-frame or bayer capability.  Panchromatic only.
% Inputs:
%   st : struct containing the STate for the PMAP iterator
% Outputs:
%   st : struct containing the updated STate for the PMAP iterator

st.fHat = pmapCore(st.fHat, st.img, st.otf, st.otfconj, st.zoomFactor);
st.iter = st.iter + 1;
end