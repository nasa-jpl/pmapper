function st = prepClassicPMAPState(img, psf, fHat)
%PREPCLASSICPMAPSTATE does all pre-computations for classic PMAP
%  classic PMAP has no multi-frame or bayer capability.  Panchromatic only.
% Inputs:
%  img  : image from the camera, double of dimension (m, n)
%  psf  : psf corresponding to img, double of dimension (a, b)
%  fHat : initial object estimate, double of dimension (a, b); defaults to
%         a resize of img
% Outputs:
%   st : struct containing the STate for the PMAP iterator
otf = ftfwd(psf);
c1 = floor(size(img,1)/2)+1; % +1 zero-based to 1-based indexing
c2 = floor(size(img,2)/2)+1;
otf = otf / otf(c1,c2);
otfconj = conj(otf);
zoomFactor = size(psf, 1) / size(img, 1);

if nargin < 3
    fHat = imresize(img, zoomFactor, "lanczos3");
end


st = struct();
st.img = img;
st.psf = psf;
st.otf = otf;
st.otfconj = otfconj;
st.fHat = fHat;
st.zoomFactor = zoomFactor;
st.iter = 0;
end