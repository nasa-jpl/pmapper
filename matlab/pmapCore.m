function fHat = pmapCore(fHat, g, H, Hstar, zoomFactor)
%PMAPCORE is the mathematical core of PMAP.
% Inputs:
%  fHat       : kth object estimate, double of dimension (m, n)
%  g          : source image, double of dimension (a, b)
%  H          : optical transfer function; double (complex) of dimension (m, n)
%  Hstar      : complex conjugate of H; double (complex) of dimension (m, n)
%  zoomfactor : scalar float, ratio of m/a or n/b
% Outputs:
%   fhat      : (k+1)th object estimate, double of dimension (m, n)
if (zoomFactor == 1) && (size(fHat, 1) ~= size(H, 1))
    error('unity zoom, but previous object estimate and OTF not of same dimension')
end
Fhat = ftfwd(fHat);
denom = real(ftrev(Fhat .* H));

% lanczos = truncated sinc ~= optimal interpolator
if zoomFactor ~= 1
    denom = imresize(denom, 1/zoomFactor, "lanczos3");
    kernel = (g ./ denom) - 1;
    kernel = imresize(kernel, zoomFactor, "lanczos3");
else
    kernel = (g ./ denom) - 1;
end

R = ftfwd(kernel);
grad = real(ftrev(R .* Hstar));
fHat = fHat .* exp(grad);
end