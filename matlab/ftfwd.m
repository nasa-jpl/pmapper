function out = ftfwd(inp)
out = fftshift(fft2(ifftshift(inp)));
end