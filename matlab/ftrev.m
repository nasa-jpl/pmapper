function out = ftrev(inp)
out = fftshift(ifft2(ifftshift(inp)));
end