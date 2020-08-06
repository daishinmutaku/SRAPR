using Statistics

data = [1 0; 0 1]

scaledData = zeros((size(data)[1], size(data)[2]))
for i in 1:size(data)[2]
  scaledData[:, i] = row .- m / s
end
scaledData
