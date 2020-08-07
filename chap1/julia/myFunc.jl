module MyFunc
  export scale, outer, cul1se

  using Statistics


  function scale(data)
    scaledData = zeros((size(data)[1], size(data)[2]))
    for i in 1:size(data)[2]
      row = data[:, i]
      scaledData[:, i] = (row .- mean(row)) / std(row)
    end
    scaledData
  end

  function outer(range1, range2, fun)
    vec1 = Vector(range1)
    vec2 = Vector(range2)
    mat = zeros((length(vec1), length(vec2)))
    for i in 1:size(mat)[1]
      for j in 1:size(mat)[2]
        mat[i, j] = fun(vec1[i], vec2[j])
      end
    end
    mat
  end

  function cul1se(cv)
    minloss, index = findmin(cv.meanloss)
    maximum(cv.lambda[cv.meanloss .<= minloss + cv.stdloss[index]])
  end
end