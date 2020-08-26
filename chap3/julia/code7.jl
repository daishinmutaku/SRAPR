using Flux
using Flux.Data.MNIST
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Printf, BSON
using Statistics


model = Chain(
  Conv((3,3), 1 => 64, relu, pad=SamePad()),
  x -> maxpool(x, (2, 2)),
  Conv((3,3), 64 => 32, relu, pad=SamePad()),
  x -> maxpool(x, (2, 2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(1568, 128, relu),
  Dense(128, 64, relu),
  Dense(64, 10),
  softmax
)

## ---
train_labels = MNIST.labels()
train_imgs = MNIST.images()
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end
batch_size = 128
mb_idxs = partition(1:length(train_imgs), batch_size)
train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]
test_imgs = MNIST.images(:test)
test_labels = MNIST.labels(:test)
test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs))

loss(x, y) = crossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
opt = RMSProp()
best_acc = 0.0
last_improvement = 0
evalcb = () -> @show(accuracy(test_set...))
epochs = 1
historyTest = zeros(epochs)
for epoch_idx in 1:epochs
  global best_acc, last_improvement
  # Train for a single epoch
  Flux.train!(loss, params(model), train_set, opt, cb = throttle(evalcb, 10))

  # Calculate accuracy:
  acc = accuracy(test_set...)
  @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))

  # If our accuracy is good enough, quit out.
  if acc >= 0.999
      @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
      break
  end

  # If this is the best accuracy we've seen so far, save the model out
  if acc >= best_acc
      @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
      BSON.@save "mnist_conv.bson" model epoch_idx acc
      best_acc = acc
      last_improvement = epoch_idx
  end

  # If we haven't seen improvement in 5 epochs, drop our learning rate:
  if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
      opt.eta /= 10.0
      @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

      # After dropping learning rate, give it a few epochs to improve
      last_improvement = epoch_idx
  end

  if epoch_idx - last_improvement >= 10
      @warn(" -> We're calling this converged.")
      break
  end
end
