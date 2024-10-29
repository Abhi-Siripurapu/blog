
  July 19, 2021

# leveling up with torch: nuances for better speed, efficiency, and sanity

  

when you're working with `torch`, it’s easy to get trapped in familiar patterns. we build models, define layers, train, repeat. but there's a lot under the hood that, if leveraged right, can save time, memory, and sometimes even sanity. here’s a look at some less-talked-about techniques and idioms that can change how you approach `torch`, especially when dealing with large models or datasets. if you’ve been following standard tutorials, prepare for a little unlearning.

  The following are some things that irritate me when I see it in projects,

## 1. don’t default to `torch.Tensor`—use `torch.FloatTensor` instead

  

it sounds like a minor tweak, but `torch.Tensor` doesn’t do what you think it does. by default, `torch.Tensor` will create a tensor in the default dtype (`float32`), but this can lead to subtle bugs, especially if you’re dealing with different precision levels (`float16` for mixed precision, `float64` for double precision).

  

if you know the dtype you need, specify it explicitly. `torch.FloatTensor` or `torch.DoubleTensor` avoids ambiguity and enforces consistency. otherwise, you’re at the mercy of the default type, which can change with updates or different devices.

  

```python

# don’t do this

x = torch.Tensor([1.0, 2.0, 3.0])

  

# do this

x = torch.FloatTensor([1.0, 2.0, 3.0])

```

  

also, note that using `FloatTensor` vs. `DoubleTensor` can have a huge impact on speed, especially on gpus. unless you need high precision, stick to `FloatTensor`.

  

## 2. ditch `torch.cat()` in favor of `torch.stack()` for efficient concatenations

  

when combining tensors along a new axis, most people reach for `torch.cat()`, but `torch.stack()` is often more efficient. `torch.stack()` avoids the need for reallocation in many cases, making it faster and less memory-intensive.

  

```python

# most people do this

a = torch.rand(10, 5)

b = torch.rand(10, 5)

c = torch.cat((a.unsqueeze(0), b.unsqueeze(0)), dim=0)

  

# do this instead

c = torch.stack((a, b))

```

  

`torch.stack()` is ideal when you’re building a batch of tensors or combining tensors along a new dimension. it’s a subtle difference, but it adds up, especially in loops.

  

## 3. use `.to()` smartly—avoid redundant device transfers

  

`tensor.to(device)` is the standard way to send a tensor to gpu or change dtype, but doing this repeatedly wastes time. if you’re calling `.to()` multiple times on the same tensor, consolidate those calls. also, avoid calls to `.to()` inside loops if you can move everything to the device once.

  

consider this refactoring:

  

```python

# bad practice

x = torch.rand(1000, 1000)

for _ in range(1000):

    x_gpu = x.to('cuda')  # moving to gpu each time in the loop

  

# better

x = torch.rand(1000, 1000, device='cuda')  # move to device once, outside the loop

for _ in range(1000):

    # do your operations with x_gpu

    pass

```

  

also, when possible, set `device='cuda'` during tensor creation (e.g., `torch.zeros(..., device='cuda')`) instead of creating on cpu and then calling `.to('cuda')`.

  

## 4. get serious about `torch.nn.functional` for flexibility and speed

  

if you’re using `nn.Conv2d` or `nn.ReLU`, you’re probably initializing extra parameters each time. `torch.nn.functional` provides layer equivalents like `F.conv2d` and `F.relu` that skip the parameter-heavy initializations, making them faster for certain operations. `torch.nn.functional` is especially powerful in cases where you want finer control over layers without adding unnecessary parameters.

  

```python

import torch.nn.functional as F

  

# instead of this

conv_layer = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1)

output = conv_layer(input)

  

# use this when you don’t need parameters

weight = torch.rand(10, 1, 3, 3)  # manually specify weights if needed

output = F.conv2d(input, weight, stride=1)

```

  

## 5. leverage `torch.no_grad()` and `torch.inference_mode()` for read-only inference

  

when you’re running inference, `torch.no_grad()` is your friend, but `torch.inference_mode()` is even better for certain situations. `torch.inference_mode()` goes beyond `no_grad` by further optimizing memory usage, especially when you’re working with large models in read-only mode.

  

```python

# common, but not optimal

with torch.no_grad():

    output = model(input)

  

# better for inference

with torch.inference_mode():

    output = model(input)

```

  

`torch.inference_mode()` is new-ish, but it’s more memory efficient because it skips autograd-related data tracking altogether. use it when you’re absolutely certain there’s no backprop needed.

  

## 6. profile first, optimize later—`torch.profiler` to the rescue

  

before you jump into optimization, make sure you know where the bottlenecks actually are. `torch.profiler` is a powerful tool for measuring performance across multiple dimensions (memory usage, device transfers, etc.). run a profiling session to figure out where the lags are, and target those areas first.

  

```python

import torch

import torch.profiler

  

# basic profiling setup

with torch.profiler.profile(

    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],

    record_shapes=True

) as prof:

    model(input)

  

print(prof.key_averages().table(sort_by="cuda_time_total"))

```

  

this profiling session helps you catch inefficient operations (like redundant `.to()` calls or `torch.cat()` where `torch.stack()` would suffice).

  

## 7. beware of python’s garbage collector

  

torch doesn’t play nicely with python’s garbage collector (`gc`), especially on cuda. holding onto unnecessary references or using `gc.collect()` excessively can hurt performance. for models running on cuda, let pytorch manage memory, and avoid calling `gc.collect()` unless you absolutely need to clear memory.

  

also, to free cuda memory manually, use `torch.cuda.empty_cache()` but sparingly—it should be a last resort rather than a regular step in your workflow.

  

```python

# avoid unless absolutely necessary

import gc

gc.collect()  # slows things down when called excessively

  

# better approach

torch.cuda.empty_cache()  # clears cache only if needed, helps avoid cuda OOM errors

```

  

## wrapping up: little changes, big impact

  

torch has a lot of hidden tricks that most users don’t notice. by tweaking how you create tensors, combine data, handle device transfers, and manage inference, you can shave seconds off your run times (or even avoid hidden bugs and memory leaks). these aren’t radical changes, but when you add them up, they can mean the difference between smooth model training and frustrating gpu hangs.

  
