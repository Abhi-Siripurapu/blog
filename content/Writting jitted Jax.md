June 21, 2024
  

[Torch's design decisions](https://neel04.github.io/my-website/blog/pytorch_rant/) have been pretty poor for a while, so I've been experimenting with Jax. It's great, but only if you can actually take advantage of the speedups from jit. Below are some best practices I've learned from playing around with this for a while. 


## What Exactly is `jax.jit`?

JIT stands for *Just-In-Time* compilation. Normally, Python runs line-by-line, interpreting everything at runtime, which is convenient for flexibility but can be slow for performance-critical code. JAX changes the game by introducing the ability to compile your Python functions with `jax.jit`. This allows JAX to optimize your code ahead of time, reducing overhead during execution.

  
In essence, `jax.jit` takes your Python code and translates it into highly optimized, compiled code using XLA (Accelerated Linear Algebra), making your functions run orders of magnitude faster—especially for large-scale tensor operations.

  

## How to Use `jax.jit`

  

Using `jax.jit` is as simple as wrapping your function with it. Here’s a basic example (try this out yourself):


 ```python

import jax

import jax.numpy as jnp

  

def matmul(x, y):

    return jnp.dot(x, y)
``` 

# Without JIT
``` python 
x = jnp.ones((1000, 1000))

y = jnp.ones((1000, 1000))

%timeit matmul(x, y)
```



  

# With JIT
``` python 
matmul_jit = jax.jit(matmul)

%timeit matmul_jit(x, y)
```


This small change can result in **dramatic** speedups, especially when running the function repeatedly. But why?

### Under the Hood

  
When you use `jax.jit`, JAX compiles your function into a single execution graph, optimizing it with XLA. XLA looks for opportunities to fuse operations together, minimize memory access, and streamline the underlying computations. This results in reduced overhead for running your function, turning Python’s interpreted sluggishness into C++-like efficiency.

  

It also memoizes the compiled function, meaning that if you run the function multiple times with the same shapes and data types, you’ll skip recompilation and just execute the precompiled code.

  

## When Should You Use `jax.jit`?

  

You might be tempted to slap `jax.jit` on everything, but you will end up wasting more time debugging than if you had simply just used torch..

  

1. **Hot Functions**: Functions that are called repeatedly with the same or similar shapes will benefit the most. The compilation time for `jax.jit` can sometimes be non-trivial, so you want to make sure the payoff justifies the upfront cost.

2. **Numerical Computation**: If your function is doing heavy number crunching (matrix multiplications, tensor manipulations, etc.), `jax.jit` will likely provide a huge boost. Pure control flow functions, or those with lots of I/O, won't benefit as much.
3. **Stable Shapes**: `jax.jit` works best when the shapes of your arrays remain consistent between function calls. If the shape varies a lot, you might end up recompiling the function frequently, which negates the performance gains.

  

### Example: Optimizing a Neural Network Layer

  

Let’s see how `jax.jit` can turbocharge something more interesting: a fully connected layer in a neural network.

  

``` python
import jax

import jax.numpy as jnp

  

def fully_connected(x, w, b):

    return jnp.dot(x, w) + b

  

# Data and parameters

x = jnp.ones((1000, 512))  # Input batch of 1000 examples with 512 features

w = jnp.ones((512, 256))   # Weights for 256 neurons

b = jnp.ones((256,))       # Bias term

  

# Without JIT

%timeit fully_connected(x, w, b)

  

# With JIT

fully_connected_jit = jax.jit(fully_connected)

%timeit fully_connected_jit(x, w, b)
```

  

By simply adding `jax.jit`, you’ll often see anywhere from a **5x to 10x speedup** depending on the size of your data and the complexity of the function.


### JIT + Grad = Magic

  
You can also use `jax.jit` in combination with `jax.grad` for computing gradients efficiently. Here’s how you would use `jax.jit` to accelerate a loss function during training:

  

``` python
# Define a loss function

def loss_fn(params, x, y):

    predictions = fully_connected(x, params['w'], params['b'])

    return jnp.mean((predictions - y) ** 2)

  

# Initialize parameters

params = {'w': jnp.ones((512, 256)), 'b': jnp.ones((256,))}

x = jnp.ones((1000, 512))

y = jnp.ones((1000, 256))

  

# Get the gradient of the loss function

grad_fn = jax.grad(loss_fn)

  

# Without JIT

%timeit grad_fn(params, x, y)

  

# With JIT

grad_fn_jit = jax.jit(grad_fn)

%timeit grad_fn_jit(params, x, y)
```

  

Combining `jax.jit` with `jax.grad` not only accelerates the forward pass (calculating the predictions), but also speeds up the backward pass (computing the gradients), very useful for training nns at scale.

  

## Mistakes with `jax.jit`

  

While `jax.jit` is extremely powerful, it’s not without its quirks. Here are a few common pitfalls to avoid:

  

1. **Side Effects**: `jax.jit` expects pure functions, meaning that any side effects (e.g., printing, modifying global variables) won’t work as expected. If your function has side effects, they’ll be silently ignored.

2. **Dynamic Shapes**: As mentioned earlier, `jax.jit` is most effective when your input shapes are stable. If the shape of your arrays changes often, JAX will need to recompile the function, which can offset the performance gains. I spent a lot of hours trying to figure out why my jitted code wasn't working, don't do the same think.

3. **Python Control Flow**: Python’s native control flow (e.g., `if`, `for`, `while`) inside `jax.jit`-compiled functions won’t always behave as expected. JAX uses its own control flow primitives like `jax.lax.cond` and `jax.lax.scan` to handle loops and conditionals efficiently. If you have complex control logic, make sure to use JAX’s alternatives.

  

### Example: Control Flow in JIT

  

``` python
import jax

import jax.numpy as jnp

  

# This won't work well with jax.jit due to Python control flow

def faulty_function(x):

    if x.sum() > 0:

        return x * 2

    else:

        return x / 2

  

# Use JAX control flow primitives instead

def better_function(x):

    return jax.lax.cond(x.sum() > 0, lambda x: x * 2, lambda x: x / 2, x)

  

better_function_jit = jax.jit(better_function)
```
  

## Conclusion: Embrace the Speed

This takes a while to get used to, especially having to make your arrays non-dynamic, but if you can get the hang of it the performance gains are well worth it. Spend a weekend or two playing with Jax.