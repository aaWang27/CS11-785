import numpy as np
from regex import A

import mytorch.tensor as tensor
from mytorch.autograd_engine import Function


def unbroadcast(grad, shape, to_keep=0):
    # print(grad.shape, shape)
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        # print(i)
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
            # print(grad.shape, shape)
    return grad


class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T)


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data)


"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.multiply(np.ones(a.shape), grad_output.data)
        # dL/db = dout/db * dL/dout
        grad_b = np.multiply(np.ones(b.shape), grad_output.data)

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)

        return c
        # raise Exception("TODO: Implement '-' forward")

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = np.multiply(np.ones(a.shape), grad_output.data)
        grad_b = -1 * np.multiply(np.ones(b.shape), grad_output.data)

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b
        # raise Exception("TODO: Implement '-' backward")


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis=axis, keepdims=keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None


# TODO: Implement more Functions below
class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.multiply(a.data, b.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)

        return c
        # raise Exception("TODO: Implement '-' forward")

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = np.multiply(b.data, grad_output.data)
        grad_b = np.multiply(a.data, grad_output.data)

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.divide(a.data, b.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)

        return c
        # raise Exception("TODO: Implement '-' forward")

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = np.multiply(np.divide(np.ones(b.shape), b.data), grad_output.data)
        grad_b = -1 * np.multiply(np.multiply(a.data, np.divide(np.ones(b.shape), np.square(b.data))), grad_output.data)

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("Arg must be Tensor: {}".format(type(a).__name__))

        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        c = tensor.Tensor(np.maximum(0, a.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        grad_a = np.multiply(grad_output.data, np.where(a.data > 0, 1, 0))

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))

        return grad_a


class Matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.matmul(a.data, b.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = np.dot(grad_output.data, b.data.T)
        grad_b = np.dot(a.data.T, grad_output.data)

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class XELoss(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Arg must be Tensor: {}, {}".format(type(a).__name__, type(b).__name__))

        ctx.save_for_backward(a, b)

        batch_size, num_classes = a.shape

        max = np.expand_dims(np.max(a.data, axis=1), 1)
        exps = np.exp(a.data - max)
        log_softmax = exps / np.expand_dims(np.sum(exps, axis=1), 1)

        log_likelihood = -np.log(log_softmax[np.arange(batch_size), b.data] + 1e-8)
        nllloss = np.sum(log_likelihood) / batch_size

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(nllloss, requires_grad=requires_grad,
                          is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        batch_size, num_classes = a.shape

        max = np.expand_dims(a.data.max(axis=1), 1)
        exps = np.exp(a.data - max)
        log_softmax = exps / np.expand_dims(np.sum(exps, axis=1), 1)

        log_softmax[np.arange(batch_size), b.data] -= 1
        grad_a = log_softmax / batch_size

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))

        return grad_a


class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("Arg must be Tensor: {}".format(type(a).__name__))

        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        c = tensor.Tensor(np.sqrt(a.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        grad_a = np.multiply(grad_output.data, np.ones(a.shape) / (2 * np.sqrt(a.data)))

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        return grad_a


class Var(Function):
    @staticmethod
    def forward(ctx, a, axis=0, ddof=0):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("Arg must be Tensor: {}".format(type(a).__name__))
        ctx.axis = axis
        ctx.ddof = ddof
        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        c = tensor.Tensor(np.var(a.data, axis=axis, ddof=ddof), requires_grad=requires_grad,
                          is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        num = 2 * (a.data - np.mean(a.data, axis=ctx.axis))
        denom = a.shape[0] - ctx.ddof

        grad_a = np.multiply(grad_output.data, num / denom)

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        return grad_a


class Dropout(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, is_train=False):
        """Forward pass for dropout layer.

        Args:
            ctx (ContextManager): For saving variables between forward and backward passes.
            x (Tensor): Data tensor to perform dropout on
            p (float): The probability of dropping a neuron output.
                       (i.e. 0.2 -> 20% chance of dropping)
            is_train (bool, optional): If true, then the Dropout module that called this
                                       is in training mode (`<dropout_layer>.is_train == True`).

                                       Remember that Dropout operates differently during train
                                       and eval mode. During train it drops certain neuron outputs.
                                       During eval, it should NOT drop any outputs and return the input
                                       as is. This will also affect backprop correspondingly.
        """
        if not type(x).__name__ == 'Tensor':
            raise Exception("Only dropout for tensors is supported")

        # raise NotImplementedError("TODO: Implement Dropout(Function).forward() for hw1 bonus!")
        ctx.prob = p
        ctx.train = is_train
        ctx.save_for_backward(x)

        if is_train:
            requires_grad = x.requires_grad
            mask = np.random.binomial(1, 1 - p, x.shape) / (1 - p)
            ctx.mask = mask
            c = tensor.Tensor(mask * x.data, requires_grad=requires_grad, is_leaf=not requires_grad)
        else:
            requires_grad = x.requires_grad
            c = tensor.Tensor(x.data, requires_grad=requires_grad, is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # raise NotImplementedError("TODO: Implement Dropout(Function).backward() for hw1 bonus!")
        if ctx.train:
            a = ctx.saved_tensors[0]
            grad_a = grad_output.data * ctx.mask
            grad_a = tensor.Tensor(unbroadcast(grad_a, ctx.mask.shape))
        else:
            a, _ = ctx.saved_tensors
            grad_a = a.data
            grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))

        return grad_a


def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape

    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.

    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss

    raise Exception("TODO: Implement XELoss for comp graph")


def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]

    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad=True)
