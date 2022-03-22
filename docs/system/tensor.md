# Tensor

A small tensor library for manipulating numerical data in DeepMind Lab2D. The
functionality is similar to that of tensors in PyTorch. Tensor indices are
1-based.

Tensors can share storage. When data is mutated in one tensor, all other tensors
that share that storage observe the mutation.

Copies of a tensor share storage with the source tensor. All
[layout operations](#layout-operations) result in tensors that share storage
with the source tensor.

Tensors with new storage (initially not shared with any other tensor) can only
be created by one of the explicit [creation](#creation) methods described below.

Example:

```lua
local tensor = require 'system.tensor'

-- Create a rank-2 tensor with extents 4 and 5. Tensors constructed with this
-- function are zero-initialized.
> z = tensor.DoubleTensor(4, 5)
> =z
[tensor.DoubleTensor]
Shape: [4, 5]
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
```

You can create a tensor from a table, too:

```lua
> =tensor.DoubleTensor{{1, 2}, {3, 4}, {5, 6}}
[tensor.DoubleTensor]
Shape: [3, 2]
[[1, 2],
 [3, 4],
 [5, 6]]
```

## Types

There are 7 types of tensor:

*   `tensor.ByteTensor`
*   `tensor.CharTensor`
*   `tensor.Int16Tensor`
*   `tensor.Int32Tensor`
*   `tensor.Int64Tensor`
*   `tensor.FloatTensor`
*   `tensor.DoubleTensor`

By default use `DoubleTensor`. Other data types may be difficult to use
correctly, because numeric conversions between Lua's number representation and
the representations used by the underlying tensor implementation may be invalid.

## Reading/Writing

Reading and writing to a tensor can be done via the function `val()`.

### val() -> value|table

When called on a scalar tensor, val() returns the value of that element.
Otherwise returns a nested table of the same shape and values.

Value form:

```lua
> myTensor = tensor.Int64Tensor{{1, 2}, {3, 4}}
> assert(myTensor(1, 1):val() == 1)
> assert(myTensor(1, 2):val() == 2)
> assert(myTensor(2, 1):val() == 3)
> assert(myTensor(2, 2):val() == 4)
```

Table form:

```lua
> myTensor = tensor.Int64Tensor{{1, 2}, {3, 4}}
> t = myTensor:val()
> assert(t[1][1] == 1)
> assert(t[1][2] == 2)
> assert(t[2][1] == 3)
> assert(t[2][2] == 4)
```

### val(value|table)

When called on a scalar tensor, it sets the value of that element to `value`.
Otherwise the value must be a nested table that matches the shape of the tensor.
The tensor is then assigned to the values in the table.

Value form:

```lua
> myTensor = tensor.Int64Tensor{{0, 0}, {0, 0}}
> myTensor(1, 1):val(1)
> myTensor(1, 2):val(2)
> myTensor(2, 1):val(3)
> myTensor(2, 2):val(4)
> =myTensor
[tensor.Int64Tensor]
Shape: [2, 2]
[[1, 2],
 [3, 4]]
```

Table form:

```lua
> myTensor = tensor.Int64Tensor{{0,0}, {0,0}}
> myTensor:val{{1,2}, {2,3}}
> =myTensor
[tensor.Int64Tensor]
Shape: [2, 2]
[[1, 2],
 [2, 3]]
```

## Creation

For each type of tensor there is a corresponding construction function and
conversion method.

Type           | Construction          | Conversion
:------------- | :-------------------- | :----------
`ByteTensor`   | `tensor.ByteTensor`   | `:byte()`
`CharTensor`   | `tensor.CharTensor`   | `:char()`
`Int16Tensor`  | `tensor.Int16Tensor`  | `:int16()`
`Int32Tensor`  | `tensor.Int32Tensor`  | `:int32()`
`Int64Tensor`  | `tensor.Int64Tensor`  | `:int64()`
`FloatTensor`  | `tensor.FloatTensor`  | `:float()`
`DoubleTensor` | `tensor.DoubleTensor` | `:double()`

The construction functions and conversion methods, including `clone`(), create
tensors that own their storage. The tensor starts in row-major order, where
consecutive elements of the tensor in layout order are contiguous in memory.
(Note in particular that even a conversion to the same type has the effect of
creating a compacted, contiguous copy of the storage.)

External code may be able to create tensors that do not own their storage. See
[Ownership](#ownership) for more details.

### `tensor.DoubleTensor`()

Creates a rank-0 tensor (a scalar).

```lua
-- Creates a scalar with the value 5.
> =tensor.DoubleTensor():fill(5)
[tensor.DoubleTensor]
5
```

### `tensor.DoubleTensor`(*dim1*, *dim2*, ..., *dimK*)

Creates a rank-K tensor with extents dim1, dim2, ..., dimK. We refer to the
tuple (dim1, dim2, ..., dimK) as the *shape* of the tensor.

```lua
> =tensor.DoubleTensor(3, 2)
[tensor.DoubleTensor]
Shape: [3, 2]
[[0, 0],
 [0, 0],
 [0, 0]]
```

### `tensor.DoubleTensor`\{\{*val11*, ...\}, \{*val21*, ...\}, ...\}

Creates a tensor from the given hierarchy of tables and values. The shape is
implied.

```lua
> =tensor.DoubleTensor{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}
[tensor.DoubleTensor]
Shape: [2, 2, 2]
[[[1, 2],
  [3, 4]],

 [[5, 6],
  [7, 8]]]
```

### `tensor.DoubleTensor`\{file=\*kwargs*\}

Creates a rank-1 tensor by reading bytes within a file. The file is read in
system local endian order. 'kwargs' must be a table with the following keyword
arguments:

*   'name': Name of the file to read. Must be a valid filename.
*   'byteOffset': Offset from the beginning of the file at which reading starts.
    Optional, default 0.
*   'numElements': Number of elements to read starting at the offset. Optional,
    defaults to largest number of elements that is available from the given
    offset.

If the offset is outside of the range \[0, file size\] or if reading count
values plus the offset would exceed the size of the file an error is thrown.

Assuming the file 'data.bin' is 64 * 8 bytes long, generated by writing the list
of doubles from 0 to 63 inclusive in local endian format:

```lua
-- Read whole file.
assert(
    tensor.DoubleTensor{
        file = {name = 'data.bin'}
    } == tensor.DoubleTensor{range = {0, 63}}
)

-- Read count elements.
assert(
    tensor.DoubleTensor{
        file = {name = 'data.bin', numElements = 10}
    } == tensor.DoubleTensor{range = {0, 9}}
)

-- Read end of file.
assert(
    tensor.DoubleTensor{
        file = {name = 'data.bin', byteOffset = 40 * 8}
    } == tensor.DoubleTensor{range = {40, 63}}
)

-- Read middle of file.
assert(
    tensor.DoubleTensor{
        file = {name = 'data.bin', byteOffset = 40 * 8, numElements = 6}
    } == tensor.DoubleTensor{range = {40, 45}}
)
```

### `tensor.DoubleTensor`\{range=\{*from*, *to*, *step*\}\}

Creates a rank-1 tensor with values in the closed interval spanned by endpoints
*from* and *to*, starting from *from* and advancing by the given *step*. Ranges
can be also defined as range=\{*from*, *to*\}, where the implicit *step* value
is 1, and range=\{*to*\}, where the implicit *from* value is also 1.

```lua
> =tensor.Int64Tensor{range = {5}}
[tensor.Int64Tensor]
Shape: [5]
[1, 2, 3, 4, 5]
> =tensor.Int64Tensor{range = {3, 5}}
[tensor.Int64Tensor]
Shape: [3]
[3, 4, 5]
> =tensor.DoubleTensor{range = {1, 2, 0.5}}
[tensor.DoubleTensor]
Shape: [3]
[  1, 1.5,   2]
```

Please note that the upper bound is included in the range if and only if the
interval width is divisible by the step size:

```lua
> =tensor.DoubleTensor{range = {1, 3, 1}}
[tensor.DoubleTensor]
Shape: [3]
[1, 2, 3]
> =tensor.DoubleTensor{range = {1, 2.75, 1}}
[tensor.DoubleTensor]
Shape: [2]
[1, 2]
```

Also, the extents of the range are subject to the numerical precision of the
element type:

```lua
> =tensor.FloatTensor{range = {300000000, 300000001, 0.5}}
[tensor.FloatTensor]
Shape: [1]
[3e+08]
```

### Conversion

A new tensor is returned with new storage and of the specified type. The new
tensor has the same shape as the original and contains the same values after
conversion.

**Warning!** Conversion may have undefined behaviour if an element value cannot
be represented in the new type.

```lua
local z = tensor.DoubleTensor{{1, 2}, {3, 4}, {5, 6}}
assert(z:byte() == tensor.ByteTensor{{1, 2}, {3, 4}, {5, 6}})
assert(z:char() == tensor.CharTensor{{1, 2}, {3, 4}, {5, 6}})
assert(z:int16() == tensor.Int16Tensor{{1, 2}, {3, 4}, {5, 6}})
assert(z:int32() == tensor.Int32Tensor{{1, 2}, {3, 4}, {5, 6}})
assert(z:int64() == tensor.Int64Tensor{{1, 2}, {3, 4}, {5, 6}})
assert(z:float() == tensor.FloatTensor{{1, 2}, {3, 4}, {5, 6}})

-- This call is equivalent to calling z:clone().
assert(z:double() == tensor.DoubleTensor{{1, 2}, {3, 4}, {5, 6}})
```

### `clone`()

Equivalent to the conversion function of the same type as the source tensor.

```lua
z1 = tensor.DoubleTensor{...}
z2 = z1:clone()   -- same as z1:double()
```

## Metadata

### `size`()

Returns the number of elements in the tensor.

```lua
> z = tensor.DoubleTensor{{1, 2}, {3, 4}, {5, 6}}
> =z:size()
6
```

### `shape`()

Returns the shape of the tensor.

```lua
> z = tensor.DoubleTensor{{1, 2}, {3, 4}, {5, 6}}
> s = z:shape()
> assert(#s == 2 and s[1] == 3 and s[2] == 2)
```

### `ownsStorage`()

Returns whether the tensor storage will stay valid. See [Ownership](#ownership)
for more details.

### `isContiguous`()

Returns whether the tensor is in row-major order and consecutive elements are
contiguous in memory. [Layout operations](#layout-operations) may cause the
tensor to become non-contiguous.

```lua
-- All tensors are created contiguous.
> =tensor.DoubleTensor{{1, 2}, {3, 4}}:isContiguous()
true
> =tensor.DoubleTensor{{1, 2}, {3, 4}}:transpose(1, 2):isContiguous()
false
> =tensor.DoubleTensor{{1, 2}, {3, 4}}:transpose(1, 2):clone():isContiguous()
true
```

When calling an API that requires a contiguous tensor, a contiguous copy may be
created via the `clone()` method.

```lua
-- Calling an API that requires a contiguous tensor.
> callApi(myTensor:isContiguous() and myTensor or myTensor:clone())
```

## Rounding/Clamping

Rounding operations can be applied to nonintegral types.

### `floor`()

Rounds the tensor elements to the greatest preceding integers.

```lua
> z = tensor.DoubleTensor{{-2.25, -1.75}, {0.5, 1.0}}
> =z
[tensor.DoubleTensor]
Shape: [2, 2]
[[-2.25, -1.75],
 [  0.5,     1]]
> =z:floor()
[tensor.DoubleTensor]
Shape: [2, 2]
[[-3, -2],
 [ 0,  1]]
```

### `ceil`()

Rounds the tensor elements to the least succeeding integers.

```lua
> z = tensor.DoubleTensor{{-2.25, -1.75}, {0.5, 1.0}}
> =z
[tensor.DoubleTensor]
Shape: [2, 2]
[[-2.25, -1.75],
 [  0.5,     1]]
> =z:ceil()
[tensor.DoubleTensor]
Shape: [2, 2]
[[-2, -1],
 [ 1,  1]]
```

### `round`()

Rounds the tensor elements to the closest integers, defaulting to the away from
zero in case of the value being equidistant.

```lua
> z = tensor.DoubleTensor{{-2.25, -1.75}, {0.5, 1.0}}
> =z
[tensor.DoubleTensor]
Shape: [2, 2]
[[-2.25, -1.75],
 [  0.5,     1]]
> =z:round()
[tensor.DoubleTensor]
Shape: [2, 2]
[[-2, -2],
 [ 1,  1]]
```

### `clamp`(\[`minValue`\], \[`maxValue`\])

Clamps the tensor elements to range \[`minValue`, `maxValue`\]. If either value
is `nil` then the values are not clamped in that direction.

`minValue` must not exceed `maxValue`.

```lua
> z = tensor.DoubleTensor(2, 3)
> z:fill{-500, 25, 500}
> =z
[tensor.DoubleTensor]
Shape: [2, 3]
[[-500,   25,  500],
 [-500,   25,  500]]

> =z:clone():clamp(0, 255)
[tensor.DoubleTensor]
Shape: [2, 3]
[[  0,  25, 255],
 [  0,  25, 255]]

> =z:clone():clamp(0)
[tensor.DoubleTensor]
Shape: [2, 3]
[[  0,  25, 500],
 [  0,  25, 500]]

> =z:clone():clamp(nill, 255)
[tensor.DoubleTensor]
Shape: [2, 3]
[[-500,   25,  255],
 [-500,   25,  255]]

> -- noop.
> =z:clone():clamp()
[tensor.DoubleTensor]
Shape: [2, 3]
[[-500,   25,  500],
 [-500,   25,  500]]
```

## Layout Operations

There are operations that do not affect the underlying storage of the tensor,
but create new views on the same underlying data.

### `transpose`(*dim1*, *dim2*)

Swaps `dim1` and `dim2` of the Tensor.

```lua
> z = tensor.DoubleTensor{{1, 2}, {3, 4}, {5, 6}}
> =z
[tensor.DoubleTensor]
Shape: [3, 2]
[[1, 2],
 [3, 4],
 [5, 6]]
> =z:transpose(1, 2)
[tensor.DoubleTensor]
Shape: [2, 3]
[[1, 3, 5],
 [2, 4, 6]]
```

### `select`(*dim*, *index*) and `__call`(*index1*, *index2*, ...)

Returns a slice of the tensor at index in dim.

```lua
> z = tensor.DoubleTensor{{1, 2}, {3, 4}, {5, 6}}
> =z
[tensor.DoubleTensor]
Shape: [3, 2]
[[1, 2],
 [3, 4],
 [5, 6]]
> =z:select(2, 2)
[tensor.DoubleTensor]
Shape: [3]
[2, 4, 6]
```

`z(index1, index2, ..., indexN)` is equivalent to `z:select(1, index1):select(1,
index2): ... :select(1, indexN)`.

```lua
> z = tensor.DoubleTensor{{1, 2}, {3, 4}, {5, 6}}
> =z
[tensor.DoubleTensor]
Shape: [3, 2]
[[1, 2],
 [3, 4],
 [5, 6]]
> =z(2)
[tensor.DoubleTensor]
Shape: [2]
[3, 4]
> =z:select(1, 2)
[tensor.DoubleTensor]
Shape: [2]
[3, 4]
> =z(3, 1)
[tensor.DoubleTensor]
5
```

### `narrow`(*dim*, *index*, *size*)

Returns a new Tensor which is a narrowed version of the current one. The
dimension `dim` is narrowed from `index` to `index + size - 1`.

```lua
> z = tensor.DoubleTensor{range={16}}:reshape{4, 4}
> =z
[tensor.DoubleTensor]
Shape: [4, 4]
[[ 1,  2,  3,  4],
 [ 5,  6,  7,  8],
 [ 9, 10, 11, 12],
 [13, 14, 15, 16]]
> =z:narrow(2, 2, 3)
[tensor.DoubleTensor]
Shape: [4, 3]
[[ 2,  3,  4],
 [ 6,  7,  8],
 [10, 11, 12],
 [14, 15, 16]]
> =z:narrow(1, 3, 2)
[tensor.DoubleTensor]
Shape: [2, 4]
[[ 9, 10, 11, 12],
 [13, 14, 15, 16]]
```

### `reverse`(*dim*)

Returns a new Tensor with a reversed the dimension `dim`. The dimension `dim`
must be a valid dimension.

```lua
> z = tensor.DoubleTensor{range={12}}:reshape{3, 4}
> =z
[tensor.DoubleTensor]
Shape: [3, 4]
[[ 1,  2,  3,  4],
 [ 5,  6,  7,  8],
 [ 9, 10, 11, 12]]
> =z:reverse(1)  -- Reverse major dimension.
[tensor.DoubleTensor]
Shape: [3, 4]
[[ 9, 10, 11, 12],
 [ 5,  6,  7,  8],
 [ 1,  2,  3,  4]]
> =z:reverse(2)  -- Reverse minor dimension
[tensor.DoubleTensor]
Shape: [3, 4]
[[ 4,  3,  2,  1],
 [ 8,  7,  6,  5],
 [12, 11, 10,  9]]
> =z:reverse(1):reverse(2)  -- Rotate 180 degrees.
[tensor.DoubleTensor]
Shape: [3, 4]
[[12, 11, 10,  9],
 [ 8,  7,  6,  5],
 [ 4,  3,  2,  1]]
```

This can also be combined with transpose to create 90 degree rotations.

```lua
> z = tensor.DoubleTensor{range={12}}:reshape{3, 4}
> =z
[tensor.DoubleTensor]
Shape: [3, 4]
[[ 1,  2,  3,  4],
 [ 5,  6,  7,  8],
 [ 9, 10, 11, 12]]
> =z:transpose(1, 2):reverse(1)  -- Rotate 90 degrees counterclockwise.
[tensor.DoubleTensor]
Shape: [4, 3]
[[ 4,  8, 12],
 [ 3,  7, 11],
 [ 2,  6, 10],
 [ 1,  5,  9]]
> =z:transpose(1, 2):reverse(2)  -- Rotate 90 degrees clockwise.
[tensor.DoubleTensor]
Shape: [4, 3]
[[ 9,  5,  1],
 [10,  6,  2],
 [11,  7,  3],
 [12,  8,  4]]
```

## Reduce Operations

In the following operations the tensor is reduced along dimension `dim`, i.e.
the rank of the resulting tensor is one less than the rank of the operand and
the tensor returned will have the shape of the original tensor without the
selected dimension.

### `argMax(*dim*)`

Returns a tensor containing the max value index across given dimension.

```lua
> doubles = tensor.DoubleTensor{{1, 2}, {33, 11}, {222, 333}}
> =doubles:argMax(2)  -- Row max indices.
[tensor.DoubleTensor]
Shape: [3]
[2, 1, 2]
> =doubles:argMax(1)  -- Col max indices.
[tensor.Int64Tensor]
Shape: [2]
[3, 3]
```

### `argMin(*dim*)`

Returns a tensor containing the min value index across given dimension.

```lua
> doubles = tensor.DoubleTensor{{1, 2}, {33, 11}, {222, 333}}
> =doubles:argMin(2)  -- Row min indices.
[tensor.DoubleTensor]
Shape: [3]
[1, 2, 1]
> =doubles:argMin(1)  -- Col min indices.
[tensor.Int64Tensor]
Shape: [2]
[1, 1]
```

### `max(*dim*)`

Returns a tensor containing the largest value across given dimension.

```lua
> doubles = tensor.DoubleTensor{{1, 2}, {33, 11}, {222, 333}}
> =doubles:max(2)  -- Row min.
[tensor.DoubleTensor]
Shape: [3]
[  3,  33, 333]
> =doubles:max(1)   -- Col min.
[tensor.DoubleTensor]
Shape: [3]
[111, 222, 333]
```

### `min([*dim*])`

Returns a tensor containing the smallest value across given dimension.

```lua
> doubles = tensor.DoubleTensor{{1, 2}, {33, 11}, {222, 333}}
> =doubles:min(2)  -- Row min.
[tensor.DoubleTensor]
Shape: [3]
[  1,  11, 111]
> =doubles:min(1)  -- Col min.
[tensor.DoubleTensor]
Shape: [3]
[1, 2, 3]
```

## Accumulate Operations

The elements are accumulated under the respective binary operation. Each value
is converted to a double before the operation is applied to help prevent
overflow.

### `sum`()

Calculates the sum of all the elements. Returns 0 of the tensor is empty.

```lua
> z = tensor.DoubleTensor{3, 3, 3, 3}
> assert(z:sum() == 4 * 3)
```

Elements are converted to double before accumulation:

```lua
> z = tensor.ByteTensor{255, 255}
> assert(z:sum() == 255 + 255)
```

### `product`()

Calculates the product of all the elements. Returns 1 of the tensor is empty.

```lua
> z = tensor.DoubleTensor{3, 3, 3, 3}
> assert(z:product() == 3 * 3 * 3 * 3)
```

Elements are converted to double before accumulation:

```lua
> z = tensor.ByteTensor{2, 4, 8, 16}
> assert(z:product() == 2 * 4 * 8 * 16)
```

### `lengthSquared`()

Calculates the sum of all the elements squared. Returns 0 of the tensor is
empty.

```lua
> z = tensor.DoubleTensor{1, 2, 3, 4}
> assert(z:lengthSquared() == 1 + 2 * 2 + 3 * 3 + 4 * 4)
```

Elements are converted to double before accumulation:

```lua
> z = tensor.ByteTensor{5, 25, 100}
> assert(z:lengthSquared() == 5 * 5 + 25 * 25 + 100 * 100)
```

### `argMaxElement()`

Returns index of largest value.

```lua
> doubles = tensor.DoubleTensor{{1, 2}, {33, 11}, {222, 333}}
> =doubles:argMaxElement()
3 2
```

### `argMinElement()`

Returns index of smallest value.

```lua
> doubles = tensor.DoubleTensor{{1, 2}, {33, 11}, {222, 333}}
> =doubles:argMinElement()
1   1
```

### `maxElement()`

Returns the largest value.

```lua
> doubles = tensor.DoubleTensor{{1, 2, 3}, {33, 11, 22}, {222, 333, 111}}
> =doubles:maxElement()
333
```

### `minElement()`

Returns the smallest value.

```lua
> doubles = tensor.DoubleTensor{{1, 2, 3}, {33, 11, 22}, {222, 333, 111}}
> =doubles:minElement()
1
```

## Scalar Operations

Scalar operations work on the tensor object and a single scalar argument value.
The respective element-wise binary operation is applied to pairs of each tensor
element and the scalar value. If an array is passed then it becomes equivalent
to:

```lua
fillValue = {1, 2}
myTensor:<scalarOp>(fillValue)
-- Equivalent to:
for i, value in ipairs(fillValue) do
  myTensor:Select(#myTensor:shape(), i):<scalarOp>(value)
end
```

Where <scalarOp> is any of the functions defined below.

### `fill`(*value*)

Fills the tensor with value.

```lua
> z = tensor.DoubleTensor(4, 4)
> =z:fill(7)
[tensor.DoubleTensor]
Shape: [4, 4]
[[7, 7, 7, 7],
 [7, 7, 7, 7],
 [7, 7, 7, 7],
 [7, 7, 7, 7]]
```

Table form:

```lua
> z = tensor.DoubleTensor(4, 4)
> =z:fill{1, 2, 3, 4}
[tensor.DoubleTensor]
Shape: [4, 4]
[[1, 2, 3, 4],
 [1, 2, 3, 4],
 [1, 2, 3, 4],
 [1, 2, 3, 4]]
```

### `add`(*value*)

```lua
> z = tensor.DoubleTensor(4, 4):fill(5)
> =z:add(1)
[tensor.DoubleTensor]
Shape: [4, 4]
[[6, 6, 6, 6],
 [6, 6, 6, 6],
 [6, 6, 6, 6],
 [6, 6, 6, 6]]
```

Table form:

```lua
> z = tensor.DoubleTensor(4, 4):fill(5)
> =z:add{1, 2, 3, 4}
[tensor.DoubleTensor]
Shape: [4, 4]
[[6, 7, 8, 9],
 [6, 7, 8, 9],
 [6, 7, 8, 9],
 [6, 7, 8, 9]]
```

### `mul`(*value*)

```lua
> z = tensor.DoubleTensor(4, 4):fill(0.5)
> =z:mul(4)
[tensor.DoubleTensor]
Shape: [4, 4]
[[2, 2, 2, 2],
 [2, 2, 2, 2],
 [2, 2, 2, 2],
 [2, 2, 2, 2]]
```

Table form:

```lua
> z = tensor.DoubleTensor(4, 4):fill(0.5)
> =z:mul{4, 6, 8, 10}
[tensor.DoubleTensor]
Shape: [4, 4]
[[2, 3, 4, 5],
 [2, 3, 4, 5],
 [2, 3, 4, 5],
 [2, 3, 4, 5]]
```

### `sub`(*value*)

```lua
> z = tensor.DoubleTensor(4, 4):fill(2)
> =z:sub(4)
[tensor.DoubleTensor]
Shape: [4, 4]
[[-2, -2, -2, -2],
 [-2, -2, -2, -2],
 [-2, -2, -2, -2],
 [-2, -2, -2, -2]]
```

Table form:

```lua
> z = tensor.DoubleTensor(4, 4):fill(2)
> =z:sub{4, 5, 6, 7}
[tensor.DoubleTensor]
Shape: [4, 4]
[[-2, -3, -4, -5],
 [-2, -3, -4, -5],
 [-2, -3, -4, -5],
 [-2, -3, -4, -5]]
```

### `div`(*value*)

```lua
> z = tensor.DoubleTensor(4, 4):fill(-2)
> =z:div(2)
[tensor.DoubleTensor]
Shape: [4, 4]
[[-1, -1, -1, -1],
 [-1, -1, -1, -1],
 [-1, -1, -1, -1],
 [-1, -1, -1, -1]]
```

Table form:

```lua
> z = tensor.DoubleTensor(4, 4):fill(12)
> =z:div{1, 2, 3, 4}
[tensor.DoubleTensor]
Shape: [4, 4]
[[12,  6,  4,  3],
 [12,  6,  4,  3],
 [12,  6,  4,  3],
 [12,  6,  4,  3]]
```

## Component Operations

Component operations work on two tensors (the object and the argument) that have
the same *number* of elements, but the tensor *shapes* are ignored. The
respective element-wise binary operation is applied consecutively to pairs of
tensor elements visited in their respective layout order.

It is an error if the two tensors do not have the same number of elements or the
same type.

```lua
> bt = tensor.ByteTensor{{1, 2}, {3, 4}, {5, 6}}
> =bt
[tensor.ByteTensor]
Shape: [3, 2]
[[1, 2],
 [3, 4],
 [5, 6]]
> bt2 = tensor.ByteTensor{{1, 2, 3}, {4, 5, 6}}
> =bt2
[tensor.ByteTensor]
Shape: [2, 3]
[[1, 2, 3],
 [4, 5, 6]]
```

### `cmul`(*value*)

```lua
> bt = tensor.ByteTensor{{1, 2}, {3, 4}, {5, 6}}
> bt2 = tensor.ByteTensor{{1, 2, 3}, {4, 5, 6}}
> =bt:cmul(bt2)
[tensor.ByteTensor]
Shape: [3, 2]
[[ 1,  4],
 [ 9, 16],
 [25, 36]]
```

### `cadd`(*value*)

```lua
> bt = tensor.ByteTensor{{1, 2}, {3, 4}, {5, 6}}
> bt2 = tensor.ByteTensor{{1, 2, 3}, {4, 5, 6}}
> =bt:cadd(bt2)
[tensor.ByteTensor]
Shape: [3, 2]
[[ 2,  4],
 [ 6,  8],
 [10, 12]]
```

### `cdiv`(*value*)

```lua
> bt = tensor.ByteTensor{{1, 2}, {3, 4}, {5, 6}}
> bt2 = tensor.ByteTensor{{1, 2, 3}, {4, 5, 6}}
> =bt:cdiv(bt2)
[tensor.ByteTensor]
Shape: [3, 2]
[[1, 1],
 [1, 1],
 [1, 1]]
```

### `csub`(*value*)

```lua
> bt = tensor.ByteTensor{{1, 2}, {3, 4}, {5, 6}}
> bt2 = tensor.ByteTensor{{1, 2, 3}, {4, 5, 6}}
> =bt:csub(bt2)
[tensor.ByteTensor]
Shape: [3, 2]
[[0, 0],
 [0, 0],
 [0, 0]]
```

### `copy`(*value*)

```lua
> bt = tensor.ByteTensor(3, 2)
> =bt
[tensor.ByteTensor]
Shape: [3, 2]
[[0, 0],
 [0, 0],
 [0, 0]]
> bt2 = tensor.ByteTensor{{1, 2, 3}, {4, 5, 6}}
> =bt:copy(bt2)
[tensor.ByteTensor]
Shape: [3, 2]
[[1, 2],
 [3, 4],
 [5, 6]]
```

## Accumulate Component Operations

Accumulate Component Operations work on two tensors (the object and the
argument) that have the same *number* of elements, but the tensor *shapes* are
ignored. The respective element-wise binary operation is applied consecutively
to pairs of tensor elements visited in their respective layout order. The
accumilation is applied to the result of each binary operation.

It is an error if the two tensors do not have the same number of elements or the
same type.

### `dot`(*value*)

The element-wise operation is multiply, the accumulation operation is add.
Returns 0 if both tensors are empty.

```lua
> z1 = tensor.DoubleTensor{1, 2, 3, 4}
> z2 = tensor.DoubleTensor{-1, 2, -3, 4}
> assert(z1:dot(z2) == -1 + 4 - 9 + 16)
```

Elements are converted to double before accumulation:

```lua
> z1 = tensor.ByteTensor{128, 64}
> z2 = tensor.ByteTensor{16, 64}
> assert(z1:dot(z2) == 128 * 16 + 64 * 64)
```

## Matrix operations

Matrix operations only apply to rank-2 tensors.

### `mmul`(*value*)

Matrix multiplication.

```lua
> at = tensor.FloatTensor{{1, 2, 3}, {4, 5, 6}}
> bt = tensor.FloatTensor{{1, 0}, {0, 2}}
> =bt:mmul(at)
[tensor.FloatTensor]
Shape: [2, 3]
[[ 1,  2,  3],
 [ 8, 10, 12]]
```

## Random operations

### `shuffle`(*gen*)

Shuffles the elements of a rank-1 tensor, using the permutation computed by
random bit generator 'gen'.

```lua
-- Possible output.
> random = require 'system.random'
> random:seed(0)
> =tensor.Int64Tensor{range={5}}:shuffle(random)
[tensor.Int64Tensor]
Shape: [5]
[5, 1, 3, 2, 4]
```

## Comparison

Two tensors are equal iff the type, shape and values are the same.

```lua
bt = tensor.ByteTensor{{1, 2}, {3, 4}, {5, 6}}

-- The same:
assert(bt == tensor.ByteTensor{{1, 2}, {3, 4}, {5, 6}})

-- Values are different:
assert(bt ~= tensor.ByteTensor{{1, 2}, {3, 4}, {7, 8}})

-- Shape is different:
assert(bt ~= tensor.ByteTensor{{1, 2, 3}, {4, 5, 6}})

-- Type is different:
assert(bt ~= tensor.Int32Tensor{{1, 2}, {3, 4}, {5, 6}})
```

## Ownership

A tensor may either own its storage (jointly with all copies made from it), or
otherwise it is merely a view on an existing piece of storage that it does not
control. In the latter case, the storage which the tensor is viewing may
disappear spontaneously, and the tensor becomes *invalid*. A tensor that owns
its storage can never become invalid.

You can check whether a valid tensor owns its storage by calling
`z:ownsStorage()`. You can create a tensor that definitely owns its storage by
cloning an existing tensor.

Any attempt to use an invalid tensor will result in a Lua error. There is (by
design) no direct way to check a tensor for validity, and library APIs should
document the guaranteed lifetime of non-owning tensors. In that case, you may
wish to clone the non-owning tensor if you need it for longer than the library
guarantees.

```lua
local my_api = {}

function my_api.call_me_with_a_tensor(z)
  my_api._valid_tensor = z:ownsStorage() and z or z:clone()
end

function my_api.call_me()
  return my_api._valid_tensor
end

return my_api
```

For debugging purposes, you may check for tensor validity using a pcall:

```lua
local my_api = {}

-- It is assumed some external code calls this function.
function my_api.call_me_with_a_tensor(z)
  my_api._maybe_invalid_copy = z
end

function my_api.call_me()
  if pcall(my_api._maybe_invalid_copy:type()) then
    return my_api._maybe_invalid_copy
  end
  return nil
end

return my_api
```
