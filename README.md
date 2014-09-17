# NeuralN
### Powerful Neural Network for Node.js

NeuralN is a C++ Neural Network library for Node.js with multiple advantages
compared to existing solutions:
  - Works with extra large datasets (>1Go allowed by nodejs)
  - Multi-Threaded training available.

#### Large datasets

With Node.js and the V8, it is not possible to work with large datasets since
the maximum allowed memory is around 512MB for 32-bits machines and 1GB for
64-bits machines. When you are working with datasets of several gigabytes, it
fast becomes difficult to train you network with all your data.

NeuralN allows you to use datasets as big as your memory can contain.

#### Multi-Threaded

Working with large datasets increase the performances of your final network,
but the learning period can sometimes last several days or even weeks to obtain
good results.

With the multithread training method of NeuralN, you can simply divide the
needed time, by training your network simultaneously on different parts of your
dataset. The results of each iterations are then combined.

## Install

```
npm install neuraln
```

## How it works

```javascript
var NN = require('neuraln');

/* Create a neural network with 4 layers (2 hidden layers) */
var network = new NN([ 1, 4, 3, 1 ]);

/* Add points to the training set */
for(var i = -1; i < 1; i+=0.1) {
  network.train_set_add([ i ], [ Math.abs(Math.sin(i)) ]);
}

/* Mono thread (blocking) training */
network.train(0.005, 10000);

/* Multi thread (non-blocking) training */
network.mt_train(0.005, 10000, 20, 4, function(err) {
  // Network is trained
});

/* Run */
var result = network.run((Math.random() * 2) - 1);

/* Retrieve the network's string representation */
var string = network.to_string();
```

## Methods

```javascript
var network = new NN(layers, momentum, learning_rate, bias);
var network = new NN(string);
```

Instantiate a new network with the following parameters:
- `layers` is an array representing the layers of the network
- `momentum` is a number between 0 and 1. This parameter is optional and default to `0.3`
- `learning_rate` is a number. This parameter is optional and default to `0.1`
- `bias` is a number. This parameter is optional and default to `-1`

Or

- `string` a string from a previous network (using `to_string`)

```javascript
network.train_set_add(input, output);
```

Add a training data point with `input` and `output` being arrays of numbers.
`input` and `output` must contain as many values as the first and last layers'
number of neurons.

```javascript
network.train(target_error, max_iterations);
```

Train the network with the training set until the `target_error` or the
`max_iterations` has been reached.

```javascript
network.mt_train(target_error, max_iterations, step_size, threads, callback);
```

Train the network using the multi-threaded method. The first two parameters are
exactly the same as for `train`.
`step_size` represents the number of points of the training set to use by thread
at each iteration.
`threads` represents the number of threads to be used for the training.
`callback(err)` is called once the training is done.

```javascript
network.run(input)
```

Runs the given `input` throught the network and returns its `output`

```javascript
network.to_string()
```

Returns a string representation of the network in order to save and reload it
later

## Contact us

Feel free to contact us at `hello@totems.co`

## License

Distributed under the MIT License.

Copyright Teleportd Ltd. and other Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.