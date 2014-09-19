// Copyright Teleportd Ltd. and other Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.
var nn = require('./build/Release/nn.node');

module.exports = function(layers, momentum, learning_rate, bias) {
  var network = new nn.NN(layers, momentum, learning_rate, bias);

  var test_value = function(fn, value) {
    if(fn(value))
      throw new Error('Bad string format');
  }

  return {
    set_log: function(status) {
      return network.set_log(status);
    },
    train_set_add:function(input, output) {
      return network.train_set_add(input, output);
    },
    train: function(options, callback) {
      var target_error = 0.01;
      var iterations = 20000;
      var step_size = 100;
      var threads = 4;

      if(typeof options.target_error === 'number')
        target_error = options.target_error;
      if(typeof options.iterations === 'number')
        iterations = options.iterations;
      if(typeof options.step_size === 'number')
        step_size = options.step_size;
      if(typeof threads === 'number')
        threads = options.threads;

      if(options.multithread) {
        return network.mt_train(target_error, iterations, step_size, threads, callback);
      }
      else {
        network.train(target_error, iterations);
        if(typeof callback === 'function')
          return callback();
      }
    },
    run: function(input) {
      return network.run(input);
    },
    to_string: function() {
      return network.to_string();
    },
    get_state: function(compact) {
      return network.get_state(compact);
    },
    to_json: function() {
      var values = network.to_string().split(' ');
      var json = {};

      var nr_layers = parseInt(values.shift(), 10);
      test_value(isNaN, nr_layers);

      json.layers = [];
      for(var i = 0; i < nr_layers; i++) {
        json.layers[i] = parseInt(values.shift(), 10);
        test_value(isNaN, json.layers[i]);
      }

      json.momentum = parseFloat(values.shift());
      test_value(isNaN, json.momentum);
      json.learning_rate = parseFloat(values.shift());
      test_value(isNaN, json.learning_rate);
      json.bias = parseFloat(values.shift());
      test_value(isNaN, json.bias);

      json.biases = [];
      json.weights = [];
      for(var l = 0; l < nr_layers; l++) {
        json.biases[l] = json.biases[l] || [];
        json.weights[l] = json.weights[l] || [];

        for(var i = 0; i < json.layers[l]; i++) {
          if(l > 0) {
            json.biases[l][i] = parseFloat(values.shift());
            test_value(isNaN, json.biases[l][i]);

            for(var j = 0; j < json.layers[l-1]; j++) {
              json.weights[l][i] = json.weights[l][i] || [];
              json.weights[l][i][j] = parseFloat(values.shift());
              test_value(isNaN, json.weights[l][i][j]);
            }
          }
        }
      }

      return json;
    },
    get_state_json: function() {
      var values = network.get_state(true).split(' ');
      var json = {};

      var nr_layers = parseInt(values.shift(), 10);
      test_value(isNaN, nr_layers);

      json.layers = [];
      for(var i = 0; i < nr_layers; i++) {
        json.layers[i] = parseInt(values.shift(), 10);
        test_value(isNaN, json.layers[i]);
      }

      var type = values.shift();
      test_value(function(v) { return (typeof v !== 'string' ||
                                       !(v === 'full' || v === 'compact')) }, type);

      json.values = [];
      while(values.length > 0) {
        var l = parseInt(values.shift());
        test_value(isNaN, l);
        var i = parseInt(values.shift());
        test_value(isNaN, i);
        var j = parseInt(values.shift());
        test_value(isNaN, j);
        var value = parseFloat(values.shift());
        test_value(isNaN, value);

        json.values[l] = json.values[l] || [];
        json.values[l][i] = json.values[l][i] || [];
        json.values[l][i][j] = value;
      }

      return json;
    }
  }
};
