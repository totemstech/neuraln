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
module.exports = nn.NN;

/******************************************/
/*                HELPERS                 */
/******************************************/
var test_value = function(fn, value) {
  if(fn(value))
    throw new Error('Bad string format');
}

module.exports.string_to_json = function(string) {
  var values = string.split(' ');
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
};

module.exports.state_to_json = function(string) {
  var values = string.split(' ');
  var json = {};

  var nr_layers = parseInt(values.shift(), 10);
  test_value(isNaN, nr_layers);

  json.layers = [];
  for(var i = 0; i < nr_layers; i++) {
    json.layers[i] = parseInt(values.shift(), 10);
    test_value(isNaN, json.layers[i]);
  }

  json.values = [];
  for(var l = 0; l < nr_layers; l++) {
    json.values[l] = json.values[l] || [];
    for(var i = 0; i < json.layers[l]; i++) {
      json.values[l][i] = parseFloat(values.shift());
      test_value(isNaN, json.values[l][i]);
    }
  }

  return json;
};
