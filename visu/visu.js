#!/usr/bin/env node
/*
 * Nitrogram: visu.js
 *
 * (c) Copyright Teleportd Ltd. 2014, All rights reserved.
 *
 * @author: spolu
 *
 * @log:
 * 2014-09-17 spolu   Creation
 */
"use strict"

var path = require('path');
var fs = require('fs');


//
// The `visu` tool takes as argument a neural network and generates an html
// page rendering the neural network using SVG
//
// The `visu` command takes a filename as argument containing the neural network
// string as returned by the `to_string` method. And additional optional
// argument is a filename containing the network state during classification
// as returned by the `get_state` method.
//

var VISU_WIDTH = 800;
var VISU_HEIGHT = 600;

if(process.argv.length < 3) {
  console.log('Usage: `visu nn.out [nn.state]`');
  process.exit(0);
}

var net_str = fs.readFileSync(path.resolve(process.argv[2])).toString();
var input = net_str.split(' ');

//console.log(net_str);

var B = [], W = [], L = [];
var d = parseInt(input.shift(), 10);

for(var l = 0; l < d; l ++) {
  L[l] = parseInt(input.shift(), 10);
};

var alpha = parseFloat(input.shift());
var beta = parseFloat(input.shift());
var bias = parseFloat(input.shift());

var Wmax = 3;
var Wmin = -3;

var t = 0;

/*
for(var l = 0; l < L.length; l ++) {
  if(l > 0) {
    for(var i = 0; i < L[l]; i ++) {
      console.log('>> ' + l + ' ' + i + ' ' + Wmin + ' ' + Wmax);
      var b = parseFloat(input.shift());
      for(var j = 0; j < L[l-1]; j++) {
        var w = parseFloat(input.shift());
        Wmax = w > Wmax ? w : Wmax;
        Wmin = w < Wmin ? w : Wmin;
      }
    }
  }
}
*/

console.log('<svg height="' + VISU_HEIGHT + '" width="' + VISU_WIDTH + '">');
for(var l = 0; l < L.length; l ++) {
  if(l > 0) {
    var x1 = Math.floor(VISU_WIDTH / (L.length - 1) * (l-1));
    var x2 = Math.floor(VISU_WIDTH / (L.length - 1) * l);
    for(var i = 0; i < L[l]; i ++) {
      var y2 = Math.floor(VISU_HEIGHT / (L[l] - 1) * i);
      //console.log('>> ' + l + ' ' + i);
      var b = parseFloat(input.shift());
      for(var j = 0; j < L[l-1]; j++) {
        var y1 = Math.floor(VISU_HEIGHT / (L[l-1] - 1) * j);
        //console.log('>>> ' + j);
        var w = parseFloat(input.shift());
        if(w > 1 || w < -1) {
          var r = 255;
          var b = 255;
          if(w > 0) {
            b = Math.max(Math.floor(255 - (w/3 * 255), 0));
          }
          if(w < 0) {
            r = Math.max(Math.floor(255 - (-w/3 * 255), 0));
          }
          console.log('<line x1="' + x1 + 
                      '" y1="' + y1 + 
                      '" x2="' + x2 + 
                      '" y2="' + y2 + 
                      '" style="stroke:rgba(' + r + ',255,' + b + ', 0.5);stroke-width:1" />');
        }
      }
    }
  }
}
console.log('</svg>');

console.log(alpha);
console.log(bias);
console.log(L[1]);

/*
for(var l = 0; l < L.length; l ++) {
  if(l > 0) {
    var x1 = Math.floor(VISU_WIDTH / (L.length - 1) * (l-1));
    var x2 = Math.floor(VISU_WIDTH / (L.length - 1) * l);
    for(var i = 0; i < L[l]; i ++) {
      y2 = Math.floor(VISU_HEIGHT / (L[l] - 1) * i);
      for(var j = 0; j < L[l-1]; j++) {
        y1 = Math.floor(VISU_HEIGHT / (L[l-1] - 1) * j);
        var c = Math.floor(255 * (W[l][i][j] - Wmin) / (Wmax - Wmin));
        console.log('<line x1="" y1="" x2="" y2="" style="stroke:rgb(' + c + ',0,0);stroke-width:1" />');
      }
    }
  }
  //<line x1="0" y1="0" x2="200" y2="200" style="stroke:rgb(255,0,0);stroke-width:2" />
}
*/
