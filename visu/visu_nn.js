#!/usr/bin/env node
/*
 * Nitrogram: visu_nn.js
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
// string as returned by the `to_string` method. 
// 

var VISU_WIDTH = 800;
var VISU_HEIGHT = 600;
var W_THRESHOLD = 1;
var DEFAULT_WMAX = 3;
var DEFAULT_WMIN = -3;

var PRE_RUN = true;

if(process.argv.length < 3) {
  console.log('Usage: `visu nn.out`');
  process.exit(0);
}


/* Extracts the NeuralN network data from the file passed as an argument and */
/* splits the string into an array of value string.                          */
var net_str = fs.readFileSync(path.resolve(process.argv[2])).toString();
var input = net_str.split(' ');

/* We extract the network structure and parameters first */
var B = [], W = [], L = [];
var d = parseInt(input.shift(), 10);

for(var l = 0; l < d; l ++) {
  L[l] = parseInt(input.shift(), 10);
};

var alpha = parseFloat(input.shift());
var beta = parseFloat(input.shift());
var bias = parseFloat(input.shift());

var Wmax = PRE_RUN ? 0 : DEFAULT_WMAX;
var Wmin = PRE_RUN ? 0 : DEFAULT_WMIN;


/* Next we evaluate Wmax and Wmin the maximum and minimum weight used across */
/* the network, to normalize the coloration of the edges of the network.     */
/* As this process can be lenghty, we have a DEFAULT_WMAX and a DEFAULT_WMIN */
/* variable to skip that step once run once.                                 */
if(PRE_RUN) {
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
  console.log('PRE-RUN:');
  console.log('alpha: ' + alpha);
  console.log('bias : ' + bias);
  console.log('Wmax : ' + Wmax);
  console.log('Wmin : ' + Wmin);
  process.exiT(0);
}

/* Finally we create an html file containing SVG data based on the network   */
/* weights. We use W_THRESHOLD to filter the nodes we want to display or not */
/* in order to keep the graph readable even when there are a lot of nodes.   */

console.log('<svg height="' + VISU_HEIGHT + '" width="' + VISU_WIDTH + '">');

for(var l = 0; l < L.length; l ++) {
  if(l > 0) {
    var x1 = Math.floor(VISU_WIDTH / (L.length - 1) * (l-1));
    var x2 = Math.floor(VISU_WIDTH / (L.length - 1) * l);
    for(var i = 0; i < L[l]; i ++) {
      var y2 = Math.floor(VISU_HEIGHT / (L[l] - 1) * i);
      var b = parseFloat(input.shift());
      for(var j = 0; j < L[l-1]; j++) {
        var y1 = Math.floor(VISU_HEIGHT / (L[l-1] - 1) * j);
        var w = parseFloat(input.shift());

        if(w > W_THRESHOLD || w < -W_THRESHOLD) {
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

