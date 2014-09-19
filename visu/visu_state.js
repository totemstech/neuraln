#!/usr/bin/env node
/*
 * Nitrogram: visu_state.js
 *
 * (c) Copyright Teleportd Ltd. 2014, All rights reserved.
 *
 * @author: spolu
 *
 * @log:
 * 2014-09-18 spolu   Creation
 */
"use strict"

var path = require('path');
var fs = require('fs');

//
// The `visu_state` tool takes as argument the state of a neural network right
// after classificiation and generates an html page rendering the classification 
// state using SVG
//
// The `visu_state` command takes a filename as argument containing the neural 
// network state string as returned by the `get_state` method. 
// 

var VISU_WIDTH = 800;
var VISU_HEIGHT = 600;
var W_THRESHOLD = 0.3;
var DEFAULT_WMAX = 10;
var DEFAULT_WMIN = -10;

var PRE_RUN = false;

if(process.argv.length < 3) {
  console.log('Usage: `visu user_id.state`');
  process.exit(0);
}


/* Extracts the NeuralN network state from the file passed as an argument and */
/* splits the string into an array of value string.                           */
var state_str = fs.readFileSync(path.resolve(process.argv[2])).toString();
var input = state_str.split(' ');

/* We extract the network structure and parameters first */
var W = [], L = [];
var d = parseInt(input.shift(), 10);

for(var l = 0; l < d; l ++) {
  L[l] = parseInt(input.shift(), 10);
  W[l] = [];
};

var type = input.shift();
if(type !== 'compact') {
  console.log('Compact format supported only');
}

var Wmax = PRE_RUN ? 0 : DEFAULT_WMAX;
var Wmin = PRE_RUN ? 0 : DEFAULT_WMIN;

/* Next we evaluate Wmax and Wmin the maximum and minimum weight used across */
/* the network, to normalize the coloration of the edges of the network.     */
/* As this process can be lenghty, we have a DEFAULT_WMAX and a DEFAULT_WMIN */
/* variable to skip that step once run once.                                 */
if(PRE_RUN) {
  while(input.length > 0) {
    var l = parseInt(input.shift(), 10);
    var i = parseInt(input.shift(), 10);
    var j = parseInt(input.shift(), 10);
    var w = parseFloat(input.shift());
    Wmax = w > Wmax ? w : Wmax;
    Wmin = w < Wmin ? w : Wmin;
  }

  console.log('PRE-RUN:');
  console.log('Wmax : ' + Wmax);
  console.log('Wmin : ' + Wmin);
  process.exit(0);
}

/* Finally we create an html file containing SVG data based on the network   */
/* weights. We use W_THRESHOLD to filter the nodes we want to display or not */
/* in order to keep the graph readable even when there are a lot of nodes.   */

console.log('<svg height="' + VISU_HEIGHT + '" width="' + VISU_WIDTH + '">');

while(input.length > 0) {
  if(l <= 0) {
    conosole.log('Less than 0');
  }
  var l = parseInt(input.shift(), 10);
  var i = parseInt(input.shift(), 10);
  var j = parseInt(input.shift(), 10);
  var w = parseFloat(input.shift());

  var x1 = Math.floor(VISU_WIDTH / (L.length - 1) * (l-1));
  var x2 = Math.floor(VISU_WIDTH / (L.length - 1) * l);
  var y2 = Math.floor(VISU_HEIGHT / (L[l] - 1) * i);
  var y1 = Math.floor(VISU_HEIGHT / (L[l-1] - 1) * j);

  if(w > W_THRESHOLD || w < -W_THRESHOLD) {
    var r = 255;
    var g = 255;
    var b = 255;
    if(w > 0) {
      r = Math.max(Math.floor(255 - (w/Wmax * 255), 0));
      g = Math.max(Math.floor(255 - (w/Wmax * 255), 0));
    }
    if(w < 0) {
      b = Math.max(Math.floor(255 - (w/Wmin * 255), 0));
      g = Math.max(Math.floor(255 - (w/Wmin * 255), 0));
    }
    console.log('<line x1="' + x1 + 
                '" y1="' + y1 + 
                '" x2="' + x2 + 
                '" y2="' + y2 + 
                '" style="stroke:rgba(' + r + ',' + g + ',' + b + ', 0.5);stroke-width:1" />');
  }
}

console.log('</svg>');

