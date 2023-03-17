var fs = require('fs');//import file system package

console.log('server is starting');//telling a server is starting
var express = require('express');

var app = express();
var server = app.listen(3000, listening); //listening for any requests

function listening() {

  console.log("Listening..."); //telling computer it is listening
}
app.use(express.static('website')); //use expresses ability to host static files public is just a name jpg files text files ...

//when a user enters the url it requests data

//question mark makes the variable optional
