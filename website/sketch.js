let x_vals = [];
let y_vals = [];
let xx_vals = [];
var drg = false;
var slider
var stage = -1;
var xy = 1;
var range = 3.141592653589793238464;
let img;
let used = false;
let yss;
let xss;

function preload()  {
  img = loadImage('has.png');
}
//let a;


const learningRate = 2;//0.9
const adm1 = 0.4;
const adm2 = 0.4;
const delt = 0.4;
const curveX = [];
for(let i = -(range); i < (range)+.05; i+=0.05)  {
  curveX.push(i);//adds values from -1 to 1 into an array
}
const curveY = [];
for(let i = -(range); i < (range)+.05; i+=0.05)  {
  curveY.push(i);//adds values from -1 to 1 into an array
}

//const optimizer = tf.train.adam(learningRate,adm1,adm2,delt); //scastic gradient decent
const optimizer = tf.train.adam(learningRate);
function setup() {
  var button = select('#reset');
  button.mousePressed(setupvars);
  var button2 = select('#stag');
  button2.mousePressed(moveStage);
  var button3 = select('#XY');
  button3.mousePressed(xymult);
  slider = document.getElementById("myRange");
  createCanvas(800,800);
  background(0);
  setupvars();
  //image(img,0,0,800,800);
  ///-------
  //these must be kept so no removel
}
function moveStage()  {//what??
  background(0);
  for(let i = 0; i < x_vals.length; i++) {
    xx_vals[i] = (i*(range*2/x_vals.length))-range;
  }
  stage*=-1;
}
function xymult() {
  xy *= -1;
  cos_vals = [];
  sin_vals = [];
  for(var i = 0; i < 101; i++)  {
    cos_vals[i] = tf.variable(tf.scalar(random(-1,1)));
    sin_vals[i] = tf.variable(tf.scalar(random(-1,1)));
  }
}

function setupvars()  {
  x_vals = [];
  y_vals = [];
  xx_vals = [];
  cos_vals = [];
  sin_vals = [];
  for(var i = 0; i < 101; i++)  {
    cos_vals[i] = tf.variable(tf.scalar(random(-1,1)));
    sin_vals[i] = tf.variable(tf.scalar(random(-1,1)));
  }
  a = tf.variable(tf.scalar(1));
  slider.value = 0;
}


function predict(x)  {
  document.getElementById("demo").innerHTML = slider.value;
  var equ = "f(x)=";
  var tx = "";
  var s = parseFloat(cos_vals[0].toString().replace(/Tensor/,""));
  s = (Math.round(s*100))/100;
  if(s < 0)tx = s.toString();
  else tx = "+"+s.toString();
  if(tx.length == 4) tx += "00";
  if(tx.length == 5) tx += "0";
  equ += tx;
  const xs = tf.tensor1d(x);
  //console.log(slider.value);
  var ys = xs.mul(0).add(cos_vals[0]);
  for(var i = 1; i < slider.value; i++) {
    ys = ys
    .add(xs.mul(i).add(sin_vals[i+50]).sin().mul(sin_vals[i]))
    .add(xs.mul(i).add(cos_vals[i+50]).cos().mul(cos_vals[i]));
    s = parseFloat(cos_vals[i].toString().replace(/Tensor/,""));
    s = (Math.round(s*1000))/1000;
    tx = s.toString();
    if(s < 0)tx = s.toString();
    else tx = "+"+s.toString();
    if(tx.length == 3) tx += ".000";
    if(tx.length == 4) tx += "00";
    if(tx.length == 5) tx += "0";
    tx = tx.slice(0,6);
    var h = parseFloat(cos_vals[i+50].toString().replace(/Tensor/,""));
    h = (Math.round(h*1000))/1000;
    var b = h.toString();
    if(h < 0)b = h.toString();
    else b = "+"+h.toString();
    if(b.length == 3) b += ".000";
    if(b.length == 4) b += "00";
    if(b.length == 5) b += "0";
    b = b.slice(0,6);
    equ += tx+'*cos('+i.toString()+'*x'+b+')';
    //equ += ',"'+tx+b+'"';

    //equ += ',"'+s+b+'"';
    s = parseFloat(sin_vals[i].toString().replace(/Tensor/,""));
    s = (Math.round(s*1000))/1000;
    tx = s.toString();
    if(s < 0)tx = s.toString();
    else tx = "+"+s.toString();
    if(tx.length == 3) tx += ".000";
    if(tx.length == 4) tx += "00";
    if(tx.length == 5) tx += "0";
    tx = tx.slice(0,6);
    h = parseFloat(sin_vals[i+50].toString().replace(/Tensor/,""));
    h = (Math.round(h*1000))/1000;
    b = h.toString();
    if(h < 0)b = h.toString();
    else b = "+"+h.toString();
    if(b.length == 3) b += ".000";
    if(b.length == 4) b += "00";
    if(b.length == 5) b += "0";
    b = b.slice(0,6);
    //equ += ',"'+tx+b+'"';
    equ += ''+tx+'*sin('+i.toString()+'*x'+b+')';
    //equ += ',"'+s+b+'"';
  }
  document.getElementById("equation").innerHTML = equ;
  return ys
}

function loss(pred, labels) {
  console.log(pred.sub(labels).square().mean());
  return pred.sub(labels).square().mean();
  //get error (guesses value - actual value)^2
  //(pred, lables) => pred.sub(labels).square().mean();
}

function draw() {
  if(stage == -1){
    //image(img,0,0,600,600);
  }
  if(!drg && stage == 1 && xy == 1)  {
    background(0);
    tf.tidy(()  =>  {
      if(xx_vals.length > 0)  {
        const ys = tf.tensor1d(x_vals);
        optimizer.minimize(() => loss(predict(xx_vals),ys));
      }
    });
    const ys = tf.tidy(() => predict(curveX));
    let curveY = ys.dataSync();
    ys.dispose();//gets ride of ys to save space
    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);
    for(let i = 0; i < curveX.length; i+=1)  {
      //let px = map(curveX[i],-range,range,0,width);
      //let py = map(curveY[i],-range,range,height,0);
      let px = map(curveX[i],-range,range,0,width);
      let py = map(curveY[i],-range,range,height,0);
      vertex(px,py);
    }
    endShape();
    stroke(255);
    strokeWeight(4);
    for(let i = 0; i < x_vals.length; i++)  {
      if(i ==1) {
        stroke(255,0,0);
      }
      else stroke(255);
      let px = map(xx_vals[i],-range,range,0,width);
      let py = map(x_vals[i],-range,range,height,0);
      point(px,py);
    }
  //console.log(tf.memory().numTensors);
  }
  if(!drg && stage == 1 && xy == -1)  {
    background(0);
    tf.tidy(()  =>  {
      if(xx_vals.length > 0)  {
        const ys = tf.tensor1d(y_vals);
        optimizer.minimize(() => loss(predict(xx_vals),ys));
      }
    });
    const ys = tf.tidy(() => predict(curveX));
    let curveY = ys.dataSync();
    ys.dispose();//gets ride of ys to save space
    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);
    for(let i = 0; i < curveX.length; i+=1)  {
      let px = map(curveX[i],-range,range,0,width);
      let py = map(curveY[i],-range,range,height,0);
      vertex(px,py);
    }
    endShape();
    stroke(255);
    strokeWeight(4);
    for(let i = 0; i < y_vals.length; i++)  {
      if(i ==1) {
        stroke(255,0,0);
      }
      else stroke(255);
      let px = map(xx_vals[i],-range,range,0,width);
      let py = map(y_vals[i],-range,range,height,0);
      point(px,py);
    }
  //console.log(tf.memory().numTensors);
  }
}



function mousePressed() {
  if(mouseX >0 && mouseX< 800 && mouseY >0 && mouseY< 800)drg = true;
}


function mouseReleased() {
  drg = false;
}


function mouseDragged()  {

  if(drg && stage == -1 && !used) {
    stroke(255);
    strokeWeight(4);
    /*for(let i = 0; i < xss.length; i++) {
      let x = map(xss[i],0,width,-range,range);
      let y = map(yss[i],0,height,range,-range);
      point(xss[i],yss[i]);
      x_vals.push(x);
      y_vals.push(y);
      used = true;
    }*/

    let x = map(mouseX,0,width,-range,range);
    let y = map(mouseY,0,height,range,-range);
    stroke(255);
    point(mouseX,mouseY);
    x_vals.push(x);
    y_vals.push(y);

  }
}
