const path = require("path");
const modelPathRoot = "./models";
const tf = require("@tensorflow/tfjs-node");

// import '@tensorflow/tfjs-node';
// import * as canvas from 'canvas';
const canvas = require('canvas');
const faceapi = require('face-api.js');
// import * as faceapi from 'face-api.js';
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

let optionsSSDMobileNet;

async function image(file) {
  const decoded = tf.node.decodeImage(file);
  const casted = decoded.toFloat();
  const result = casted.expandDims(0);
  decoded.dispose();
  casted.dispose();
  return result;
}

async function detect(tensor) {
  const result = await faceapi.detectAllFaces(tensor, optionsSSDMobileNet);
  return result;
}

function hello() {
    return "hello";
}

const main = async () => {
    // await faceapi.tf.setBackend("tensorflow");
    // await faceapi.tf.enableProdMode();
    // await faceapi.tf.ENV.set("DEBUG", false);
    // await faceapi.tf.ready();

    // console.log(
    //     `Version: TensorFlow/JS ${faceapi.tf?.version_core} FaceAPI ${faceapi.version.faceapi
    //     } Backend: ${faceapi.tf?.getBackend()}`
    // );

    console.log("Loading FaceAPI models");
    const modelPath = path.join(__dirname, modelPathRoot);
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({
        minConfidence: 0.5,
    });

    const tensor = await image(file);
    const result = await detect(tensor);
    console.log("Detected faces:", result.length);

    tensor.dispose();

    return result;
}

module.exports = {
    detect: main,
    hello,
};