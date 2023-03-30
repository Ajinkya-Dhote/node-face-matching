const path = require("path");
const modelPathRoot = "./models";
// import { faceDetectionOptions } from './commons';

const saveFile = require("./commons/saveFile")


// import '@tensorflow/tfjs-node';
// import * as canvas from 'canvas';
const canvas = require('canvas');
const faceapi = require('face-api.js');
// import * as faceapi from 'face-api.js';
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

let optionsSSDMobileNet;



const loadModels = async (file) => {
  const modelsPath = "./models";
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsPath);
  console.info("loading model complete: ssdMobilenetv1");
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelsPath)
  console.info("loading model complete: faceLandmark68Net");
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelsPath)
  console.info("loading model complete: faceRecognitionNet");
}
const main = async (file) => {
  console.log("running main function");
  const minConfidence = 0.5

  await loadModels();
  const img = await canvas.loadImage(file);
  const detections = await faceapi.detectAllFaces(img, new faceapi.SsdMobilenetv1Options({ minConfidence }));
  const out = faceapi.createCanvasFromMedia(img);
  faceapi.draw.drawDetections(out, detections);

  saveFile('faceDetection.jpg', out.toBuffer('image/jpeg'));
  console.log('done, saved results to out/faceDetection.jpg');
  return detections;
}

const match = async (ref, query) => {
  const REFERENCE_IMAGE = ref;
  const QUERY_IMAGE = query;

  await loadModels();

  const minConfidence = 0.5;
  const faceDetectionOptions = new faceapi.SsdMobilenetv1Options({ minConfidence });

  const referenceImage = await canvas.loadImage(REFERENCE_IMAGE);
  const queryImage = await canvas.loadImage(QUERY_IMAGE);

  const resultsRef = await faceapi.detectSingleFace(referenceImage, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptor();

  const resultsQuery = await faceapi.detectSingleFace(queryImage, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptor();

  const faceMatcher = new faceapi.FaceMatcher(resultsRef);
  const bestMatch = faceMatcher.findBestMatch(resultsQuery.descriptor);
  return bestMatch;

  // const labels = faceMatcher.labeledDescriptors
  //   .map(ld => ld.label)
  // const refDrawBoxes = resultsRef
  //   .map(res => res.detection.box)
  //   .map((box, i) => new faceapi.draw.DrawBox(box, { label: labels[i] }))
  // const outRef = faceapi.createCanvasFromMedia(referenceImage)
  // refDrawBoxes.forEach(drawBox => drawBox.draw(outRef))

  // saveFile('referenceImage.jpg', (outRef).toBuffer('image/jpeg'));

  // const queryDrawBoxes = resultsQuery.map(res => {
  //   const bestMatch = faceMatcher.findBestMatch(res.descriptor)
  //   return new faceapi.draw.DrawBox(res.detection.box, { label: bestMatch.toString() })
  // })
  // const outQuery = faceapi.createCanvasFromMedia(queryImage)
  // queryDrawBoxes.forEach(drawBox => drawBox.draw(outQuery))
  // saveFile('queryImage.jpg', (outQuery).toBuffer('image/jpeg'))
  // console.log('done, saved results to out/queryImage.jpg')
}

module.exports = {
    detect: main,
    match
};

// https://npm.runkit.com/node-webcam
// https://github.com/justadudewhohacks/face-api.js/blob/master/examples/examples-nodejs/faceDetection.ts