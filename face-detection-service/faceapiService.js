const path = require("path");
const modelPathRoot = "./models";
// import { faceDetectionOptions } from './commons';

const saveFile = require("./commons/saveFile")

const canvas = require('canvas');
const faceapi = require('face-api.js');
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });


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
}

module.exports = {
    detect: main,
    match
};

// https://npm.runkit.com/node-webcam
// https://github.com/justadudewhohacks/face-api.js/blob/master/examples/examples-nodejs/faceDetection.ts