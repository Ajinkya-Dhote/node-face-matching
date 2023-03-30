const faceApiService = require("./faceapiService");

const express = require("express");
const fileUpload = require("express-fileupload");

const app = express();
const port = process.env.PORT || 3000;

app.use(fileUpload());

app.post("/upload", async (req, res) => {
  const { file } = req.files;
  const result = await faceApiService.detect(file.data);

  res.json({
    detectedFaces: result.length,
  });
});

app.post("/matcher", async (req, res) => {
  const { file1, file2 } = req.files;
  const result = await faceApiService.match(file1.data, file2.data);
  res.json({
   "result": result
  });
});

app.listen(port, () => {
  console.log("Server started on port" + port);
});