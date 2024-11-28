// Wait until the ONNX Runtime Web library is fully loaded
if (typeof ort === 'undefined') {
  console.error("Error: ONNX Runtime Web (ort) is not available. Make sure it's loaded properly.");
} else {
  // Your previous code to initialize the app
  const MODEL_PATH = "best1.onnx?v=" + new Date().getTime(); // Replace with actual model path
  const IMAGE_WIDTH = 640;
  const IMAGE_HEIGHT = 640;

  const imageUpload = document.getElementById("image-upload");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  // Set canvas dimensions (initially set to the target size)
  canvas.width = IMAGE_WIDTH;
  canvas.height = IMAGE_HEIGHT;

  const YOLO_CLASSES = ['Bonnet', 'Bumper', 'Dickey', 'Door', 'Fender', 'Light', 'Windshield'];

  // Load the ONNX model
  let session;
  (async function loadModel() {
    try {
      console.log("Loading ONNX model...");
      session = await ort.InferenceSession.create(MODEL_PATH);

      console.log("Model output names:", session.outputNames);

      console.log("ONNX model loaded successfully");
    } catch (error) {
      console.error("Error loading the ONNX model:", error);
    }
  })();

  // Handle image upload
  imageUpload.addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (file) {
      const img = new Image();
      img.src = URL.createObjectURL(file);
      img.onload = () => {
        console.log("Image loaded:", img);
        // Resize the canvas to match the image aspect ratio while keeping the target size
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, img.width, img.height);
        processImageFrame(img);
      };
    }
  });

  // Process the image
  // Process the image
async function processImageFrame(img) {
  console.log("Processing image...");

  // Preprocess the image
  const imageData = ctx.getImageData(0, 0, img.width, img.height);
  const inputTensor = preprocessImage(imageData);

  if (!session) {
    console.error("Inference session is not loaded.");
    return;
  }

  console.log("Running model inference...");

  // Run model inference with the correct input name
  try {
    const results = await session.run({
      "input.1": inputTensor // Use the correct input name from the model
    });

    // Log the results to check if model output is as expected
    console.log("Model inference results:", results);
    console.log("Output keys:", Object.keys(results));

    // Now using the correct output key ("1769") for the model predictions
    const outputData = results["1769"];  // Use the correct key "1769"
    if (!outputData) {
      console.error("Output tensor is missing or does not have the expected property.");
      return;
    }

    console.log("Output tensor shape:", outputData.dims);
    console.log("Output tensor data:", outputData.data);

    const detections = processResults(outputData.data, img.width, img.height);

    if (detections.length === 0) {
      console.log("No detections found.");
    }

    // Apply NMS to filter overlapping boxes
    const filteredDetections = nonMaxSuppression(detections, 0.013, 2); // Confidence threshold: 0.5, IoU threshold: 0.4

    // Draw bounding boxes
    drawDetections(filteredDetections, img);
  } catch (error) {
    console.error("Error during inference:", error);
  }
}

  // Preprocess the image for YOLOv8 input
  function preprocessImage(imageData) {
    const { data, width, height } = imageData;
    const input = new Float32Array(IMAGE_WIDTH * IMAGE_HEIGHT * 3);

    const scaleX = IMAGE_WIDTH / width;
    const scaleY = IMAGE_HEIGHT / height;

    // Resize to IMAGE_WIDTH x IMAGE_HEIGHT
    let index = 0;
    for (let i = 0; i < IMAGE_HEIGHT; i++) {
      for (let j = 0; j < IMAGE_WIDTH; j++) {
        const x = Math.min(width - 1, Math.floor(j / scaleX));
        const y = Math.min(height - 1, Math.floor(i / scaleY));
        const pixelIndex = (y * width + x) * 4;
        input[index++] = data[pixelIndex] / 255.0;  // Normalize the R channel
        input[index++] = data[pixelIndex + 1] / 255.0;  // Normalize the G channel
        input[index++] = data[pixelIndex + 2] / 255.0;  // Normalize the B channel
      }
    }

    return new ort.Tensor("float32", input, [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH]);
  }

  // Adjusting processResults function based on new model output structure
  function processResults(outputData, originalWidth, originalHeight) {
    const detections = [];
    const NUM_CLASSES = 7; // Adjust based on your model

    // Log the output shape and data for debugging
    console.log("Model output data length:", outputData.length);

    const numPredictions = outputData.length / (NUM_CLASSES + 5);
    console.log("Number of Predictions:", numPredictions);

    const scaleX = IMAGE_WIDTH / originalWidth;
    const scaleY = IMAGE_HEIGHT / originalHeight;

    for (let i = 0; i < numPredictions; i++) {
      const startIdx = i * (NUM_CLASSES + 5);

      const x_center = outputData[startIdx];
      const y_center = outputData[startIdx + 1];
      const width = outputData[startIdx + 2];
      const height = outputData[startIdx + 3];
      const confidence = outputData[startIdx + 4];

      if (confidence > 1) { // Confidence threshold
        const classProbabilities = outputData.slice(
          startIdx + 5,
          startIdx + 5 + NUM_CLASSES
        );
        const classID = classProbabilities.indexOf(
          Math.max(...classProbabilities)
        );

        if (classID >= 0 && classID < NUM_CLASSES) {
          const xMin = Math.max(0, (x_center - width / 2) * scaleX);
          const yMin = Math.max(0, (y_center - height / 2) * scaleY);
          const boxWidth = Math.min(width * scaleX);
          const boxHeight = Math.min(height * scaleY);

          detections.push({
            xMin,
            yMin,
            boxWidth,
            boxHeight,
            confidence,
            classID,
          });
        }
      }
    }

    return detections;
  }

  // Non-Maximum Suppression to remove overlapping boxes
  function nonMaxSuppression(detections, iouThreshold) {
    

    // Sort detections by confidence
    detections.sort((a, b) => b.confidence - a.confidence);

    const filteredDetections = [];

    while (detections.length > 0) {
      const bestDetection  = detections.shift(); // Get the highest confidence detection
      filteredDetections.push(bestDetection);

      // Filter out detections that overlap with the current one
      detections = detections.filter(det => {
        const iou = calculateIoU(bestDetection, det);
        return iou < iouThreshold;
      });
    }

    return filteredDetections;
  }

  // Calculate Intersection over Union (IoU)
  function calculateIoU(boxA, boxB) {
    const x1 = Math.max(boxA.xMin, boxB.xMin);
    const y1 = Math.max(boxA.yMin, boxB.yMin);
    const x2 = Math.min(boxA.xMin + boxA.boxWidth, boxB.xMin + boxB.boxWidth);
    const y2 = Math.min(boxA.yMin + boxA.boxHeight, boxB.yMin + boxB.boxHeight);

    const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = boxA.boxWidth * boxA.boxHeight;
    const areaB = boxB.boxWidth * boxB.boxHeight;

    const unionArea = areaA + areaB - intersectionArea;

    return intersectionArea / unionArea;
  }

  // Draw bounding boxes
  function drawDetections(detections, img) {
    detections.forEach((detection) => {
      const { xMin, yMin, boxWidth, boxHeight, confidence, classID } = detection;
      const classLabel = YOLO_CLASSES[classID] || "Unknown";

      const scaledX = xMin * (canvas.width / img.width);
      const scaledY = yMin * (canvas.height / img.height);
      const scaledWidth = boxWidth * (canvas.width / img.width);
      const scaledHeight = boxHeight * (canvas.height / img.height);

      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;
      ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
      ctx.fillStyle = "red";
      ctx.font = "16px Arial";
      ctx.fillText(
        `${classLabel}: ${(confidence * 100).toFixed(2)}%`,
        scaledX,
        scaledY > 10 ? scaledY - 5 : 10
      );
    });
  }
}
