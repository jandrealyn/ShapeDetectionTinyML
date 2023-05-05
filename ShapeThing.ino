#include <TensorFlowLite_ESP32.h>
#include <EloquentTinyML.h>

#include <tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h>
#include <tensorflow/lite/experimental/micro/micro_error_reporter.h>
#include <tensorflow/lite/experimental/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "eloquent.h"
#include "eloquent_tinyml/tensorflow.h"
#include "eloquent/vision/camera/wrover.h"

#include "esp_camera.h"
#include "img_converters.h"
#include "Arduino.h"

#include "model.h"

#define _OPEN_SYS_ITOA_EXT

const uint16_t imageWidth = 160;
const uint16_t imageHeight = 120;

//eloquent::vision::camera::wrover<imageWidth, imageHeight> tf;
Eloquent::TinyML::TensorFlow::MutableTensorFlow<IN, OUT, ARENA> tf

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::ops::micro::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize];

// array to map gesture index to a name
const char* CLASSES[] = {
  "Square", 
  "Triangle", 
  "Star",
  "Circle"
};

int shapeCount = 0; // initialize the person counter
static const int MAX_PEOPLE = 4; // to store the start time of each detected person
char timeFrame[100]; // to store the label for each detected person
unsigned long startTime[MAX_PEOPLE];
char shapeName[MAX_PEOPLE][4]; 

#define NUM_CLASSES (sizeof(CLASSES) / sizeof(CLASSES[0]))

void setup() {
  Serial.begin(115200);
  delay(3000);
  // configure camera
  camera.grayscale();
  camera.qqvga();

  while (!camera.begin())
      Serial.println("Cannot init camera");

    // configure a threshold for "robust" person detection
    // if no threshold is set, "person" would be detected everytime
    // person_score > not_person_score, even if just by 1
    // by trial and error, considering that scores range from 0 to 255,
    // a threshold of 190-200 dramatically reduces the number of false positives
  
  tf.setDetectionAbsoluteThreshold(190);
  tf.begin();

    // abort if an error occurred on the detector
  while (!tf.isOk()) {
      Serial.print("Detector init error: ");
      Serial.println(tf.getErrorMessage());

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  // capture an image from the camera
    camera.capture();

    // detect shapes in the captured image
    tf.detect(camera);

    // check if any shapes were detected
    if (tf.getShapeCount() > 0) {
        // get the bounding box for each detected shape
        for (int i = 0; i < tf.getShapeCount(); i++) {
            Rectangle r = tf.getShapeRect(i);
            // crop the image to the bounding box of the detected shape
            camera.crop(r.x, r.y, r.width, r.height);
            // resize the image to the required input size for the model
            camera.resize(imageWidth, imageHeight);

            // copy the image data to the input tensor of the model
            for (int j = 0; j < imageWidth * imageHeight; j++) {
                tflInputTensor->data.f[j] = static_cast<float>(camera.getPixels()[j]) / 255.0f;
            }

            // run inference on the input tensor
            tflInterpreter->Invoke();

            // get the index of the highest output value, which corresponds to the predicted shape
            int maxIndex = 0;
            for (int j = 0; j < NUM_CLASSES; j++) {
                if (tflOutputTensor->data.f[j] > tflOutputTensor->data.f[maxIndex]) {
                    maxIndex = j;
                }
            }

            // store the name and start time of the detected shape
            strcpy(shapeName[shapeCount], CLASSES[maxIndex]);
            startTime[shapeCount] = millis();

            // increment the shape count
            shapeCount++;
            if (shapeCount >= MAX_PEOPLE) {
                shapeCount = 0;
            }

            // draw a rectangle around the detected shape and display its name
            camera.drawRect(r.x, r.y, r.width, r.height, RGB_COLOR_GREEN);
            sprintf(timeFrame, "%s %dms", CLASSES[maxIndex], millis());
            camera.drawText(r.x, r.y - 10, timeFrame, RGB_COLOR_GREEN, FONT_FACE_TERMINUS_6X12);
        }
    }

    // display the captured image on the screen
    camera.drawFrame();
}
  
}