#ifndef DETECTOR_H
#define DETECTOR_H

#include "bbox.h"
#include <edgetpu.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <vector>
#include <chrono>
#include <iostream>


class Detector
{
public:
    Detector();
    const char* GetVersion();

    // Construção interpretador
    bool BuildInterpreter(const char* _modelPath);

    // Rodar inferência
    std::unique_ptr<std::vector<Bbox>> RunInference(const std::vector<uint8_t>& inputData, std::chrono::duration<double, std::milli>& timeSpan);
private:

    // Declaraçao operadores
    std::unique_ptr<tflite::FlatBufferModel> _model;
    tflite::ops::builtin::BuiltinOpResolver* _resolver;
    std::unique_ptr<tflite::Interpreter> _interpreter;
    std::shared_ptr<edgetpu::EdgeTpuContext> _edgeTpu;

    //Declação Tensores
    TfLiteTensor* _outputLocations = nullptr;
    TfLiteTensor* _outputClasses = nullptr;
    TfLiteTensor* _outputScores = nullptr;
    TfLiteTensor* _numDetections = nullptr;

    // Threshold
    float score_threshold_ = 0.5;

    std::vector<int> input_tensor_shape;
    size_t input_array_size = 1;

    // Funções internas para construação do modelo e grafo de inferência
    bool BuildEdgeTpuInterpreter(const char* modelPath);
    float* GetTensorData(TfLiteTensor& tensor, const int index = 0);
    TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src);

    // Input do Modelo
    float _inputHeight;
    float _inputWidth;
    float _inputChannels;

};

#endif // DETECTOR_H
