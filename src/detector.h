#ifndef DETECTOR_H
#define DETECTOR_H

#include "bbox.h"
#include "serializer.h"
#include <edgetpu.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

#define LOG(x) std::cerr

#if defined(__GNUC__)
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
    #define CDECL __attribute__((cdecl))
#endif

class Detector
{
public:
    Detector();
    const char* GetVersion();

    // Tensor inputs
    const int Width() const;
    const int Height() const;
    const int Channels() const;

    // Construção interpretador
    void BuildInterpreter(const char* _modelPath);

    // Rodar inferência
    std::unique_ptr<std::vector<Bbox>> RunInference(const std::vector<uint8_t>& inputData);
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

    std::vector<int> input_tensor_shape;

    // Funções internas para construação do modelo e grafo de inferência
    void BuildEdgeTpuInterpreter(const char* modelPath);

    // Input do Modelo
    float _inputHeight;
    float _inputWidth;
    float _inputChannels;
};

extern "C" {
    // chamadas nativas
    EXPORT Detector* CDECL ClassificadorDetectorEnv();
    EXPORT Detector* CDECL ClassificadorDetector(const char* weight_path);
    EXPORT char* CDECL RunInference(Detector* handle, unsigned char* image, size_t imgSize);
    EXPORT char* CDECL RunInferenceROI(Detector* handle, unsigned char* image, size_t imgSize, int x, int y, int width, int height);
    EXPORT void CDECL ClassificadorDestroy(Detector* handle);
    EXPORT void CDECL FreeResult(char* result);
    EXPORT const char* CDECL GetVersion();
}

#endif // DETECTOR_H
