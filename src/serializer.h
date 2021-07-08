#ifndef SERIALIZER_H
#define SERIALIZER_H
#include "bbox.h"
#include <sstream>
#include <fstream>
#include <memory>
#include <vector>

class Serializer
{
public:
    static std::string WriteJson(std::vector<Bbox> Object);
};

#endif // SERIALIZER_H
