#include "serializer.h"

std::string Serializer::WriteJson(std::vector<Bbox> Object)
{
    std::stringstream ss;

    ss << "{\"detections\": [";
    for(std::vector<Bbox>::iterator it = Object.begin(); it != Object.end();)
    {
        ss << "{\"classe\":\""<< it->class_id;
        ss << "\",\"confidence\":\"" << it->confidence;
        ss << "\",\"boxes\":";
        ss << "{\"x\":\"" << it->x;
        ss << "\",\"y\":" << it->y;
        ss << ",\"w\":" << it->width;
        ss << ",\"h\":" << it->height;
        ss << "}";
        if(++it == Object.end()){
           ss<<"}";
        }else {
           ss<<"},";
        }
    }
    ss << "]";
    ss << "}";

    return ss.str();
}
