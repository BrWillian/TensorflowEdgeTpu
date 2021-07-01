#ifndef BBOX_H
#define BBOX_H


class Bbox
{
public:
    int class_id;
    float confidence = 0.0;
    float x = 0.0;
    float y = 0.0;
    float width = 0.0;
    float height = 0.0;
    float center_x = 0.0;
    float center_y = 0.0;
};

#endif // BBOX_H
