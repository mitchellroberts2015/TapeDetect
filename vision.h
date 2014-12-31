#ifndef VISION_H
#define VISION_H
using namespace cv;

int goalIsHot();
Mat filter(Mat& src, Scalar minColor, Scalar maxColor);
Vec3i findRect(Mat& src, Scalar minColor, Scalar maxColor, int minRad, int maxRad);

#endif // VISION_H
