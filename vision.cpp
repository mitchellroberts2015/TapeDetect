#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "vision.h"
#include <iostream>
#include <math.h>
#include <string.h>
using namespace cv;
using namespace std;


Mat filter(Mat& src, Scalar minColor, Scalar maxColor)
{
    assert(src.type() == CV_8UC3);

    Mat filtered;
    inRange(src, minColor, maxColor, filtered);
    return filtered;
}

int thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                    {
                        Rect bound = boundingRect(approx);
                        double ratio = abs(1 - (double)bound.height / bound.width);
                        if (ratio>3&&ratio<8)
                            squares.push_back(approx);
                            return;
                    }
                }
            }
        }
    }
}


// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, CV_AA);
    }

    imshow(wndname, image);
}

static RotatedRect bestRect(const vector<vector<Point> >& squares)
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        RotatedRect cont = fitEllipse(squares[i]);
        Size2f contSize;
        contSize = cont.size;
        if (contSize.height/contSize.width>2||i == squares.size()-1)
            return cont;
    }
    return fitEllipse(squares[1]);
}


int goalIsHot()
{
    vector<vector<Point> > squares;

    VideoCapture stream1(0);
    Mat image;
    stream1.read(image);
    Mat imageBW = image;

        if( image.empty() )
        {
            cout << "Couldn't load image" << endl;
            return -1;
        }
    cvtColor(image,image,CV_BGR2HSV);
    imageBW = filter(image, Scalar(0,0,220), Scalar(255,255,255));
    cvtColor(imageBW,image,CV_GRAY2BGR);

    findSquares(image, squares);
    if (squares.size()>0)
        putText(image, "hot", Point(200,200), FONT_HERSHEY_COMPLEX, 3, Scalar(0,0,255));
    else
        putText(image, "cold", Point(200,200), FONT_HERSHEY_COMPLEX, 3, Scalar(255,0,0));
    drawSquares(image, squares);
    waitKey(30);

    return squares.size()>0;
}
