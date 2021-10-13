#include "tpHistogram.h"
#include <cmath>
#include <algorithm>
#include <tuple>
using namespace cv;
using namespace std;

/**
    Inverse a grayscale image with float values.
    for all pixel p: res(p) = 1.0 - image(p)
*/
Mat inverse(Mat image)
{
    // clone original image
    Mat res = image.clone();
      /********************************************
                YOUR CODE HERE
    *********************************************/
    res = 1.0 - res;
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Thresholds a grayscale image with float values.
    for all pixel p: res(p) =
        | 0 if image(p) <= lowT
        | image(p) if lowT < image(p) <= hightT
        | 1 otherwise
*/
Mat threshold(Mat image, float lowT, float highT)
{
    Mat res = image.clone();
    assert(lowT <= highT);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    for(int y = 0; y < res.rows; y++) {
        for(int x = 0; x < res.cols; x++) {
                if(res.at<float>(y, x) <= lowT){
                    res.at<float>(y, x) = 0;
                } else if (res.at<float>(y, x) >= highT) {
                    res.at<float>(y, x) = 1;
                }
        }
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Quantize the input float image in [0,1] in numberOfLevels different gray levels.
    eg. for numberOfLevels = 3 the result should be for all pixel p: res(p) =
        | 0 if image(p) < 1/3
        | 1/2 if 1/3 <= image(p) < 2/3
        | 1 otherwise
*/
Mat quantize(Mat image, int numberOfLevels)
{
    Mat res = image.clone();
    assert(numberOfLevels>0);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    for(int y = 0; y < res.rows; y++) {
        for(int x = 0; x < res.cols; x++) {
            float step = 1.0f / numberOfLevels;
            for(int lvl = 0; lvl < numberOfLevels; lvl++) {
                float lvlMaxVal = ((float)lvl+1) * step;
                float lvlMinVal = (float)lvl * step;
                if(res.at<float>(y, x) < lvlMaxVal && res.at<float>(y, x) >= lvlMinVal) {
                    res.at<float>(y, x) = (1.0f / (numberOfLevels-1)) * lvl; //step * lvl;
                }
            }
        }
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Normalize a grayscale image with float values
    Target range is [minValue, maxValue].
*/
Mat normalize(Mat image, float minValue, float maxValue)
{
    Mat res = image.clone();
    assert(minValue <= maxValue);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    float fmin = 1;
    float fmax = 0;
    for(int y = 0; y < res.rows; y++) {
        for(int x = 0; x < res.cols; x++) {
            if(res.at<float>(y, x) < fmin){
                fmin = res.at<float>(y, x);
            }
            if(res.at<float>(y, x) > fmax){
               fmax = res.at<float>(y, x); 
            }
        }
    }
    for(int y = 0; y < res.rows; y++) {
        for(int x = 0; x < res.cols; x++) {
            res.at<float>(y, x) = (res.at<float>(y, x) - fmin) * ((maxValue - minValue) / (fmax - fmin)) + minValue;
        }
    }

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}



/**
    Equalize image histogram with unsigned char values ([0;255])
*/
Mat equalize(Mat image)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
    *********************************************/
    int cumul[256] {0};
    for(int y = 0; y < image.rows; y++)
    {
        for(int x = 0; x < image.cols; x++)
        {
            cumul[res.at<uchar>(y, x)]++;
        }
    }
    for(int i =1; i<256;i++)
    {
        cumul[i]+=cumul[i-1];
    }
    
    int size = res.rows*res.cols; 
    for(int y = 0; y < image.rows; y++)
        for(int x = 0; x < image.cols; x++)
            res.at<uchar>(y, x) =  round(255.f/size*cumul[res.at<uchar>(y, x)]);
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;

}

float moyenneClasse(Mat image, uchar minClasse, uchar maxClasse){
    int countPixelInClass = 0;
    int somme = 0;
    for(int y = 0; y < image.rows; y++) {
        for(int x = 0; x < image.cols; x++) {
            uchar pixelValue = image.at<uchar>(y, x);
            if(pixelValue >= minClasse && pixelValue < maxClasse){
                somme += pixelValue;
                countPixelInClass += 1;
            }
        }
    }
    if(countPixelInClass == 0) return 0;
    float moy = (float)somme / countPixelInClass;
    return moy;
}

float variance(Mat image, uchar minClasse, uchar maxClasse){
    int countPixelInClass = 0;
    // Calcul de la moyenne
    uchar moy = moyenneClasse(image, minClasse, maxClasse);
    // Calcul de la variance
    int somme = 0;
    for(int y = 0; y < image.rows; y++) {
        for(int x = 0; x < image.cols; x++) {
            uchar pixelValue = image.at<uchar>(y, x);
            if(pixelValue >= minClasse && pixelValue < maxClasse){
                uchar ecart = (pixelValue - moy);
                somme += ecart * ecart;
                countPixelInClass += 1;
            }
        }
    }
    float variance = (float) somme / countPixelInClass;
    return variance; 
}

float probaClasse(Mat image, uchar minClasse, uchar maxClasse){
    int count = 0;
    for(int y = 0; y < image.rows; y++) {
        for(int x = 0; x < image.cols; x++) {
            uchar pixelValue = image.at<uchar>(y, x);
            if(pixelValue >= minClasse && pixelValue < maxClasse){
                count += 1;
            }
        }
    }
    if(count == 0) return 0;
    return (float)count / image.total();
}

/**
    Compute a binarization of the input float image using an automatic Otsu threshold.
    Input image is of type unsigned char ([0;255])
*/
Mat thresholdOtsu(Mat image)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
    *********************************************/
    float valMax = 0;
    uchar thresholdVal = 0;
    std::cout << 0 << std::endl;
    for(uchar i=1; i < 255; i++){
        Mat temp = res.clone();
        //temp = threshold(temp, i, i);
        //float varianceInf = variance(temp, 0, i);
        //float varianceSup = variance(temp, i+1, 255);
        float moyInf = moyenneClasse(temp, 0, i);
        float moySup = moyenneClasse(temp, i+1, 255);
        float probaInf = probaClasse(temp, 0, i);
        float probaSup = probaClasse(temp, i+1, 255);
        //std::cout << "moyInf " << moyInf << " moySup " << moySup << " probaInf " << probaInf << " probaSup " << probaSup << std::endl;

        //float val = (probaInf * varianceInf) + (probaSup * varianceSup);
        float val = probaInf * probaSup * ((moyInf - moySup) * (moyInf - moySup));
        if(val > valMax) {
            thresholdVal = i;
            valMax = val;
        }
        //std::cout << val << std::endl;
    }
    //std::cout << valMax << std::endl;
    std::cout << (int)thresholdVal << std::endl;
    /********************************************
                END OF YOUR CODE
    *********************************************/
    //res = threshold(res, thresholdVal, thresholdVal);
    for(int y = 0; y < res.rows; y++) {
        for(int x = 0; x < res.cols; x++) {
                if(res.at<uchar>(y, x) <= thresholdVal){
                    res.at<uchar>(y, x) = 0;
                } else if (res.at<uchar>(y, x) >= thresholdVal) {
                    res.at<uchar>(y, x) = 255;
                }
        }
    }
    return res;
}
