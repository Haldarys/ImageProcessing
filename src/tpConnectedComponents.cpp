#include "tpConnectedComponents.h"
#include <cmath>
#include <algorithm>
#include <tuple>
#include <vector>
#include <map>
#include <stack>
using namespace cv;
using namespace std;

bool inImage(Mat image, Point2i p){
    if(p.x < image.cols && p.x >= 0 && p.y < image.rows && p.y >= 0){
        return true;
    }
    return false;
}

void LabelCCForPixel(Mat image, Mat ccMat, Point2i p, int ccVal){
    vector<Point2i> directions = {{-1,0}, {0,-1}, {0,1}, {1,0}};
    int pVal = image.at<int>(p.y, p.x);
    std::stack<Point2i> pile;
    pile.push(p);
    // Pile de pixels à traiter
    while(!pile.empty()){
        Point2i pAct = pile.top();
        pile.pop();
        // On vérifie si le pixel actuel n'a pas déjà de composante connexe
        if(ccMat.at<int>(pAct.y, pAct.x) == 0 && image.at<int>(pAct.y, pAct.x) != 0){
            ccMat.at<int>(pAct.y, pAct.x) = ccVal;
            // On parcours les points adjacents
            for(Point2i direction: directions){
                Point2i neighbour = direction + pAct;
                int neighbourVal = image.at<int>(neighbour.y, neighbour.x);
                // On vérifie si le point fait partie de la composante connexe et si oui l'ajoute dans la pile
                if(inImage(image, neighbour) && neighbourVal == pVal){
                    //std::cout << "Ajout d'un voisin" << std::endl;
                    //ccMat.at<int>(neighbour.y, neighbour.x) = ccVal;
                    pile.push(neighbour);
                }
            }
        }
    }
}

/**
    Performs a labeling of image connected component with 4 connectivity
    with a depth-first exploration.
    Any non zero pixel of the image is considered as present.
*/
cv::Mat ccLabel(cv::Mat image)
{
    Mat res = Mat::zeros(image.rows, image.cols, CV_32SC1); // 32 int image
    /********************************************
                YOUR CODE HERE
    *********************************************/
    int ccActuelle = 1;
    // On parcours toute l'image
    for(int y=0; y < res.rows; y++){
        for(int x=0; x < res.cols; x++){
            // Si le pixel actuel n'a pas encore de CC on la crée
            if(res.at<int>(y, x) == 0 && image.at<int>(y, x) != 0){
                Point2i pAct = {x, y};
                LabelCCForPixel(image, res, pAct, ccActuelle);
                ccActuelle++;
                //std::cout << ccActuelle << std::endl;
            }
        }
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Deletes the connected components (4 connectivity) containg less than size pixels.
*/
cv::Mat ccAreaFilter(cv::Mat image, int size)
{
    //Mat res = Mat::zeros(image.rows, image.cols, image.type());
    Mat res = ccLabel(image);
    assert(size>0);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    vector<int> CC;
    for(int y=0; y < res.rows; y++){
        for(int x=0; x < res.cols; x++){
            // Si la CC n'est pas dans la liste CC
            if(!(std::find(CC.begin(), CC.end(), res.at<int>(y, x)) != CC.end())){
                CC.push_back(res.at<int>(y, x));
            }
        }
    }
    int nbCC = *std::max_element(CC.begin(), CC.end());
    int* sizeCC = new int[nbCC]{0};
    for(int y=0; y < res.rows; y++){
        for(int x=0; x < res.cols; x++){
            if(res.at<int>(y, x) != 0){
                int ccAct = res.at<int>(y, x);
                sizeCC[ccAct-1]++;
            }
        }
    }
    for(int y=0; y < res.rows; y++){
        for(int x=0; x < res.cols; x++){
            if(res.at<int>(y, x) != 0){
                int ccAct = res.at<int>(y, x);
                if(sizeCC[ccAct-1] < size){
                    res.at<int>(y, x) = 0;
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
    Performs a labeling of image connected component with 4 connectivity using a
    2 pass algorithm.
    Any non zero pixel of the image is considered as present.
*/
cv::Mat ccTwoPassLabel(cv::Mat image)
{
    Mat res = Mat::zeros(image.rows, image.cols, CV_32SC1); // 32 int image
    /********************************************
                YOUR CODE HERE
    *********************************************/
  
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}