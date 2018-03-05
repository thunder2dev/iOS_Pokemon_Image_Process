#pragma once

#include<opencv2/core/core.hpp>

const int watchCenterX = 540;
const int watchCenterY = 683;
const int watchR = 438;
const int pockerX = 233;
const int pockerY = 385;
const int pockerWidth = 615;
const int pockerHeight = 483;
const int nameX = 44;
const int nameY = 868;
const int nameWidth = 994;
const int nameHeight = 78;

const int cpX = 357;
const int cpY = 130;
const int cpWidth = 300;
const int cpHeight = 80;
const int hpX = 383;
const int hpY = 1000;
const int hpWidth = 312;
const int hpHeight = 55;
const int greenCenterX = 539;
const int greenCenterY = 991;
const int greenLineWidth = 541;
const int greenLineHeight = 20;

const int upDustX = 594;
const int upDustY = 1531;
const int upDustWidth = 125;
const int upDustHeight = 44;

const int upCandyX = 812;
const int upCandyY = 1532;
const int upCandyWidth = 50;
const int upCandyHeight = 50;


const int tinyDotR = 9;


const int relativeScreenWidth = 1080;
const int relativeScreenHeight = 1920;
const float RESIZE_WIDTH = 800;

/*
extern int image_ratio;
extern int image_width;
extern int image_height;*/

extern int greenLineYOffset;

namespace cf {
    void getWatchCenter(cv::Point& center, float& R);
    int getGreenLineWidth();
    int getTinyR();
    void getPockerRoi(cv::Rect& roi);
    void getNameRoi(cv::Rect& roi);
    void getHpRoi(cv::Rect& roi);
    void getCpRoi(cv::Rect& roi);
    void getGreenLineLtPos(cv::Point& pos);
    void getUpDustRoi(cv::Rect& roi);
    void getUpCandyRoi(cv::Rect& roi);
}


