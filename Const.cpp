#include"Const.h"

/*
int image_ratio;
int image_width;
int image_height;
int anchorX;
int anchorY;*/

int greenLineYOffset;

namespace cf {
    
    float getRatio(){
        float ratio = relativeScreenWidth/RESIZE_WIDTH;
        return ratio;
    }

    
    int getGreenLineWidth(){
        float ratio = getRatio();
        return static_cast<int>(greenLineWidth/ratio);
    }
    
    void getGreenLineLtPos(cv::Point& pos){
        float ratio = getRatio();
        pos = cv::Point((greenCenterX - greenLineWidth/2)/ratio, (greenCenterY - greenLineHeight/2)/ratio);
    }
    
    void getWatchCenter(cv::Point& center, float& R) {
        float ratio = getRatio();
        center = cv::Point(watchCenterX / ratio, watchCenterY / ratio) + cv::Point(0, greenLineYOffset);
        R = watchR / ratio;
        
    }
    
    int getTinyR(){
        float ratio = getRatio();
        return static_cast<int>(tinyDotR/ratio);
    }
    
    void getPockerRoi(cv::Rect& roi){
        
        float ratio = getRatio();
        
        cv::Point tl(pockerX/ratio, pockerY/ratio + greenLineYOffset);
        
        float width = pockerWidth/ratio;
        float height = pockerHeight/ratio;
        
        roi = cv::Rect(tl.x, tl.y, width, height);
        
    }
    
    void getNameRoi(cv::Rect& roi){
        
        float ratio = getRatio();
        cv::Point tl(nameX/ratio, (nameY/ratio + greenLineYOffset));
        
        float width = nameWidth/ratio;
        float height = nameHeight/ratio;
        
        roi = cv::Rect(tl.x, tl.y, width, height);
    }
    
    void getUpDustRoi(cv::Rect& roi){
        
        float ratio = getRatio();
        cv::Point tl(upDustX/ratio, (upDustY/ratio + greenLineYOffset));
        
        float width = upDustWidth/ratio;
        float height = upDustHeight/ratio;
        
        roi = cv::Rect(tl.x, tl.y, width, height);
    }
    
    void getUpCandyRoi(cv::Rect& roi){
        
        float ratio = getRatio();
        cv::Point tl(upCandyX/ratio, (upCandyY/ratio + greenLineYOffset));
        
        float width = upCandyWidth/ratio;
        float height = upCandyHeight/ratio;
        
        roi = cv::Rect(tl.x, tl.y, width, height);
    }
    
    
    
    

    void getHpRoi(cv::Rect& roi){
        
        float ratio = getRatio();
        cv::Point tl(hpX/ratio, (hpY/ratio + greenLineYOffset));
        
        float width = hpWidth/ratio;
        float height = hpHeight/ratio;
        
        roi = cv::Rect(tl.x, tl.y, width, height);
    }
    
    void getCpRoi(cv::Rect& roi){        
        float ratio = getRatio();
        cv::Point tl(cpX/ratio, (cpY/ratio + greenLineYOffset));
        
        float width = cpWidth/ratio;
        float height = cpHeight/ratio;
        
        roi = cv::Rect(tl.x, tl.y, width, height);
    }
    
    
}



































