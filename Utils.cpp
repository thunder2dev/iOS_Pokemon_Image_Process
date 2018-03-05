//
//  Utils.cpp
//  MonsterS
//
//  Created by n01192 on 9/16/16.
//  Copyright © 2016 n01192. All rights reserved.
//

#include "Utils.hpp"


#import <opencv2/imgcodecs/ios.h>
#import <opencv2/imgproc.hpp>
#import <opencv2/xphoto.hpp>

using namespace std;
using namespace cv;


void Utils::saveDebugImage(const cv::Mat image, std::string prefix){
 
    return;
    static int imageIndex = 0;
    static bool isPrintBase = false;
    
    cv::Mat imagecopy;
    image.copyTo(imagecopy);
    
    
    UIImage* mm = MatToUIImage(imagecopy);
    
    NSArray * paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString * basePath = ([paths count] > 0) ? [paths objectAtIndex:0] : nil;
    
    if(!isPrintBase){
        NSLog(@"save path: %@", basePath);
        isPrintBase = true;
    }
    
    NSData * binaryImageData = [NSData dataWithData:UIImageJPEGRepresentation(mm, 0.8)];
    
    
    NSString* fileNameNum = [NSString stringWithFormat:@"%d", imageIndex];
    NSString* fileName = [[fileNameNum stringByAppendingString:[NSString stringWithCString:prefix.c_str() encoding:[NSString defaultCStringEncoding]]] stringByAppendingString:@".jpg"];
    NSString* fullPath = [[basePath stringByAppendingPathComponent: @"debug"] stringByAppendingPathComponent:fileName];
    
    NSFileManager *fileManager = [NSFileManager defaultManager];
    if([fileManager fileExistsAtPath:fullPath]){
        [fileManager removeItemAtPath:fullPath error:nil];
    }
    
    
    
    [binaryImageData writeToFile:fullPath atomically:YES];
    
    imageIndex += 1;
    
}